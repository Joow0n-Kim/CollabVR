"""
CollabVR pipeline.

Inputs to one sample:
    I_0  : input image
    q    : task prompt
    N_max: max planning steps
    M    : per-step attempt budget

Loop:
    H = []                            # history of accepted clips
    f = I_0                           # current conditioning frame
    for t in 1..N_max:
        a_t = plan(I_0, q, H)         # progressive planner
        for j in 1..M:
            c_t = generate(f, a_t)    # image-to-video generator
            (v, d) = verify(I_0, q, H, c_t)   # step verifier
            if v == accept:
                H.append(c_t)
                f = last_frame(c_t)
                if task_complete: return concat(H)
                break
            else:
                a_t = evolve(a_t, d)  # prompt evolution: append diagnosis
    return concat(H)

The same loop is used for both benchmarks.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from google import genai
from google.genai import types
from PIL import Image

from utils import (concatenate_clips, extract_frame_at_fraction, extract_last_frame,
                   parse_json_response, trim_video)
from wan_loader import generate_wan_video


PROMPT_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.txt").read_text()


@dataclass
class PipelineConfig:
    # Paper-default budgets
    N_max: int = 3                       # max planning steps
    M: int = 3                           # per-step attempt budget (1 + up to 2 regens)
    # Per-step clip duration (seconds) on Gen-ViRe. On VBVR-Bench every clip
    # instead matches the ground-truth video length (sample["clip_seconds"]).
    seconds_first: float = 6.0           # step 1
    seconds_subseq: float = 3.0          # step 2+ and re-generations
    # Image-to-video generator hyperparameters
    fps: int = 16
    num_inference_steps: int = 20
    guidance_scale: float = 5.0
    seed: int = 1
    # VLM
    vlm_model: str = "gemini-2.5-pro"
    # Partial re-generation: keep the verified-good prefix when good_fraction
    # exceeds this threshold; otherwise re-generate from the step's input frame.
    partial_regen_threshold: float = 0.2


@dataclass
class StepLog:
    t: int
    instruction: str
    target_state: str
    clip_path: Optional[Path] = None
    accepted: bool = False
    regen_attempts: int = 0
    # verifier diagnosis for each rejected attempt, keyed by attempt index j
    diagnoses: Dict[int, dict] = field(default_factory=dict)


@dataclass
class SampleLog:
    sample_key: str
    final_video: Optional[Path] = None
    steps: List[StepLog] = field(default_factory=list)
    clips_generated: int = 0
    vlm_calls: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# VLM call
# ---------------------------------------------------------------------------

def _pil_to_part(img: Image.Image) -> types.Part:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")


def _call_vlm(client: genai.Client, model: str, prompt_text: str,
              images: Optional[List[Image.Image]] = None,
              video_path: Optional[Path] = None,
              max_retries: int = 3) -> dict:
    parts = []
    if video_path is not None:
        parts.append(types.Part.from_bytes(
            data=video_path.read_bytes(), mime_type="video/mp4"))
    if images:
        for img in images:
            parts.append(_pil_to_part(img))
    parts.append(prompt_text)

    for attempt in range(max_retries):
        try:
            r = client.models.generate_content(
                model=model, contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0.2, response_mime_type="application/json"),
            )
            return parse_json_response(r.text)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# CollabVR
# ---------------------------------------------------------------------------

class CollabVRPipeline:
    """CollabVR pipeline."""

    def __init__(self, vgm, vlm_client: genai.Client,
                 config: Optional[PipelineConfig] = None):
        self.vgm = vgm
        self.vlm = vlm_client
        self.cfg = config or PipelineConfig()
        self.planner_prompt = _load_prompt("step_planner")
        self.verifier_prompt = _load_prompt("step_verifier")

    # -------- VLM roles --------

    def _plan(self, f: Image.Image, q: str,
              completed: List[dict], step_number: int) -> dict:
        """π_plan(f, q, H) -> next action prompt + task_complete flag."""
        history = "\n".join(
            f"Step {s['step']}: {s['instruction']} -> {s['outcome']}"
            for s in completed
        ) or "(none)"
        text = (
            f"{self.planner_prompt}\n\n---\n"
            f"TASK_PROMPT: {q}\n\n"
            f"STEP_NUMBER: {step_number} of {self.cfg.N_max} maximum\n\n"
            f"COMPLETED_STEPS:\n{history}\n\n"
            f"IMPORTANT: You have a maximum of {self.cfg.N_max} steps total. "
            f"If this is step {self.cfg.N_max} (the final step), your "
            f"instruction MUST complete the entire remaining task in this "
            f"single step.\n\n"
            f"CURRENT_IMAGE is attached. Plan the next step."
        )
        return _call_vlm(self.vlm, self.cfg.vlm_model, text, images=[f])

    def _verify(self, c_t: Path, q: str,
                instruction: str, target_state: str) -> dict:
        """π_verify(I_0, q, H, c_t) -> {verdict, reason, suggestion, good_fraction}."""
        text = (
            f"{self.verifier_prompt}\n\n---\n"
            f"TASK_PROMPT: {q}\n\n"
            f"PLANNED_ACTION: {instruction}\n\n"
            f"TARGET_STATE: {target_state}\n\n"
            f"The generated video clip is attached. Watch the full motion and "
            f"evaluate whether the planned action was executed correctly."
        )
        return _call_vlm(self.vlm, self.cfg.vlm_model, text, video_path=c_t)

    # -------- Generator --------

    def _generate(self, conditioning_frame: Image.Image, action_prompt: str,
                  output_path: Path, seconds: float) -> bool:
        """g(f, a_t) -> short clip c_t saved to output_path."""
        try:
            generate_wan_video(
                pipe=self.vgm,
                image=conditioning_frame, prompt=action_prompt,
                output_path=output_path,
                seconds=seconds, fps=self.cfg.fps,
                num_inference_steps=self.cfg.num_inference_steps,
                guidance_scale=self.cfg.guidance_scale,
                seed=self.cfg.seed,
            )
            return True
        except Exception as e:
            print(f"  [generate] error: {e}", flush=True)
            return False

    # -------- Prompt evolution --------

    @staticmethod
    def _evolve(a_t: str, d: dict) -> str:
        """evolve(a_t, d): fold the verifier's diagnosis into the action prompt.

        Following the paper, the verifier's `suggestion` field is appended to
        the previous action prompt. No additional VLM call is required.
        """
        suggestion = (d or {}).get("suggestion", "") or ""
        return (a_t + " " + suggestion).strip() if suggestion else a_t

    # -------- One step --------

    @staticmethod
    def _attempt_path(sample_dir: Path, sample_id: str, t: int, j: int) -> Path:
        """Clip path for attempt j of step t (attempt 1 has no suffix)."""
        if j == 1:
            return sample_dir / f"{sample_id}_step{t}.mp4"
        return sample_dir / f"{sample_id}_step{t}_attempt{j}.mp4"

    def _partial_regen_source(self, step_log: StepLog, sample_dir: Path,
                              sample_id: str, t: int, j: int):
        """If attempt j-1 was rejected with a usable good prefix, return
        (previous clip path, good_fraction); otherwise None."""
        if j <= 1:
            return None
        d = step_log.diagnoses.get(j - 1)
        if not d:
            return None
        good = d.get("good_fraction", 0.0) or 0.0
        if good <= self.cfg.partial_regen_threshold:
            return None
        return self._attempt_path(sample_dir, sample_id, t, j - 1), good

    def _run_step(self, t: int, q: str, I0: Image.Image, f: Image.Image,
                  H: List[Path], completed: List[dict],
                  sample_dir: Path, sample_id: str,
                  log: SampleLog, clip_seconds: Optional[float] = None) -> Optional[Path]:
        """Plan + run M attempts. Return the accepted clip path, or None."""
        # Plan the next action: a_t = π_plan(f, q, H). The planner is shown the
        # current conditioning frame f (I_0 at t=1, afterwards the last frame of
        # the previous accepted clip) together with the textual history.
        plan = self._plan(f, q, completed, t)
        log.vlm_calls += 1
        if plan.get("error"):
            return None
        if plan.get("task_complete"):
            return None  # Planner declares the task is already complete

        a_t = plan.get("instruction", "")
        target_state = plan.get("target_state", "")
        step_log = StepLog(t=t, instruction=a_t, target_state=target_state)
        if clip_seconds is None:
            clip_seconds = self.cfg.seconds_first if t == 1 else self.cfg.seconds_subseq

        # Inner loop: up to M attempts per step
        for j in range(1, self.cfg.M + 1):
            # Generate clip: c_t = g(f, a_t)
            clip_path = self._attempt_path(sample_dir, sample_id, t, j)
            partial = self._partial_regen_source(step_log, sample_dir, sample_id, t, j)

            if not clip_path.exists():
                # Default: full re-generation from the step's conditioning frame.
                # When the verifier reports good_fraction > threshold, we use
                # partial re-generation: keep the verified-good prefix and
                # resume from the error frame.
                cond = f
                if partial is not None:
                    prev_path, good_fraction = partial
                    cond = extract_frame_at_fraction(prev_path, good_fraction)
                ok = self._generate(cond, a_t, clip_path, seconds=clip_seconds)
                log.clips_generated += 1
                if not ok:
                    continue

            # Verify clip: (v, d) = π_verify(I_0, q, H, c_t)
            d = self._verify(clip_path, q, a_t, target_state)
            log.vlm_calls += 1
            verdict = d.get("verdict", "accept")

            if verdict == "accept":
                # Append the accepted clip to the history H
                # If we kept a good prefix from a prior attempt, splice it in.
                kept = clip_path
                if partial is not None:
                    prev_path, good_fraction = partial
                    good_part = sample_dir / f"{sample_id}_step{t}_good_for_attempt{j}.mp4"
                    if not good_part.exists():
                        trim_video(prev_path, good_part, good_fraction)
                    merged = sample_dir / f"{sample_id}_step{t}_combined.mp4"
                    if concatenate_clips([good_part, clip_path], merged):
                        kept = merged
                step_log.clip_path = kept
                step_log.accepted = True
                step_log.regen_attempts = j - 1
                log.steps.append(step_log)
                return kept

            # Prompt evolution on rejection: a_t = evolve(a_t, d)
            step_log.regen_attempts = j
            step_log.diagnoses[j] = d
            a_t = self._evolve(a_t, d)

        # All M attempts rejected: nothing is appended to H.
        # Record the step log with the final attempt (not appended to H).
        step_log.clip_path = clip_path
        step_log.accepted = False
        log.steps.append(step_log)
        return None

    # -------- Public entry: identical for both benchmarks --------

    def run(self, sample: dict, output_dir: Path) -> SampleLog:
        """
        Run the CollabVR pipeline on a single sample.

        `sample` must provide:
            sample_key: str        a unique identifier
            input_image: PIL.Image input image I_0
            task_prompt: str       task prompt q
            output_subdir: Path    subdirectory under output_dir for this sample
            output_id: str         basename for clip files
            clip_seconds: float    (optional) fixed clip length for every step;
                                   VBVR-Bench sets this to the ground-truth
                                   video duration, Gen-ViRe leaves it unset

        Returns a SampleLog with the final concatenated mp4 path (or None on
        failure) and per-step bookkeeping.
        """
        log = SampleLog(sample_key=sample["sample_key"])
        sample_dir = output_dir / sample["output_subdir"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_id = sample["output_id"]
        final = sample_dir / f"{sample_id}.mp4"
        if final.exists():
            log.final_video = final
            return log

        I0 = sample["input_image"].convert("RGB")
        q = sample["task_prompt"]
        f = I0                  # current conditioning frame
        H: List[Path] = []      # accepted clips
        completed: List[dict] = []

        for t in range(1, self.cfg.N_max + 1):
            kept = self._run_step(t, q, I0, f, H, completed,
                                  sample_dir, sample_id, log,
                                  clip_seconds=sample.get("clip_seconds"))
            if kept is None:
                # Either planner emitted task_complete or all M attempts failed.
                break
            H.append(kept)
            f = extract_last_frame(kept)
            completed.append({
                "step": t,
                "instruction": log.steps[-1].instruction,
                "outcome": "accept",
            })

        # Concatenate accepted clips into the final video
        if H:
            ok = concatenate_clips(H, final)
            if ok:
                log.final_video = final
            else:
                log.error = "concatenation failed"
        return log

