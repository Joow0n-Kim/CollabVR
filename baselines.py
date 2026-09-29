#!/usr/bin/env python3
"""
Baselines: single-shot generation (Pass@1) and Pass@k.

Both baselines reuse the same VBVR-Wan2.2 generator so end-to-end VGM cost
is matched to CollabVR's clip count. Clip length follows the N=1 branch of
CollabVR: 6 s on Gen-ViRe, the ground-truth video length on VBVR-Bench. Selection in Pass@k is done by the
same VLM used as our verifier (Gemini 2.5 Pro), shown all k candidates in
one call and asked to pick the one that best satisfies the task.

Usage:
    # Pass@1 (single-shot generation)
    python baselines.py --method pass1 --benchmark genvire \
        --hf-dataset ciciciciliu/Gen-Vire_ds \
        --vgm-path /path/to/VBVR-Wan2.2 \
        --output-dir results/pass1_genvire

    # Pass@k (k=4)
    python baselines.py --method passk --k 4 --benchmark genvire \
        --hf-dataset ciciciciliu/Gen-Vire_ds \
        --vgm-path /path/to/VBVR-Wan2.2 \
        --api-key-file api_key.txt \
        --output-dir results/pass4_genvire
"""
import argparse
import json
import shutil
import time
from pathlib import Path
from typing import List

from google import genai
from google.genai import types

from pipeline import PipelineConfig
from run import _load_genvire, _load_vbvr
from utils import load_api_key, parse_json_response
from wan_loader import generate_wan_video, load_wan_pipeline


GENVIRE_SECONDS = PipelineConfig().seconds_first   # 6 s single clip (no decomposition)
FPS = PipelineConfig().fps
NUM_INF = 20
GUIDANCE = 5.0


def _selector_prompt() -> str:
    return (
        "You are picking the single best video that achieves the task "
        "described by TASK_PROMPT. Watch all candidate videos and select "
        "the one that best satisfies the task. Reply with strict JSON: "
        '{"best_index": <0-indexed integer>, "reason": "..."}.'
    )


def _vlm_select(client: genai.Client, model: str,
                videos: List[Path], task_prompt: str) -> int:
    parts = []
    for i, v in enumerate(videos):
        parts.append(types.Part.from_bytes(
            data=v.read_bytes(), mime_type="video/mp4"))
    parts.append(
        f"{_selector_prompt()}\n\n---\nTASK_PROMPT: {task_prompt}\n\n"
        f"Candidate videos in order are attached as the first {len(videos)} attachments."
    )
    r = client.models.generate_content(
        model=model, contents=parts,
        config=types.GenerateContentConfig(
            temperature=0.0, response_mime_type="application/json"),
    )
    obj = parse_json_response(r.text)
    return int(obj.get("best_index", 0)) % len(videos)


def _load_samples(args):
    """Same loaders as run.py; VBVR samples carry `clip_seconds` (GT length)."""
    if args.benchmark == "genvire":
        return _load_genvire(args.hf_dataset)
    return _load_vbvr(Path(args.data_root), fps=FPS)


def _clip_seconds(s: dict) -> float:
    return s.get("clip_seconds", GENVIRE_SECONDS)


def _run_pass1(args, vgm, samples, output_dir: Path):
    log = []
    for i, s in enumerate(samples, 1):
        sample_dir = output_dir / s["output_subdir"]
        final = sample_dir / f"{s['output_id']}.mp4"
        if final.exists():
            log.append({"key": s["sample_key"], "final_video": str(final)}); continue
        print(f"\n[{i}] {s['sample_key']}")
        try:
            generate_wan_video(
                pipe=vgm, image=s["input_image"].convert("RGB"),
                prompt=s["task_prompt"], output_path=final,
                seconds=_clip_seconds(s), fps=FPS, num_inference_steps=NUM_INF,
                guidance_scale=GUIDANCE, seed=1,
            )
            log.append({"key": s["sample_key"], "final_video": str(final)})
        except Exception as e:
            log.append({"key": s["sample_key"], "error": str(e)})
        (output_dir / "results.json").write_text(json.dumps(log, indent=2))


def _run_passk(args, vgm, vlm, samples, output_dir: Path, k: int):
    log = []
    for i, s in enumerate(samples, 1):
        sample_dir = output_dir / s["output_subdir"]
        final = sample_dir / f"{s['output_id']}.mp4"
        if final.exists():
            log.append({"key": s["sample_key"], "final_video": str(final)}); continue
        print(f"\n[{i}] {s['sample_key']} (k={k})")
        candidates = []
        for seed in range(1, k + 1):
            cand = sample_dir / f"{s['output_id']}_seed{seed}.mp4"
            if not cand.exists():
                try:
                    generate_wan_video(
                        pipe=vgm, image=s["input_image"].convert("RGB"),
                        prompt=s["task_prompt"], output_path=cand,
                        seconds=_clip_seconds(s), fps=FPS,
                        num_inference_steps=NUM_INF, guidance_scale=GUIDANCE,
                        seed=seed,
                    )
                except Exception as e:
                    print(f"  seed {seed} failed: {e}"); continue
            candidates.append(cand)
        if not candidates:
            log.append({"key": s["sample_key"], "error": "no candidates generated"}); continue
        try:
            best = _vlm_select(vlm, "gemini-2.5-pro",
                               candidates, s["task_prompt"])
        except Exception as e:
            print(f"  selector failed, defaulting to seed 1: {e}")
            best = 0
        shutil.copy2(candidates[best], final)
        log.append({"key": s["sample_key"], "final_video": str(final),
                    "best_seed": best + 1, "candidates": [str(c) for c in candidates]})
        (output_dir / "results.json").write_text(json.dumps(log, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["pass1", "passk"], required=True)
    p.add_argument("--k", type=int, default=4, help="k for Pass@k")
    p.add_argument("--benchmark", choices=["genvire", "vbvr"], required=True)
    p.add_argument("--hf-dataset", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--vgm-path", required=True)
    p.add_argument("--api-key-file", default=None,
                   help="Gemini API key file for the Pass@k selector VLM "
                        "(default: read $GOOGLE_API_KEY).")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = _load_samples(args)
    print(f"[data] {len(samples)} samples")

    vgm = load_wan_pipeline(args.vgm_path)
    vlm = None
    if args.method == "passk":
        api_key = load_api_key(args.api_key_file)
        vlm = genai.Client(api_key=api_key)

    t0 = time.time()
    if args.method == "pass1":
        _run_pass1(args, vgm, samples, output_dir)
    else:
        _run_passk(args, vgm, vlm, samples, output_dir, args.k)
    print(f"\n[done] {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
