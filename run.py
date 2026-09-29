#!/usr/bin/env python3
"""
Run the CollabVR pipeline on either benchmark.

The pipeline itself is benchmark-agnostic; this script differs only in how
it loads samples (an HuggingFace dataset vs. a local directory layout).

Usage:
    # Gen-ViRe (HuggingFace dataset):
    python run.py --benchmark genvire --vgm-path /path/to/VBVR-Wan2.2 \\
        --hf-dataset ciciciciliu/Gen-Vire_ds --api-key-file api_key.txt \\
        --output-dir results/genvire

    # VBVR-Bench (local VBVR-Bench-Data root containing VBVR-Bench.json):
    python run.py --benchmark vbvr --vgm-path /path/to/VBVR-Wan2.2 \\
        --data-root /path/to/VBVR-Bench-Data --api-key-file api_key.txt \\
        --output-dir results/vbvr
"""
import argparse
import json
import time
from pathlib import Path

from PIL import Image
from google import genai

from pipeline import CollabVRPipeline, PipelineConfig
from utils import load_api_key, video_frame_count
from wan_loader import load_wan_pipeline


# ---------------------------------------------------------------------------
# Sample loaders. Each yields a list of dicts with the keys expected by
# CollabVRPipeline.run(): sample_key, input_image, task_prompt,
# output_subdir, output_id.
# ---------------------------------------------------------------------------

def _load_genvire(hf_dataset: str, limit: int = 0,
                  category_filter: str = None):
    """Gen-ViRe: HuggingFace dataset with main_category / sub_category / id."""
    from datasets import load_dataset
    ds = load_dataset(hf_dataset)["train"]
    out = []
    for s in ds:
        if category_filter and s.get("main_category") != category_filter:
            continue
        out.append({
            "sample_key": f"{s['main_category']}-{s['sub_category']}-{s['id']}",
            "input_image": s["image"],
            "task_prompt": s.get("i2v_prompt", s.get("prompt", "")),
            "output_subdir": Path(s["main_category"]) / s["sub_category"],
            "output_id": str(s["id"]),
        })
    return out[:limit] if limit else out


def _load_vbvr(data_root: Path, fps: int, split: str = None, limit: int = 0):
    """VBVR-Bench: samples listed in <data_root>/VBVR-Bench.json.

    Following the official protocol, every generated clip of a sample is as
    long as its ground-truth video, so `clip_seconds` is derived from
    ground_truth.mp4 (frame count / fps).
    """
    entries = json.loads((data_root / "VBVR-Bench.json").read_text())
    out = []
    for e in entries:
        rel = Path(e["first_frame_path"]).parent          # <split>/<task>/<idx>
        if split and rel.parts[0] != split:
            continue
        gt = data_root / e["ground_truth_video_path"]
        out.append({
            "sample_key": str(rel),
            "input_image": Image.open(data_root / e["first_frame_path"]),
            "task_prompt": (data_root / e["prompt_path"]).read_text().strip(),
            "output_subdir": rel.parent,
            "output_id": rel.name,
            "clip_seconds": video_frame_count(gt) / fps,
        })
    return out[:limit] if limit else out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["genvire", "vbvr"], required=True,
                   help="genvire: HuggingFace-style Gen-ViRe; vbvr: local VBVR-Bench-Data root.")
    p.add_argument("--hf-dataset", default=None,
                   help="HuggingFace dataset id (Gen-ViRe).")
    p.add_argument("--data-root", default=None,
                   help="Local VBVR-Bench-Data root (VBVR-Bench).")
    p.add_argument("--vgm-path", required=True,
                   help="HuggingFace repo id or local path of VBVR-Wan2.2 (image-to-video generator).")
    p.add_argument("--api-key-file", default=None,
                   help="Text file containing the Gemini API key "
                        "(default: read $GOOGLE_API_KEY).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--category", default=None,
                   help="Optional category filter (Gen-ViRe).")
    p.add_argument("--split", default=None, choices=["In-Domain_50", "Out-of-Domain_50"],
                   help="Optional split filter (VBVR-Bench).")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    api_key = load_api_key(args.api_key_file)

    # 1. Load samples
    if args.benchmark == "genvire":
        if not args.hf_dataset:
            raise SystemExit("--hf-dataset required for Gen-ViRe")
        samples = _load_genvire(args.hf_dataset, limit=args.limit,
                                category_filter=args.category)
    else:
        if not args.data_root:
            raise SystemExit("--data-root required for VBVR-Bench")
        samples = _load_vbvr(Path(args.data_root), fps=PipelineConfig().fps,
                             split=args.split, limit=args.limit)
    print(f"[data] {len(samples)} samples loaded for {args.benchmark}",
          flush=True)

    # 2. Load models
    vgm = load_wan_pipeline(args.vgm_path)
    vlm = genai.Client(api_key=api_key)
    pipe = CollabVRPipeline(vgm, vlm, PipelineConfig())

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Run
    log = []
    t0 = time.time()
    for i, s in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] {s['sample_key']}", flush=True)
        try:
            r = pipe.run(s, output_dir)
            log.append({
                "key": r.sample_key,
                "final_video": str(r.final_video) if r.final_video else None,
                "clips_generated": r.clips_generated,
                "vlm_calls": r.vlm_calls,
                "error": r.error,
                "steps": [
                    {"t": st.t, "accepted": st.accepted,
                     "regen_attempts": st.regen_attempts,
                     "instruction": st.instruction[:120]}
                    for st in r.steps
                ],
            })
        except Exception as e:
            log.append({"key": s["sample_key"], "error": str(e)})
        (output_dir / "results.json").write_text(json.dumps(log, indent=2))

    print(f"\n[done] {len(samples)} samples in {(time.time()-t0)/60:.1f} min.", flush=True)
    print(f"        log: {output_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
