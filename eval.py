#!/usr/bin/env python3
"""
Score the generated mp4s with each benchmark's official evaluator.

    Gen-ViRe:    rubric-based VLM judge (Gemini 2.5 Pro) shipped with the
                 official Gen-ViRe repository. We link our videos into the
                 layout its scripts expect and run step1..step4 there, once
                 per judge run (paper default: 3 runs, averaged).

    VBVR-Bench:  deterministic, rule-based comparison against ground-truth
                 videos, delegated to the official VBVR evaluator.

Usage:
    # Gen-ViRe (official evaluation scripts):
    python eval.py --mode genvire --videos-dir results/genvire \\
        --hf-dataset ciciciciliu/Gen-Vire_ds --api-key-file api_key.txt \\
        --genvire-repo /path/to/Gen-ViRe --model-name collabvr --num-runs 3

    # VBVR-Bench (official rule-based evaluator):
    python eval.py --mode vbvr --videos-dir results/vbvr \\
        --evaluator /path/to/VBVR-EvalKit/evaluate_vbvr.py \\
        --output vbvr_scores.json
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from utils import load_api_key


def _run(cmd, cwd):
    print(f"$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def _eval_genvire(args):
    """Delegate to the official Gen-ViRe scripts (github.com/L-CodingSpace/GVR).

    Those scripts evaluate five run folders 0_generated_videos/<model>/<model>_<i>.
    We link the same videos into the first `num_runs` folders so each video is
    judged `num_runs` times, and leave the remaining folders empty.
    """
    from datasets import load_dataset

    repo = Path(args.genvire_repo).resolve()
    code_dir = repo / "code"
    if not (code_dir / "step3_batch_eval.py").exists():
        sys.exit(f"official Gen-ViRe code not found under {code_dir}")

    api_key = load_api_key(args.api_key_file)
    os.environ["VLM_API_KEY"] = api_key       # read by the per-task judge scripts

    # 1. Link results/<main>/<sub>/<id>.mp4 into the official layout.
    videos_dir = Path(args.videos_dir).resolve()
    model_dir = repo / "0_generated_videos" / args.model_name
    ds = load_dataset(args.hf_dataset)["train"]
    missing = 0
    for i in range(args.num_runs):
        for s in ds:
            src = videos_dir / s["main_category"] / s["sub_category"] / f"{s['id']}.mp4"
            if not src.exists():
                missing += 1
                continue
            dst = model_dir / f"{args.model_name}_{i}" / s["main_category"] / s["sub_category"] / f"{s['id']}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
    if missing:
        print(f"[warn] {missing // args.num_runs} samples have no video under {videos_dir}")

    # 2. Official pipeline: last frames -> eval configs -> judge -> summary.
    common = ["--video-model", args.model_name, "--mode", "batch"]
    _run([sys.executable, "step1_extract_all_frames.py", *common], code_dir)
    _run([sys.executable, "step2_generate_configs.py", *common], code_dir)
    for i in range(args.num_runs, 5):          # no videos there: skip judging
        shutil.rmtree(model_dir / "temp_eval_configs" / f"{args.model_name}_{i}", ignore_errors=True)
    _run([sys.executable, "step3_batch_eval.py", "--video-model", args.model_name], code_dir)
    _run([sys.executable, "step4_generate_summary.py", *common,
          "--output-base", str(repo / "0_generated_videos")], code_dir)
    print(f"summary: {model_dir / f'{args.model_name}_summary.csv'}")


def _eval_vbvr(args):
    evaluator = Path(args.evaluator).resolve()
    if not evaluator.exists():
        sys.exit(f"upstream evaluator not found at {evaluator}")
    # Adjust the flags here if the upstream interface differs.
    _run([sys.executable, str(evaluator),
          "--generated-dir", str(Path(args.videos_dir).resolve()),
          "--output", args.output], evaluator.parent)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["genvire", "vbvr"], required=True)
    p.add_argument("--videos-dir", required=True)
    # Gen-ViRe
    p.add_argument("--hf-dataset", default="ciciciciliu/Gen-Vire_ds")
    p.add_argument("--api-key-file", default=None,
                   help="Gemini API key file (default: read $GOOGLE_API_KEY).")
    p.add_argument("--genvire-repo", default=None,
                   help="Path to a clone of the official Gen-ViRe repository.")
    p.add_argument("--model-name", default="collabvr",
                   help="Folder name used inside the Gen-ViRe repo.")
    p.add_argument("--num-runs", type=int, default=3,
                   help="Independent judge runs to average (paper: 3).")
    # VBVR-Bench
    p.add_argument("--evaluator", default=None,
                   help="Path to the official VBVR rule-based evaluator script.")
    p.add_argument("--output", default=None, help="Score file (VBVR-Bench).")
    args = p.parse_args()

    if args.mode == "genvire":
        if not args.genvire_repo:
            sys.exit("--genvire-repo is required for Gen-ViRe")
        _eval_genvire(args)
    else:
        if not (args.evaluator and args.output):
            sys.exit("--evaluator and --output are required for VBVR-Bench")
        _eval_vbvr(args)


if __name__ == "__main__":
    main()
