# CollabVR

**CollabVR: Collaborative Video Reasoning with Vision-Language and Video Generation Models**
Joowon Kim\*, Seungho Shin\*, Joonhyung Park, Eunho Yang (\*equal contribution)

**Accepted at NeurIPS 2026.**

[[Paper (arXiv)](https://arxiv.org/abs/2605.08735)] · [[Project page](https://joow0n-kim.github.io/collabvr-project-page/)]

## Overview

![CollabVR concept](assets/concept.png)

A Vision-Language Model (VLM) is strong at logical reasoning but weak at visual simulation; a Video Generation Model (VGM) simulates short clips faithfully but lacks reasoning. This mismatch surfaces as two recurring failures on goal-directed tasks: **long-horizon drift** when a single prompt specifies a multi-step task, and **mid-clip simulation errors** that propagate through subsequent frames. CollabVR couples the two models in a closed loop: the VLM plans the immediate next action, inspects the clip the VGM produces, and routes test-time compute across qualitatively distinct recovery strategies (re-generation, action splitting) matched to the diagnosed failure.

## Pipeline

![CollabVR pipeline](assets/pipeline.png)

At each step *t*, the VLM emits a single action *a<sub>t</sub>* conditioned on the current frame, task prompt, and history. The VGM renders a clip *c<sub>t</sub>*, which the VLM verifier accepts or rejects with a diagnosed failure mode *d*. Two complementary modules consume that signal:

- **M1 — Progressive Planning.** Per-state adaptive selection of the number of sub-steps *N*. Atomic transformations stay at *N=1*; multi-step tasks expand *N>1* only when the verifier indicates that the single-shot stream cannot complete the task.
- **M2 — Verification + Re-generation.** On reject, prompt evolution updates *a<sub>t</sub>* using the diagnosis *d* and the clip is re-sampled up to a budget *M*. If all *M* attempts are rejected, the framework routes to a recovery strategy chosen from *d*.

The two modules address different failure modes (long-horizon drift vs. mid-clip error) and are activated independently per state.

In code, `CollabVRPipeline.run(sample, output_dir)` implements the loop below. The same loop, prompts, and hyperparameters are used for both benchmarks; only the sample loader and the clip length differ (see Configuration).

```
H = []                                      # accepted clips
f = I_0                                     # current conditioning frame
for t in 1..N_max:
    a_t = planner(I_0, q, history)          # progressive planner
    for j in 1..M:
        c_t = generate(f, a_t)              # image-to-video clip
        v, d = verifier(I_0, q, history, c_t)   # step verifier
        if v == accept:
            H.append(c_t)
            f = last_frame(c_t)
            if task_complete:
                break_outer
            break
        else:
            a_t = a_t + " " + d.suggestion  # prompt evolution
return concatenate(H)
```

## What is released

This repository contains the reference implementation used for the paper's
**VBVR-Wan2.2** experiments on **Gen-ViRe** (72 samples) and **VBVR-Bench**
(500 samples), together with the Pass@1 / Pass@k baselines and the scoring
scripts. The planner and verifier prompts are the ones listed in the paper's
appendix.

The Veo 3.1 results in the paper were obtained through Google's closed API
with the same planner, verifier, and prompts; that backend is not part of
this release.

```
.
├── prompts/
│   ├── step_planner.txt     # progressive planner prompt (paper appendix)
│   └── step_verifier.txt    # step verifier prompt (paper appendix)
├── pipeline.py              # CollabVRPipeline: planner + verifier + prompt evolution
├── wan_loader.py            # VBVR-Wan2.2 image-to-video loader (single A100, no offload)
├── utils.py                 # VLM response parsing, API-key loading, ffmpeg/OpenCV helpers
├── run.py                   # CLI: run CollabVR on Gen-ViRe or VBVR-Bench
├── baselines.py             # CLI: Pass@1 / Pass@k baselines on the same VGM
├── eval.py                  # CLI: score generated mp4s
├── requirements.txt
└── assets/
```

## Setup

```bash
# Python >= 3.10
pip install -r requirements.txt
sudo apt-get install -y ffmpeg

# Gemini API key (Gemini 2.5 Pro is used as planner, verifier, and judge).
# Either export it ...
export GOOGLE_API_KEY=<YOUR_API_KEY>
# ... or write it to a file and pass --api-key-file (keep this file out of version control).
echo "<YOUR_API_KEY>" > api_key.txt

# Pre-download VBVR-Wan2.2 once
huggingface-cli download Video-Reason/VBVR-Wan2.2 --local-dir /path/to/VBVR-Wan2.2

# (Optional) pre-download VBVR-Bench-Data for the VBVR runner
huggingface-cli download Video-Reason/VBVR-Bench-Data --repo-type dataset \
    --local-dir /path/to/VBVR-Bench-Data
```

## Configuration

`pipeline.PipelineConfig` holds the paper-default hyperparameters:

| Parameter | Default | Notes |
|---|---|---|
| `N_max`               | 3   | maximum number of planning steps |
| `M`                   | 3   | per-step attempt budget (1 original + up to 2 regens) |
| `seconds_first`       | 6.0 | first-step clip duration (Gen-ViRe) |
| `seconds_subseq`      | 3.0 | subsequent / regen clip duration (Gen-ViRe) |
| `num_inference_steps` | 20  | VBVR-Wan2.2 diffusion steps |
| `guidance_scale`      | 5.0 | VBVR-Wan2.2 CFG |
| `vlm_model`           | `gemini-2.5-pro` | planner + verifier |
| `partial_regen_threshold` | 0.2 | partial re-gen if `good_fraction > 0.2`, else full |

On VBVR-Bench we follow the official protocol and generate every clip at the
length of the sample's ground-truth video (`sample["clip_seconds"]`, derived
from `ground_truth.mp4`), so `seconds_first` / `seconds_subseq` are not used there.

On a single A100 with `pipe.to("cuda")`, expect roughly 5.5 min per 6-second
clip at 480p; clip generation dominates end-to-end runtime.

## Run CollabVR

The `--benchmark` switch only changes the sample loader. Both modes call the
same `CollabVRPipeline.run(sample, output_dir)`. Omit `--api-key-file` to use
`$GOOGLE_API_KEY`.

### Gen-ViRe

```bash
python run.py --benchmark genvire \
    --hf-dataset ciciciciliu/Gen-Vire_ds \
    --vgm-path /path/to/VBVR-Wan2.2 \
    --api-key-file api_key.txt \
    --output-dir results/genvire
```

Each dataset item is expected to provide
`{main_category, sub_category, id, image, i2v_prompt}`.

### VBVR-Bench

```bash
python run.py --benchmark vbvr \
    --data-root /path/to/VBVR-Bench-Data \
    --vgm-path /path/to/VBVR-Wan2.2 \
    --api-key-file api_key.txt \
    --output-dir results/vbvr
```

`--data-root` is the `VBVR-Bench-Data` download; samples are read from its
`VBVR-Bench.json`, which points to per-sample files:

```
VBVR-Bench-Data/
    VBVR-Bench.json
    <split>/<task>/<idx>/
        first_frame.png     # input image I_0
        prompt.txt          # task prompt q
        ground_truth.mp4    # sets the clip length for this sample
```

`--split In-Domain_50` or `--split Out-of-Domain_50` restricts the run to one split.

### Output

```
<output-dir>/
    results.json                                 # per-sample log
    <subdir-1>/<output-id>.mp4                   # final concatenated trajectory
    <subdir-2>/<output-id>.mp4
    ...
```

`results.json` records the steps actually taken (`t`, accept/reject, regen
attempts, instruction text) and the per-sample VLM-call / VGM-clip counts.
Intermediate per-step and per-attempt clips are kept next to the final mp4.

## Baselines

The two baselines reported in the paper share the same generator; only the
inference strategy differs. Each baseline clip has the length of the N=1
branch of CollabVR: 6 s on Gen-ViRe, the ground-truth length on VBVR-Bench.

```bash
# Pass@1 (single-shot, seed=1)
python baselines.py --method pass1 --benchmark genvire \
    --hf-dataset ciciciciliu/Gen-Vire_ds \
    --vgm-path /path/to/VBVR-Wan2.2 \
    --output-dir results/pass1_genvire

# Pass@k (k generations + VLM-based selection)
python baselines.py --method passk --k 4 --benchmark genvire \
    --hf-dataset ciciciciliu/Gen-Vire_ds \
    --vgm-path /path/to/VBVR-Wan2.2 \
    --api-key-file api_key.txt \
    --output-dir results/pass4_genvire
```

The same `--benchmark vbvr --data-root /path/to/VBVR-Bench-Data` switch runs
the baselines on VBVR-Bench.

## Score

Both benchmarks are scored with their official evaluators; `eval.py` only
wires our output directory into them.

```bash
# Gen-ViRe: official rubric-based judge (Gemini 2.5 Pro), 3 judge runs averaged
git clone https://github.com/L-CodingSpace/GVR /path/to/GVR
python eval.py --mode genvire --videos-dir results/genvire \
    --genvire-repo /path/to/GVR --model-name collabvr \
    --api-key-file api_key.txt --num-runs 3

# VBVR-Bench: official deterministic rule-based scorer
python eval.py --mode vbvr --videos-dir results/vbvr \
    --evaluator /path/to/VBVR-EvalKit/evaluate_vbvr.py \
    --output vbvr_scores.json
```

For Gen-ViRe, `eval.py` symlinks `results/genvire/<main>/<sub>/<id>.mp4` into
`0_generated_videos/<model-name>/<model-name>_<i>/` inside the GVR checkout for
`i < num-runs` and then runs its `step1`–`step4` scripts, which judge every
linked video once per run folder. The averaged summary is written to
`0_generated_videos/<model-name>/<model-name>_summary.csv` in that checkout.
The VBVR mode assumes the upstream script accepts `--generated-dir` and
`--output`; adjust `_eval_vbvr` in `eval.py` if its interface differs.

## Reproducibility

| | |
|---|---|
| Python | >= 3.10 |
| CUDA   | >= 12.1 recommended |
| GPU    | A100 80 GB (no CPU offload). On 24 GB cards switch to `enable_sequential_cpu_offload` in `wan_loader.load_wan_pipeline`. |
| Disk   | ~60 GB free for the VGM checkpoint, generated mp4s, and intermediate clips. |

Estimated wall-clock at the default settings (single A100):

| | per-sample | total |
|---|---|---|
| Gen-ViRe (72 samples)    | ~14 min  | ~17 h |
| VBVR-Bench (500 samples) | ~3.5 min | ~30 h |

Notes:

- Single-GPU; for multi-GPU sample sharding, partition the sample list across processes externally.
- Re-running skips samples whose final mp4 already exists. Delete the mp4 to force a re-run.
- The VLM is called with `temperature=0.2` for planning/verification and `temperature=0.0` for Pass@k selection and judging, so runs are close to but not bit-exact reproducible.

## Citation

```bibtex
@inproceedings{kim2026collabvr,
  title     = {CollabVR: Collaborative Video Reasoning with Vision-Language and Video Generation Models},
  author    = {Kim, Joowon and Shin, Seungho and Park, Joonhyung and Yang, Eunho},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
