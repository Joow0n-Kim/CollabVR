"""
VBVR-Wan2.2 image-to-video pipeline loader and generation helper.

Targets a single A100 (no CPU offload). For smaller GPUs (e.g. 24 GB cards)
swap `pipe.to("cuda")` for `pipe.enable_sequential_cpu_offload()` and remove
the scheduler patch below.
"""
import gc
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video


# 832 x 480 = 480p area cap. Output resolution preserves input aspect ratio.
MAX_RESOLUTION_AREA_480P = 832 * 480
RESAMPLING_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def load_wan_pipeline(model_path: str) -> WanImageToVideoPipeline:
    """Load VBVR-Wan2.2 onto the current CUDA device.

    `model_path` can be a HuggingFace repo id or a local checkpoint directory.
    """
    print(f"[VBVR-Wan2.2 loader] model = {model_path}", flush=True)
    t0 = time.time()

    vae = AutoencoderKLWan.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_path, vae=vae, torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")

    # The UniPC scheduler builds `sigmas` on CPU even when the rest of the
    # pipeline is on CUDA, which trips a CPU/CUDA mismatch in `torch.stack`
    # at step time. Patch `set_timesteps` to move sigmas to CUDA.
    _orig_set_timesteps = pipe.scheduler.set_timesteps
    def _set_timesteps_cuda(num_inference_steps, device=None):
        _orig_set_timesteps(num_inference_steps, device="cuda")
        pipe.scheduler.sigmas = pipe.scheduler.sigmas.to("cuda")
    pipe.scheduler.set_timesteps = _set_timesteps_cuda

    pipe.vae.enable_tiling()
    print(f"[VBVR-Wan2.2 loader] ready in {time.time() - t0:.1f}s", flush=True)
    return pipe


def _compute_num_frames(seconds: float, fps: int) -> int:
    """Frame count satisfying (n - 1) % 4 == 0 (Wan latent constraint)."""
    raw = round(seconds * fps)
    return (raw // 4) * 4 + 1


def _aspect_ratio_resize(pipe, image: Image.Image, max_area: int) -> Tuple[Image.Image, int, int]:
    aspect = image.height / image.width
    mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    h = round(np.sqrt(max_area * aspect)) // mod * mod
    w = round(np.sqrt(max_area / aspect)) // mod * mod
    return image.resize((int(w), int(h)), RESAMPLING_LANCZOS), int(h), int(w)


def generate_wan_video(
    pipe: WanImageToVideoPipeline,
    image: Image.Image,
    prompt: str,
    output_path: Path,
    *,
    seconds: float = 6.0,
    fps: int = 16,
    num_inference_steps: int = 20,
    guidance_scale: float = 5.0,
    seed: int = 1,
) -> dict:
    """Generate one image-to-video clip and save to `output_path`."""
    t_start = time.time()
    src_area = image.width * image.height
    max_area = min(src_area, MAX_RESOLUTION_AREA_480P)
    image_resized, target_h, target_w = _aspect_ratio_resize(pipe, image, max_area)
    num_frames = _compute_num_frames(seconds, fps)

    print(f"  source {image.width}x{image.height} -> {target_w}x{target_h}  "
          f"frames={num_frames} steps={num_inference_steps} seed={seed}", flush=True)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output = pipe(
        image=image_resized,
        prompt=prompt,
        height=target_h, width=target_w,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(output.frames[0], str(output_path), fps=fps)

    print(f"  saved {output_path.name}  ({time.time() - t_start:.1f}s)", flush=True)
    return {
        "output_path": str(output_path),
        "target_width": target_w, "target_height": target_h,
        "num_frames": num_frames,
        "generation_seconds": time.time() - t_start,
    }
