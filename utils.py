"""Shared helpers: VLM response parsing, API-key loading, and ffmpeg/OpenCV utilities."""
import json
import os
import subprocess
from pathlib import Path
from typing import List, Optional

import cv2
from PIL import Image


# ---------------------------------------------------------------------------
# VLM helpers
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> dict:
    """Parse a JSON object from a VLM reply, tolerating ``` fences."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


def load_api_key(path: Optional[str] = None) -> str:
    """Return the Gemini API key from `path` (if given) or $GOOGLE_API_KEY."""
    if path:
        key = Path(path).read_text().strip()
    else:
        key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "No API key: pass --api-key-file or set the GOOGLE_API_KEY environment variable.")
    os.environ["GOOGLE_API_KEY"] = key
    return key


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------


def video_frame_count(video_path: Path) -> int:
    """Number of frames in a video (used to match VBVR-Bench ground-truth clip length)."""
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        raise RuntimeError(f"could not read frame count of {video_path}")
    return n


def extract_last_frame(video_path: Path) -> Image.Image:
    """Return the last frame of a video as a PIL image."""
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(n_frames - 1, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read last frame of {video_path}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def extract_frame_at_fraction(video_path: Path, fraction: float) -> Image.Image:
    """Return the frame at `fraction` (0.0–1.0) into the clip."""
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = max(0, min(n_frames - 1, int(n_frames * fraction)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {target} of {video_path}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def trim_video(src: Path, dst: Path, fraction: float):
    """Keep the first `fraction` of `src`, write to `dst`. Assumes ffmpeg in PATH."""
    cap = cv2.VideoCapture(str(src))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1e-6)
    cap.release()
    keep = max(0.1, duration * fraction)
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-t", f"{keep:.3f}",
         "-c:v", "libx264", "-an", str(dst)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def concatenate_clips(clips: List[Path], dst: Path) -> bool:
    """Concatenate a list of mp4 clips into one. Returns True on success."""
    if not clips:
        return False
    list_file = dst.parent / f"_concat_{dst.stem}.txt"
    list_file.write_text("\n".join(f"file '{c.resolve()}'" for c in clips))
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c:v", "libx264", "-an", str(dst)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    list_file.unlink(missing_ok=True)
    return proc.returncode == 0
