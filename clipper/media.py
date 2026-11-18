"""ffmpeg / ffprobe helpers. Every call uses an argument list (never a shell string)."""

from __future__ import annotations

import functools
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


class MediaError(RuntimeError):
    """ffmpeg/ffprobe failed or produced unusable output."""


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


def _run(args: list[str], *, capture: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            args,
            check=False,
            capture_output=capture,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise MediaError(f"executable not found: {args[0]} (install ffmpeg or set FFMPEG_BIN)") from exc


def _parse_fps(rate: str) -> float:
    num, _, den = rate.partition("/")
    try:
        return float(num) / float(den) if den and float(den) else float(num)
    except ValueError:
        return 30.0


def probe(path: Path, ffprobe: str = "ffprobe") -> MediaInfo:
    result = _run([ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    if result.returncode != 0:
        raise MediaError(f"ffprobe could not read {path.name}: {result.stderr.strip()[:300]}")
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"{path.name} has no video stream")
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    if duration <= 0:
        raise MediaError(f"{path.name} has no usable duration")
    return MediaInfo(
        duration=duration,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=_parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate") or "30/1"),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


@functools.lru_cache(maxsize=8)
def _encoders(ffmpeg: str) -> str:
    result = _run([ffmpeg, "-hide_banner", "-encoders"])
    return result.stdout or ""


def pick_encoder(ffmpeg: str, preference: str = "auto") -> str:
    """Return ``h264_nvenc`` when it actually encodes on this machine, else ``libx264``.

    Listing an encoder is not enough: the GPU driver can still refuse to open a
    session. So a one-frame probe encode decides.
    """
    if preference == "libx264":
        return "libx264"
    if preference == "auto" and "h264_nvenc" not in _encoders(ffmpeg):
        return "libx264"
    if preference in ("auto", "h264_nvenc"):
        probe_cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                     "color=c=black:s=320x240:d=0.1", "-c:v", "h264_nvenc", "-f", "null", "-"]
        if _run(probe_cmd, timeout=20).returncode == 0:
            return "h264_nvenc"
        log.warning("h264_nvenc is listed but failed to open; falling back to libx264")
    return "libx264"


def video_encode_args(encoder: str) -> list[str]:
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "22", "-b:v", "0",
                "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p"]


def run_ffmpeg(args: list[str], *, cwd: Path | None = None, timeout: float | None = None) -> None:
    """Run ffmpeg and raise with the tail of stderr on failure."""
    result = subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-12:])
        raise MediaError(f"ffmpeg failed (exit {result.returncode}):\n{tail}")
