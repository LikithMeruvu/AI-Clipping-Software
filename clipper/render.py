"""Render one clip: cut, jump-cut silence, smart crop, captions, audio mix and encode.

Stages
------
1. ``_cut``    one ffmpeg pass: extract the clip, drop long pauses (trim/concat), force a
               constant frame rate so audio and video stay in sync, normalise the sample rate.
2. reframe     OpenCV + MediaPipe find the speaker; the crop path is computed in Python.
3. ``_final``  one ffmpeg pass: sendcmd-driven crop -> scale -> libass captions -> optional
               logo -> encode, with loudnorm and optional ducked background music.

Per-frame pixel work stays inside ffmpeg, so Python never touches raw frames.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .captions import CaptionStyle, Line, build_ass
from .media import MediaError, MediaInfo, probe, run_ffmpeg, video_encode_args
from .reframe import LayoutPlan, analyse_layout, sendcmd_script

log = logging.getLogger(__name__)

MAX_FPS = 30
AUDIO_BITRATE = "192k"
SILENT_AUDIO = "anullsrc=channel_layout=stereo:sample_rate=48000"


@dataclass(frozen=True)
class RenderJob:
    source: Path
    output: Path
    intervals: list[tuple[float, float]]      # kept source ranges (see timeline.keep_intervals)
    captions: list[Line]                      # already on the output timeline
    style: CaptionStyle
    width: int = 1080
    height: int = 1920
    hook: str | None = None
    progress_bar: bool = False
    track_faces: bool = True
    layout: str = "auto"                      # auto | single | split
    bgm: Path | None = None
    bgm_volume: float = 0.18
    logo: Path | None = None
    broll: tuple[tuple[float, float, Path], ...] = ()    # (start, end, mp4) stock footage under the captions
    emojis: tuple[tuple[float, float, Path], ...] = ()   # (start, end, png) pop-ups above the captions
    dub: Path | None = None                   # translated speech (wav) mixed over the ducked original
    encoder: str = "libx264"
    fonts_dir: Path | None = None
    models_dir: Path = Path("data/models")
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"


@dataclass(frozen=True)
class RenderResult:
    path: Path
    duration: float
    face_tracked: bool
    encoder: str
    layout: str = "center"


def _filter_path(path: Path) -> str:
    """Escape a path for use inside an ffmpeg filter option (Windows drive colons)."""
    return str(path).replace("\\", "/").replace(":", "\\:")


def _cut(job: RenderJob, out: Path, source: MediaInfo, fps: float) -> None:
    """Extract the kept intervals into a CFR intermediate with an audio track (silent if the source has none)."""
    base = job.intervals[0][0]
    total = sum(b - a for a, b in job.intervals)
    args = [job.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{base:.3f}", "-i", str(job.source)]
    if not source.has_audio:
        args += ["-f", "lavfi", "-i", SILENT_AUDIO]

    if len(job.intervals) == 1:
        a, b = job.intervals[0]
        args += ["-t", f"{b - a:.3f}", "-vf", f"fps={fps:g}", "-map", "0:v:0"]
        if source.has_audio:
            args += ["-map", "0:a:0", "-af", "aresample=48000"]
        else:
            args += ["-map", "1:a:0", "-shortest"]
    else:
        parts: list[str] = []
        labels: list[str] = []
        for i, (a, b) in enumerate(job.intervals):
            ra, rb = a - base, b - base
            parts.append(f"[0:v]trim=start={ra:.3f}:end={rb:.3f},setpts=PTS-STARTPTS[v{i}]")
            if source.has_audio:
                parts.append(f"[0:a]atrim=start={ra:.3f}:end={rb:.3f},asetpts=PTS-STARTPTS,aresample=48000[a{i}]")
                labels.append(f"[v{i}][a{i}]")
            else:
                labels.append(f"[v{i}]")
        n = len(job.intervals)
        if source.has_audio:
            parts.append("".join(labels) + f"concat=n={n}:v=1:a=1[vc][ac]")
            audio_map = "[ac]"
        else:
            parts.append("".join(labels) + f"concat=n={n}:v=1:a=0[vc]")
            audio_map = "1:a:0"
        parts.append(f"[vc]fps={fps:g}[vout]")
        args += ["-filter_complex", ";".join(parts), "-map", "[vout]", "-map", audio_map]
        if not source.has_audio:
            args += ["-t", f"{total:.3f}"]

    args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
             "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ac", "2", str(out)]
    run_ffmpeg(args, timeout=900)


def _final(job: RenderJob, work: Path, cut: Path, cut_info: MediaInfo, plan: LayoutPlan | None) -> None:
    W, H = job.width, job.height
    inputs = ["-i", str(cut)]
    next_input = 1
    bgm_idx = logo_idx = None
    if job.bgm:
        inputs += ["-stream_loop", "-1", "-i", str(job.bgm)]
        bgm_idx = next_input
        next_input += 1
    if job.logo:
        inputs += ["-loop", "1", "-i", str(job.logo)]
        logo_idx = next_input
        next_input += 1
    broll_idx = []
    for _, _, clip_path in job.broll:
        inputs += ["-stream_loop", "-1", "-i", str(clip_path)]
        broll_idx.append(next_input)
        next_input += 1
    emoji_idx = []
    for _, _, png in job.emojis:
        inputs += ["-loop", "1", "-i", str(png)]
        emoji_idx.append(next_input)
        next_input += 1
    dub_idx = None
    if job.dub:
        inputs += ["-i", str(job.dub)]
        dub_idx = next_input

    chain: list[str] = []
    pre: list[str] = []  # graph pieces that must come before the main chain (split layout)
    if plan is not None and plan.mode == "split" and plan.points2:
        (work / "cmds_a.txt").write_text(sendcmd_script(plan.points, "crop@a"), encoding="utf-8")
        (work / "cmds_b.txt").write_text(sendcmd_script(plan.points2, "crop@b"), encoding="utf-8")
        half = H // 2
        for tag, name in (("a", "top"), ("b", "bot")):
            pre.append(f"[sp{tag}]sendcmd=f=cmds_{tag}.txt,crop@{tag}=w={plan.crop_w}:h={plan.src_h}:x=0:y=0,"
                       f"scale={W}:{half}:flags=lanczos[{name}]")
        pre.insert(0, "[0:v]split=2[spa][spb]")
        pre.append("[top][bot]vstack=inputs=2[stack]")
        chain.append("[stack]null")
    elif plan is not None:
        (work / "cmds.txt").write_text(sendcmd_script(plan.points), encoding="utf-8")
        chain.append(f"[0:v]sendcmd=f=cmds.txt,crop=w={plan.crop_w}:h={plan.src_h}:x=0:y=0")
        chain.append(f"scale={W}:{H}:flags=lanczos")
    elif cut_info.width / cut_info.height > W / H:
        crop_w = int(round(cut_info.height * W / H)) & ~1
        chain.append(f"[0:v]crop=w={crop_w}:h={cut_info.height}:x=(iw-{crop_w})/2:y=0")
        chain.append(f"scale={W}:{H}:flags=lanczos")
    else:
        chain.append(f"[0:v]scale={W}:-2:flags=lanczos,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black")
    chain.append("setsar=1")
    graph: list[str] = [*pre, ",".join(chain) + "[vsc]"]
    cur = "vsc"  # B-roll goes under the captions, so it is overlaid before the ass filter
    for n, ((bs, be, _), idx) in enumerate(zip(job.broll, broll_idx, strict=False)):
        graph.append(f"[{idx}:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,"
                     f"fps=30,format=yuv420p,setpts=PTS-STARTPTS+{bs:.2f}/TB[br{n}]")
        graph.append(f"[{cur}][br{n}]overlay=eof_action=pass:enable='between(t,{bs:.2f},{be:.2f})'[bo{n}]")
        cur = f"bo{n}"

    ass_name = "captions.ass"
    (work / ass_name).write_text(
        build_ass(job.captions, width=W, height=H, style=job.style, hook=job.hook,
                  progress_bar=job.progress_bar, duration=cut_info.duration),
        encoding="utf-8",
    )
    ass_opts = f"ass={ass_name}"
    if job.fonts_dir:
        ass_opts += f":fontsdir={_filter_path(job.fonts_dir)}"
    graph.append(f"[{cur}]{ass_opts},format=yuv420p[vcap]")
    if logo_idx is not None:
        graph.append(f"[{logo_idx}:v]scale=iw*0.16:-1,format=rgba[logo]")
        graph.append("[vcap][logo]overlay=x=W-w-48:y=48:format=auto[vbase]")
    else:
        graph.append("[vcap]null[vbase]")
    current = "vbase"  # emoji pop-ups: chain one timed overlay per emoji
    for n, ((start, end, _), idx) in enumerate(zip(job.emojis, emoji_idx, strict=False)):
        size = round(W * 0.14)
        graph.append(f"[{idx}:v]scale={size}:{size},format=rgba[em{n}]")
        graph.append(f"[{current}][em{n}]overlay=x=(W-w)/2:y=H*0.60:enable='between(t,{start:.2f},{end:.2f})'[ov{n}]")
        current = f"ov{n}"
    graph.append(f"[{current}]null[vout]")

    if dub_idx is not None:
        graph.append(f"[{dub_idx}:a]aresample=48000,loudnorm=I=-14:TP=-1.5:LRA=11[dubv]")
        graph.append("[0:a]volume=0.10[orig]")
        graph.append("[orig][dubv]amix=inputs=2:duration=longest:normalize=0[voice]")
    else:
        graph.append("[0:a]loudnorm=I=-14:TP=-1.5:LRA=11[voice]")
    if bgm_idx is not None:
        graph.append(f"[{bgm_idx}:a]volume={job.bgm_volume},aresample=48000[bg]")
        graph.append("[voice]asplit=2[vo][sc]")
        graph.append("[bg][sc]sidechaincompress=threshold=0.03:ratio=8:attack=15:release=350[duck]")
        graph.append("[vo][duck]amix=inputs=2:duration=first:dropout_transition=0[aout]")
    else:
        graph.append("[voice]anull[aout]")

    args = [job.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *inputs,
            "-filter_complex", ";".join(graph),
            "-map", "[vout]", "-map", "[aout]",
            *video_encode_args(job.encoder),
            "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ar", "48000", "-ac", "2",
            "-t", f"{cut_info.duration:.3f}", "-movflags", "+faststart", str(job.output)]
    run_ffmpeg(args, cwd=work, timeout=1800)


def render_clip(job: RenderJob) -> RenderResult:
    """Render one finished vertical clip to ``job.output``."""
    job.output.parent.mkdir(parents=True, exist_ok=True)
    source = probe(job.source, job.ffprobe)
    fps = float(min(MAX_FPS, max(1, round(source.fps))))

    with tempfile.TemporaryDirectory(prefix="clip_", dir=job.output.parent) as tmp:
        work = Path(tmp)
        cut = work / "cut.mp4"
        _cut(job, cut, source, fps)
        cut_info = probe(cut, job.ffprobe)

        plan = None
        if job.track_faces:
            try:
                plan = analyse_layout(cut, cut_info.duration, job.width, job.height, fps, job.models_dir, job.layout)
            except Exception as exc:  # tracking is an enhancement; never fail a clip because of it
                log.warning("face tracking unavailable, using centre crop: %s", exc)

        _final(job, work, cut, cut_info, plan)

    if not job.output.exists() or job.output.stat().st_size == 0:
        raise MediaError(f"render produced no output for {job.output.name}")
    final = probe(job.output, job.ffprobe)
    return RenderResult(job.output, final.duration, plan is not None, job.encoder, plan.mode if plan else "center")
