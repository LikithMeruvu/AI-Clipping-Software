"""End-to-end pipeline: source -> transcript -> clip selection -> captioned vertical clips.

The pipeline is deterministic except for the LLM calls. Everything it produces is
recorded in ``manifest.json``, including which selection method ran and any
warnings, so a degraded run is never mistaken for a good one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, Field

from .broll import fetch_broll, plan_broll
from .captions import STYLES, Line, build_srt, group_words
from .config import Settings
from .dub import synthesize
from .emoji import fetch_emoji_png, pick_emoji_events
from .llm import LLM, LLMError, build_llm
from .media import MediaError, pick_encoder, probe
from .render import RenderJob, render_clip
from .selection import Clip, SelectionResult, select_clips
from .timeline import TimeMap, keep_intervals
from .transcribe import Transcriber, cached_transcribe

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float, str], None]

ASPECTS: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "4:5": (1080, 1350),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}


class IngestError(ValueError):
    """The input is not a usable YouTube URL or local video."""


@dataclass
class Options:
    num_clips: int = 5
    min_seconds: float = 20.0
    max_seconds: float = 60.0
    style: str = "hormozi"
    aspect: str = "9:16"
    focus: str | None = None
    language: str | None = None          # source language hint; None = auto-detect
    translate_to: str | None = None      # e.g. "Spanish"; captions are translated
    dub_to: str | None = None            # e.g. "Spanish"; AI voice-over (also translates captions)
    hook: bool = True
    remove_silence: bool = True
    captions: bool = True
    emoji: bool = False
    broll: bool = False                  # needs PEXELS_API_KEY
    progress_bar: bool = False
    track_faces: bool = True
    layout: str = "auto"
    bgm: Path | None = None
    bgm_volume: float = 0.18
    logo: Path | None = None

    def validate(self) -> None:
        if not 1 <= self.num_clips <= 30:
            raise IngestError("num_clips must be between 1 and 30")
        if not 5 <= self.min_seconds < self.max_seconds <= 180:
            raise IngestError("need 5 <= min_seconds < max_seconds <= 180")
        if self.style not in STYLES:
            raise IngestError(f"unknown style {self.style!r}; choose one of {', '.join(STYLES)}")
        if self.layout not in ("auto", "single", "split"):
            raise IngestError("layout must be auto, single or split")
        if self.aspect not in ASPECTS:
            raise IngestError(f"unknown aspect {self.aspect!r}; choose one of {', '.join(ASPECTS)}")


@dataclass
class Source:
    path: Path
    key: str
    title: str
    origin: str  # "youtube" or "upload"


class _Translations(BaseModel):
    items: list[str] = Field(description="Translated lines, same order and count as the input.")


def validate_youtube_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or parsed.hostname not in YOUTUBE_HOSTS:
        raise IngestError("only http(s) YouTube URLs are supported")
    if parsed.hostname and parsed.hostname.endswith("youtu.be"):
        video_id = parsed.path.lstrip("/").split("/")[0]
    else:
        video_id = parse_qs(parsed.query).get("v", [""])[0]
        if not video_id and parsed.path.startswith(("/shorts/", "/live/")):
            video_id = parsed.path.split("/")[2] if parsed.path.count("/") >= 2 else ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,20}", video_id or ""):
        raise IngestError("could not find a YouTube video id in that URL")
    return video_id


def fetch_youtube(url: str, cache_dir: Path, progress: ProgressFn) -> Source:
    """Download a YouTube video (cached by video id) with yt-dlp."""
    import yt_dlp

    video_id = validate_youtube_url(url)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"yt_{video_id}.mp4"
    progress("download", 0.02, "Fetching video from YouTube")
    opts = {
        "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(cache_dir / f"yt_{video_id}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "fragment_retries": 3,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=not target.exists())
    if not target.exists():
        raise IngestError("download finished but the mp4 file is missing")
    return Source(path=target, key=f"yt_{video_id}", title=info.get("title") or video_id, origin="youtube")


def fingerprint_file(path: Path) -> str:
    """Stable key for an uploaded file: size plus the hash of its first 8 MB."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        h.update(fh.read(8 * 1024 * 1024))
    return f"up_{path.stat().st_size}_{h.hexdigest()[:20]}"


def prepare_source(value: str | Path, settings: Settings, progress: ProgressFn) -> Source:
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        return fetch_youtube(value, settings.cache_dir, progress)
    path = Path(value)
    if not path.is_file():
        raise IngestError(f"file not found: {path}")
    info = probe(path, settings.ffprobe_bin)
    if info.duration < 5:
        raise IngestError("video is shorter than 5 seconds")
    return Source(path=path, key=fingerprint_file(path), title=path.stem, origin="upload")


def _translate(llm: LLM, lines: list[str], target: str) -> list[str]:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines))
    prompt = (
        f"Translate each numbered subtitle line into {target}. Keep the meaning, keep it short enough "
        f"for an on-screen caption, and return exactly {len(lines)} items in the same order.\n\n{numbered}"
    )
    result = llm.generate(prompt, _Translations, temperature=0.2)
    if len(result.items) != len(lines):
        raise LLMError(f"translation returned {len(result.items)} items for {len(lines)} lines")
    return result.items


def process(source_value: str | Path, options: Options, settings: Settings, job_dir: Path,
            progress: ProgressFn = lambda *_: None, manual_clips: list[Clip] | None = None) -> dict:
    """Run the full pipeline and write clips, subtitles and ``manifest.json`` into ``job_dir``."""
    options.validate()
    started = time.time()
    job_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    source = prepare_source(source_value, settings, progress)
    info = probe(source.path, settings.ffprobe_bin)
    progress("transcribe", 0.15, f"Transcribing {source.title}")

    transcriber = Transcriber(settings.whisper_model, settings.whisper_device, settings.models_dir)
    transcript = cached_transcribe(transcriber, source.path, source.key, settings.cache_dir / "transcripts",
                                   options.language)
    if not transcript.words:
        warnings.append("no speech detected; clips use the heuristic selection without captions")

    progress("select", 0.45, "Choosing viral moments")
    llm = None
    try:
        llm = build_llm(settings)
    except LLMError as exc:
        if not manual_clips:
            warnings.append(str(exc))
    if manual_clips:  # the caller (e.g. an MCP host agent) already chose the moments
        selection = SelectionResult(method="manual", clips=manual_clips)
    else:
        selection = select_clips(transcript, llm, options.num_clips,
                                 options.min_seconds, options.max_seconds, options.focus)
    warnings.extend(selection.warnings)
    if not selection.clips:
        raise MediaError("no clips could be selected from this video")

    encoder = pick_encoder(settings.ffmpeg_bin, settings.encoder)
    width, height = ASPECTS[options.aspect]
    style = STYLES[options.style]
    fonts = settings.fonts_dir

    jobs: list[tuple[int, Clip, RenderJob, dict]] = []
    for i, clip in enumerate(selection.clips, 1):
        intervals = keep_intervals(transcript.words, clip.start, clip.end) if options.remove_silence \
            else [(clip.start, clip.end)]
        timemap = TimeMap.from_intervals(intervals)
        words = timemap.map_words(transcript.words)
        lines: list[Line] = group_words(words, style) if (options.captions and words) else []
        target_lang = options.dub_to or options.translate_to
        dub_wav = None
        if target_lang and lines and llm is not None:
            try:
                translated = _translate(llm, [" ".join(w.text for w in ln.words or ()) for ln in lines], target_lang)
                lines = [Line(start=ln.start, end=ln.end, text=t) for ln, t in zip(lines, translated, strict=False)]
                if options.dub_to:
                    dub_wav = job_dir / f"clip_{i:02d}.dub.wav"
                    synthesize(settings, " ".join(translated), options.dub_to, dub_wav, timemap.duration, warnings)
            except Exception as exc:  # translation/TTS are optional extras: keep the clip, report the gap
                warnings.append(f"translation/dub skipped for clip {i}: {exc}")
                dub_wav = None

        broll_cuts: tuple = ()
        if options.broll and llm is not None and words and lines:
            if not settings.pexels_api_key:
                warnings.append("B-roll requested but PEXELS_API_KEY is not set")
            else:
                try:
                    texts = [(ln.start, ln.end, ln.text or " ".join(w.text for w in ln.words or ())) for ln in lines]
                    got = []
                    for item in plan_broll(llm, texts, timemap.duration):
                        path = fetch_broll(item.query, settings.pexels_api_key.get_secret_value(),
                                           settings.cache_dir / "broll")
                        if path:
                            got.append((item.start, item.start + item.duration, path))
                    broll_cuts = tuple(got)
                except Exception as exc:
                    warnings.append(f"B-roll skipped for clip {i}: {exc}")
        emojis = ()
        if options.emoji and words:
            found = [(e.start, e.end, fetch_emoji_png(e.emoji, settings.cache_dir / "emoji"))
                     for e in pick_emoji_events(words)]
            emojis = tuple((a, b, p) for a, b, p in found if p)

        stem = f"clip_{i:02d}_score{clip.score}"
        out = job_dir / f"{stem}.mp4"
        render_job = RenderJob(
            source=source.path, output=out, intervals=intervals, captions=lines, style=style,
            width=width, height=height, hook=clip.hook if options.hook else None,
            progress_bar=options.progress_bar, track_faces=options.track_faces, layout=options.layout,
            bgm=options.bgm, bgm_volume=options.bgm_volume, logo=options.logo, dub=dub_wav, emojis=emojis, broll=broll_cuts,
            encoder=encoder, fonts_dir=fonts, models_dir=settings.models_dir,
            ffmpeg=settings.ffmpeg_bin, ffprobe=settings.ffprobe_bin,
        )
        meta = {
            "index": i, "file": out.name, "srt": f"{stem}.srt",
            "start": round(clip.start, 2), "end": round(clip.end, 2),
            "duration": round(sum(b - a for a, b in intervals), 2), "score": clip.score,
            "hook": clip.hook, "title": clip.title, "description": clip.description,
            "hashtags": clip.hashtags, "reason": clip.reason, "selection": clip.method,
        }
        jobs.append((i, clip, render_job, meta))

    results: list[dict] = []
    progress("render", 0.6, f"Rendering {len(jobs)} clips")
    with ThreadPoolExecutor(max_workers=settings.render_workers) as pool:
        futures = {pool.submit(render_clip, rj): (rj, meta) for _, _, rj, meta in jobs}
        done = 0
        for fut in as_completed(futures):
            rj, meta = futures[fut]
            done += 1
            try:
                res = fut.result()
                meta["face_tracked"] = res.face_tracked
                meta["layout"] = res.layout
                meta["encoder"] = res.encoder
                meta["duration"] = round(res.duration, 2)
                rj.output.with_suffix(".srt").write_text(build_srt(rj.captions, style), encoding="utf-8")
                meta["status"] = "ok"
                if rj.dub:
                    meta["dubbed"] = True
                    rj.dub.unlink(missing_ok=True)
            except Exception as exc:  # one bad clip must not sink the whole job
                log.exception("clip %s failed", meta["index"])
                meta["status"] = "failed"
                meta["error"] = str(exc)[-400:]
            results.append(meta)
            progress("render", 0.6 + 0.4 * done / len(jobs), f"Rendered {done}/{len(jobs)} clips")

    results.sort(key=lambda m: m["index"])
    manifest = {
        "source": {"title": source.title, "origin": source.origin, "duration": round(info.duration, 2),
                   "width": info.width, "height": info.height},
        "transcript": {"language": transcript.language, "words": len(transcript.words)},
        "selection": selection.method,
        "encoder": encoder,
        "options": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(options).items()},
        "warnings": warnings,
        "clips": results,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    progress("done", 1.0, "Finished")
    return manifest
