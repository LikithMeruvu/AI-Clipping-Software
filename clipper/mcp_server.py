"""MCP server: lets any MCP-capable agent (Claude, Cursor, Codex, ...) drive the clipper.

Two ways to use it
1. ``clip_video``: the full pipeline. Selection uses the LLM configured in ``.env``
   (or the heuristic fallback).
2. ``transcribe_video`` + ``render_clips``: building blocks. The *host agent* reads the
   transcript, picks the moments itself and asks for exactly those ranges, so no
   separate LLM API key is required.

Long work runs in a background worker; tools return a job id immediately and
``get_job`` reports progress. Run with ``python -m clipper mcp`` (stdio) or
``python -m clipper mcp --http`` (streamable HTTP on 127.0.0.1).
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from . import __version__
from .captions import STYLES
from .config import Settings, get_settings
from .pipeline import ASPECTS, Options, prepare_source, process
from .selection import Clip
from .transcribe import Transcriber, cached_transcribe


def build_server(settings: Settings | None = None) -> MCPServer:
    settings = settings or get_settings()
    mcp = MCPServer(
        "ai-clipping-software",
        version=__version__,
        instructions=(
            "Turn long videos into short vertical clips with captions and face-tracked framing. "
            "Use clip_video for one-shot automatic clipping, or transcribe_video then render_clips "
            "to choose the moments yourself. Jobs are asynchronous: poll get_job until status is "
            "'done' or 'failed'."
        ),
    )
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-job")
    jobs: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()

    def submit(kind: str, fn) -> str:
        job_id = uuid.uuid4().hex[:12]
        job_dir = settings.output_dir / job_id
        jobs[job_id] = {"id": job_id, "kind": kind, "status": "queued", "stage": "queued", "progress": 0.0,
                        "message": "", "output_dir": str(job_dir.resolve()), "result": None, "error": None}

        def progress(stage: str, pct: float, message: str) -> None:
            with lock:
                jobs[job_id].update(stage=stage, progress=round(pct, 3), message=message)

        def run() -> None:
            with lock:
                jobs[job_id]["status"] = "running"
            try:
                result = fn(job_dir, progress)
                with lock:
                    jobs[job_id].update(status="done", progress=1.0, result=result)
            except Exception as exc:  # report every failure to the agent instead of crashing the server
                with lock:
                    jobs[job_id].update(status="failed", error=f"{type(exc).__name__}: {exc}"[-600:])

        pool.submit(run)
        return job_id

    def options_from(**kw) -> Options:
        opts = Options(**{k: v for k, v in kw.items() if v is not None})
        opts.validate()
        return opts

    @mcp.tool()
    def list_styles() -> dict[str, Any]:
        """List caption styles and aspect ratios accepted by the other tools."""
        return {"styles": {k: v.label for k, v in STYLES.items()}, "aspects": list(ASPECTS)}

    @mcp.tool()
    def clip_video(source: str, num_clips: int = 5, min_seconds: float = 20, max_seconds: float = 60,
                   style: str = "hormozi", aspect: str = "9:16", focus: str | None = None,
                   translate_to: str | None = None, hook: bool = True, remove_silence: bool = True,
                   progress_bar: bool = False, layout: str = "auto") -> dict[str, str]:
        """Automatically find the best moments in a video and render finished captioned clips.

        source: YouTube URL or absolute path to a local video file.
        focus: optional steer, e.g. "funniest moments about startups".
        Returns a job id; poll get_job. Needs the LLM configured in .env, else a heuristic is used
        (reported in the result warnings).
        """
        opts = options_from(num_clips=num_clips, min_seconds=min_seconds, max_seconds=max_seconds, style=style,
                            aspect=aspect, focus=focus, translate_to=translate_to, hook=hook,
                            remove_silence=remove_silence, progress_bar=progress_bar, layout=layout)
        return {"job_id": submit("clip_video", lambda d, p: process(source, opts, settings, d, p))}

    @mcp.tool()
    def transcribe_video(source: str, max_chars: int = 60000) -> dict[str, Any]:
        """Transcribe a video and return timestamped lines so YOU can choose clip ranges.

        Blocks until done (cached afterwards). For long videos the text is truncated to
        max_chars; call again is not needed, use the returned 'truncated_at' seconds.
        """
        src = prepare_source(source, settings, lambda *_: None)
        tr = cached_transcribe(Transcriber(settings.whisper_model, settings.whisper_device, settings.models_dir),
                               src.path, src.key, settings.cache_dir / "transcripts")
        lines, size, truncated_at = [], 0, None
        for s in tr.segments:
            row = f"[{s.start:.1f}-{s.end:.1f}] {s.text}"
            if size + len(row) > max_chars:
                truncated_at = s.start
                break
            lines.append(row)
            size += len(row) + 1
        return {"title": src.title, "source": source, "language": tr.language, "duration": round(tr.duration, 1),
                "transcript": "\n".join(lines), "truncated_at": truncated_at}

    @mcp.tool()
    def render_clips(source: str, clips: list[dict[str, Any]], style: str = "hormozi", aspect: str = "9:16",
                     hook: bool = True, remove_silence: bool = True, progress_bar: bool = False,
                     translate_to: str | None = None) -> dict[str, str]:
        """Render clips for time ranges you chose (typically after transcribe_video).

        clips: list of {"start": seconds, "end": seconds, "hook": "headline", "title": "...",
               "description": "...", "hashtags": ["#a"], "score": 0-100}. Only start and end are required.
        Returns a job id; poll get_job.
        """
        chosen: list[Clip] = []
        for i, c in enumerate(clips, 1):
            start, end = float(c["start"]), float(c["end"])
            if not 5 <= end - start <= 180:
                raise ValueError(f"clip {i}: duration must be between 5 and 180 seconds")
            chosen.append(Clip(start=start, end=end, score=int(c.get("score", 80)),
                               hook=str(c.get("hook", ""))[:80], title=str(c.get("title", f"Clip {i}"))[:100],
                               description=str(c.get("description", "")), hashtags=list(c.get("hashtags", [])),
                               reason="chosen by the calling agent", method="manual"))
        if not chosen:
            raise ValueError("provide at least one clip")
        opts = options_from(style=style, aspect=aspect, hook=hook, remove_silence=remove_silence,
                            progress_bar=progress_bar, translate_to=translate_to, num_clips=len(chosen))
        return {"job_id": submit("render_clips",
                                lambda d, p: process(source, opts, settings, d, p, manual_clips=chosen))}

    @mcp.tool()
    def get_job(job_id: str) -> dict[str, Any]:
        """Status, progress and (when done) the manifest: file paths, titles, hooks, hashtags, warnings."""
        job = jobs.get(job_id)
        if job is None:
            raise ValueError("unknown job id")
        with lock:
            view = dict(job)
        result = view.get("result")
        if result:  # attach absolute file paths so the agent can hand them to the user
            base = Path(view["output_dir"])
            view["result"] = {**result, "clips": [
                {**c, "path": str(base / c["file"]), "srt_path": str(base / c["srt"])} if c.get("status") == "ok" else c
                for c in result["clips"]]}
        return view

    @mcp.tool()
    def list_jobs() -> list[dict[str, Any]]:
        """Jobs started in this server session, newest first."""
        with lock:
            return [{k: j[k] for k in ("id", "kind", "status", "stage", "progress")} for j in reversed(jobs.values())]

    return mcp


def run_stdio() -> None:
    build_server().run(transport="stdio")


def run_http(host: str = "127.0.0.1", port: int = 8765) -> None:
    build_server().run(transport="streamable-http", host=host, port=port)

