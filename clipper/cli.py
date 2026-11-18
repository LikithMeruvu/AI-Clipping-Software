"""Command-line interface.

    python -m clipper run "https://youtu.be/VIDEO_ID" --num 5 --style hormozi
    python -m clipper run ./talk.mp4 --aspect 1:1 --translate Spanish
    python -m clipper serve            # web UI + JSON API on http://127.0.0.1:8000
    python -m clipper doctor           # check ffmpeg, GPU, LLM and model setup
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .captions import STYLES
from .config import get_settings
from .media import MediaError, pick_encoder
from .pipeline import ASPECTS, IngestError, Options, process


def _progress_printer(verbose: bool):
    last = {"stage": None}

    def emit(stage: str, pct: float, message: str) -> None:
        if stage != last["stage"] or verbose:
            print(f"[{pct * 100:5.1f}%] {message}", flush=True)
            last["stage"] = stage

    return emit


def cmd_run(args: argparse.Namespace) -> int:
    settings = get_settings()
    options = Options(
        num_clips=args.num, min_seconds=args.min, max_seconds=args.max, style=args.style,
        aspect=args.aspect, focus=args.focus, language=args.language, translate_to=args.translate, dub_to=args.dub,
        hook=not args.no_hook, emoji=args.emoji, broll=args.broll, remove_silence=not args.keep_silence, captions=not args.no_captions,
        layout=args.layout, progress_bar=args.progress_bar, track_faces=not args.no_tracking,
        bgm=Path(args.bgm) if args.bgm else None, bgm_volume=args.bgm_volume,
        logo=Path(args.logo) if args.logo else None,
    )
    job_id = time.strftime("%Y%m%d-%H%M%S")
    job_dir = settings.output_dir / job_id
    try:
        manifest = process(args.source, options, settings, job_dir, _progress_printer(args.verbose))
    except (IngestError, MediaError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print()
    print(f"Selection method: {manifest['selection']}  |  encoder: {manifest['encoder']}")
    for warning in manifest["warnings"]:
        print(f"warning: {warning}")
    for clip in manifest["clips"]:
        status = clip.get("status")
        if status == "ok":
            print(f"  #{clip['index']} score {clip['score']:>3}  {clip['duration']:>5.1f}s  "
                  f"{job_dir / clip['file']}")
            print(f"       hook: {clip['hook']}")
        else:
            print(f"  #{clip['index']} FAILED: {clip.get('error', '')[:200]}")
    print(f"\nManifest: {job_dir / 'manifest.json'}  ({manifest['elapsed_seconds']}s)")
    return 0 if any(c.get("status") == "ok" for c in manifest["clips"]) else 1


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .web import create_app

    settings = get_settings()
    host = args.host or settings.host
    port = args.port or settings.port
    if host not in ("127.0.0.1", "localhost") and not settings.api_token:
        print("refusing to serve on a public interface without API_TOKEN set", file=sys.stderr)
        return 2
    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp_server import run_http, run_stdio

    # stdout belongs to the MCP protocol on stdio; logging already goes to stderr.
    if args.http:
        run_http(args.host, args.port)
    else:
        run_stdio()
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    settings = get_settings()
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"[{'ok' if passed else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")

    try:
        import subprocess
        out = subprocess.run([settings.ffmpeg_bin, "-version"], capture_output=True, text=True, timeout=20)
        check("ffmpeg", out.returncode == 0, out.stdout.splitlines()[0] if out.stdout else "")
        check("encoder", True, pick_encoder(settings.ffmpeg_bin, settings.encoder))
        subprocess.run([settings.ffmpeg_bin, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=20)
    except FileNotFoundError:
        check("ffmpeg", False, f"'{settings.ffmpeg_bin}' not found on PATH")

    try:
        import ctranslate2
        check("GPU (CUDA)", True, f"{ctranslate2.get_cuda_device_count()} device(s) visible")
    except Exception as exc:
        check("GPU (CUDA)", True, f"not available ({exc.__class__.__name__}); CPU will be used")

    has_key = bool(settings.gemini_api_key or settings.llm_api_key)
    check("LLM configured", settings.llm_provider == "gemini" and has_key or settings.llm_provider == "openai"
          and bool(settings.llm_base_url), f"{settings.llm_provider}:{settings.llm_model}")
    check("whisper model", True, settings.whisper_model)
    print("\nAll required checks passed." if ok else "\nSome checks failed; see above.")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clipper", description="AI viral-clip generator")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="clip a YouTube URL or local video file")
    run.add_argument("source", help="YouTube URL or path to a video file")
    run.add_argument("--num", type=int, default=5, help="number of clips (1-30)")
    run.add_argument("--min", type=float, default=20.0, help="minimum clip seconds")
    run.add_argument("--max", type=float, default=60.0, help="maximum clip seconds")
    run.add_argument("--style", default="hormozi", choices=sorted(STYLES))
    run.add_argument("--aspect", default="9:16", choices=sorted(ASPECTS))
    run.add_argument("--focus", help="steer selection, e.g. 'funny moments about startups'")
    run.add_argument("--language", help="source language code, e.g. en (default: auto)")
    run.add_argument("--translate", metavar="LANGUAGE", help="translate captions, e.g. Spanish")
    run.add_argument("--dub", metavar="LANGUAGE", help="AI voice-over in this language (Gemini TTS)")
    run.add_argument("--emoji", action="store_true", help="pop up emoji on trigger words")
    run.add_argument("--broll", action="store_true", help="AI-placed stock B-roll (needs PEXELS_API_KEY)")
    run.add_argument("--keep-silence", action="store_true", help="do not remove long pauses")
    run.add_argument("--no-hook", action="store_true", help="no hook headline at the top")
    run.add_argument("--no-captions", action="store_true", help="no burned-in captions")
    run.add_argument("--layout", default="auto", choices=["auto", "single", "split"],
                     help="split stacks two on-screen speakers; auto decides per clip")
    run.add_argument("--no-tracking", action="store_true", help="centre crop instead of face tracking")
    run.add_argument("--progress-bar", action="store_true", help="draw a progress bar along the bottom")
    run.add_argument("--bgm", help="background music file (ducked under speech)")
    run.add_argument("--bgm-volume", type=float, default=0.18)
    run.add_argument("--logo", help="PNG logo overlaid in the top-right corner")
    run.add_argument("-v", "--verbose", action="store_true")
    run.set_defaults(func=cmd_run)

    serve = sub.add_parser("serve", help="start the web UI and API")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.set_defaults(func=cmd_serve)

    mcp = sub.add_parser("mcp", help="run as an MCP server for AI agents (stdio by default)")
    mcp.add_argument("--http", action="store_true", help="streamable HTTP instead of stdio")
    mcp.add_argument("--host", default="127.0.0.1")
    mcp.add_argument("--port", type=int, default=8765)
    mcp.set_defaults(func=cmd_mcp)

    doctor = sub.add_parser("doctor", help="check the environment")
    doctor.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
