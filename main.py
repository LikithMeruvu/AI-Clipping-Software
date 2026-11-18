"""Backwards-compatible entry point: ``python main.py <youtube-url-or-file>``.

The project now lives in the ``clipper`` package; see README.md for the full CLI
(``python -m clipper --help``), the web UI and the MCP server.
"""
import sys

from clipper.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] not in {"run", "serve", "mcp", "doctor", "-h", "--help"}:
        args = ["run", *args]  # legacy form: python main.py <url>
    elif not args:
        url = input("Enter YouTube URL or video path: ").strip()
        args = ["run", url]
    raise SystemExit(main(args))
