# AI Clipping Software

Turn long videos (YouTube links or local files) into short, vertical, captioned clips — from the command line, a web UI, or as an **MCP server** that any AI agent can drive.

**Pipeline:** download → Whisper transcript (word timestamps, cached) → LLM picks viral moments (with hook, title, description, hashtags, score) → silence jump-cuts → face-tracked 9:16 reframing → animated word-by-word captions → loudness-normalised encode (GPU NVENC when available).

## Quick start

Requirements: Python 3.10+, [ffmpeg](https://ffmpeg.org/download.html) on `PATH`.

```bash
git clone https://github.com/LikithMeruvu/AI-Clipping-Software.git
cd AI-Clipping-Software
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # add GEMINI_API_KEY (or configure another provider)
python -m clipper doctor                               # checks ffmpeg, GPU, LLM config

python -m clipper run "https://www.youtube.com/watch?v=VIDEO_ID" --num 5 --style hormozi
python -m clipper run ./talk.mp4 --aspect 1:1 --translate Spanish --focus "funny moments"
```
`python main.py <url>` still works. Output goes to `output/<timestamp>/` as `clip_XX.mp4`, `clip_XX.srt` and `manifest.json` (titles, hooks, descriptions, hashtags, the selection method and any warnings).

CPU-only? Set `WHISPER_MODEL=small` in `.env` (the default `large-v3-turbo` is intended for a GPU).

## Web UI
```bash
python -m clipper serve        # http://127.0.0.1:8000
```
Upload or paste a link, pick style/aspect/options, watch progress, preview and download clips. Binding to a non-local address requires `API_TOKEN` (bearer token on `/api/*`). Docker (Dockerfile provided, **not built or tested yet**): `docker build -t clipper . && docker run -e API_TOKEN=... -e GEMINI_API_KEY=... -p 8000:8000 -v clipper-data:/data clipper`.

## MCP server (for AI agents)
```bash
python -m clipper mcp            # stdio (Claude Desktop, Claude Code, Cursor, ...)
python -m clipper mcp --http     # streamable HTTP on 127.0.0.1:8765
```
Claude Code: `claude mcp add clipper -- python -m clipper mcp` (run from the repo, or use absolute paths). Generic config:
```json
{ "mcpServers": { "clipper": { "command": "/path/to/.venv/bin/python", "args": ["-m", "clipper", "mcp"], "cwd": "/path/to/AI-Clipping-Software" } } }
```
| Tool | Purpose |
|---|---|
| `clip_video` | One-shot: find the best moments with the configured LLM and render finished clips (async job) |
| `transcribe_video` | Timestamped transcript so **the calling agent picks the moments itself** — no separate LLM key needed |
| `render_clips` | Render the exact ranges the agent chose, with hook/title/hashtags (async job) |
| `get_job` / `list_jobs` | Poll progress; the result lists absolute file paths and metadata |
| `list_styles` | Caption styles and aspect ratios |

## LLM providers
Selection, hooks, titles and translation go through one small interface (`clipper/llm.py`), with validated structured output and retries.
- `LLM_PROVIDER=gemini` – native Google SDK (tested live).
- `LLM_PROVIDER=openai` + `LLM_BASE_URL` – any OpenAI-compatible endpoint: OpenAI, OpenRouter, Groq, DeepSeek, Mistral, Together, Ollama, LM Studio, vLLM… (implemented, **not yet tested against a live non-Gemini endpoint**).

If the LLM fails, the run falls back to a deterministic heuristic and says so in the console and `manifest.json` — it never silently produces random clips.

## Feature status (vs. Opus Clip-style tools)
| Feature | Status |
|---|---|
| AI clip selection with virality score, hook, title, description, hashtags; `--focus` prompt | ✅ tested live |
| Auto-reframe 9:16 / 4:5 / 1:1 / 16:9 with face tracking | ✅ tested live |
| Active-speaker switching (mouth-movement signal) | ✅ unit-tested; single-camera interview footage only partially checked |
| Split-screen layout for two speakers on screen (`--layout`) | ✅ mechanism verified on a composite test video; framing quality on real two-shot footage not verified |
| Animated word-by-word captions (5 styles), SRT, hook headline, progress bar, logo | ✅ tested live |
| Silence/pause removal, caption translation | ✅ tested live |
| AI dubbing (`--dub Spanish`, Gemini TTS, original kept underneath) | ✅ tested live (output transcribes back as Spanish) |
| Emoji pop-ups on trigger words (`--emoji`) | ✅ overlay verified; keyword list is simple |
| AI B-roll from Pexels (`--broll`, needs `PEXELS_API_KEY`) | ⚠️ render verified with a local clip; the Pexels API calls are only mock-tested |
| Publish/schedule to YouTube Shorts (`publish`, `auth`) | ⚠️ preview verified; **upload never run** (needs your OAuth client); unverified API projects are forced private by YouTube |
| Performance stats vs predicted score (`stats`) | ⚠️ mock-tested only, YouTube only |
| Background music with ducking (`--bgm`), web UI, job queue, MCP server | ✅ |
| TikTok / Instagram / LinkedIn publishing | ❌ needs approved developer apps; not built |
| Speaker diarisation / voice cloning | ❌ not built |

### Publishing (YouTube)
```bash
# one time: create an OAuth *Desktop* client (YouTube Data API v3), set YOUTUBE_CLIENT_SECRETS in .env
python -m clipper auth youtube
python -m clipper publish output/20261118-120000 --clip 1                       # preview only
python -m clipper publish output/20261118-120000 --clip 1 --schedule 2026-12-01T15:00:00Z --yes
python -m clipper stats
```
The MCP tool `publish_youtube` is preview-only unless `confirm=true`, so an agent cannot upload without your approval step.

## Development
```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check clipper tests && pytest -q
```
Layout: `clipper/` (pipeline, `llm.py`, `selection.py`, `captions.py`, `reframe.py`, `render.py`, `web.py`, `mcp_server.py`, `cli.py`), `tests/`.

## Notes
Only download and clip content you have the rights to use. MIT licensed.
