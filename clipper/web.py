"""Web service: JSON API, persisted job queue and the single-page UI.

Design notes
- Jobs run in a bounded thread pool (default 1 worker: one GPU, one Whisper model).
- Job state is persisted to ``data/jobs/<id>/job.json`` so a restart keeps history
  and marks interrupted jobs as failed instead of leaving them "running" forever.
- Uploads are streamed to disk with a size cap and an extension allow-list; every
  filename served back to the client is validated against a strict pattern.
- With ``API_TOKEN`` set, every ``/api`` request needs ``Authorization: Bearer <token>``.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .captions import STYLES
from .config import Settings
from .pipeline import ASPECTS, IngestError, Options, process

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
FILE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")
JOB_ID = re.compile(r"^[a-f0-9]{12}$")


class JobOptions(BaseModel):
    num_clips: int = Field(default=5, ge=1, le=30)
    min_seconds: float = Field(default=20.0, ge=5, le=180)
    max_seconds: float = Field(default=60.0, ge=5, le=180)
    style: str = "hormozi"
    aspect: str = "9:16"
    focus: str | None = Field(default=None, max_length=300)
    language: str | None = Field(default=None, max_length=10)
    translate_to: str | None = Field(default=None, max_length=40)
    dub_to: str | None = Field(default=None, max_length=40)
    hook: bool = True
    remove_silence: bool = True
    captions: bool = True
    emoji: bool = False
    progress_bar: bool = False
    track_faces: bool = True
    layout: str = "auto"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def path(self, job_id: str) -> Path:
        return self.root / job_id

    def save(self, job: dict[str, Any]) -> None:
        job["updated"] = _now()
        target = self.path(job["id"]) / "job.json"
        tmp = target.with_suffix(".tmp")
        with self._lock:
            tmp.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(target)

    def load(self, job_id: str) -> dict[str, Any] | None:
        if not JOB_ID.match(job_id):
            return None
        target = self.path(job_id) / "job.json"
        if not target.exists():
            return None
        return json.loads(target.read_text(encoding="utf-8"))

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        jobs = []
        for child in self.root.iterdir():
            if child.is_dir() and JOB_ID.match(child.name):
                job = self.load(child.name)
                if job:
                    jobs.append(job)
        jobs.sort(key=lambda j: j.get("created", ""), reverse=True)
        return jobs[:limit]

    def mark_interrupted(self) -> None:
        for job in self.list(limit=10_000):
            if job["status"] in ("queued", "running"):
                job.update(status="failed", error="interrupted by a server restart", stage="failed")
                self.save(job)


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="AI Clipping Software", version=__version__)
    store = JobStore(settings.jobs_dir)
    store.mark_interrupted()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clip-job")
    uploads_dir = settings.data_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    def require_token(request: Request) -> None:
        if not settings.api_token:
            return
        header = request.headers.get("authorization", "")
        supplied = header[7:] if header.lower().startswith("bearer ") else ""
        if not secrets.compare_digest(supplied, settings.api_token.get_secret_value()):
            raise HTTPException(status_code=401, detail="missing or invalid API token")

    def active_jobs() -> int:
        return sum(1 for j in store.list(limit=10_000) if j["status"] in ("queued", "running"))

    def run_job(job_id: str, source: str, options: Options) -> None:
        job = store.load(job_id)
        assert job is not None
        job.update(status="running", stage="start", progress=0.0, message="Starting")
        store.save(job)

        def progress(stage: str, pct: float, message: str) -> None:
            current = store.load(job_id) or job
            current.update(stage=stage, progress=round(pct, 3), message=message)
            store.save(current)

        try:
            manifest = process(source, options, settings, store.path(job_id) / "clips", progress)
            done = store.load(job_id) or job
            done.update(status="done", stage="done", progress=1.0, message="Finished",
                        manifest=manifest, error=None)
            store.save(done)
        except (IngestError, Exception) as exc:  # every failure lands in job.json, never silently
            log.exception("job %s failed", job_id)
            failed = store.load(job_id) or job
            failed.update(status="failed", stage="failed", error=str(exc)[-600:] or type(exc).__name__)
            store.save(failed)

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "llm": f"{settings.llm_provider}:{settings.llm_model}",
                "styles": {k: v.label for k, v in STYLES.items()}, "aspects": list(ASPECTS)}

    @app.post("/api/jobs", status_code=201, dependencies=[Depends(require_token)])
    async def create_job(url: str | None = Form(default=None), file: UploadFile | None = File(default=None),
                         options: str = Form(default="{}")) -> JSONResponse:
        try:
            opts = JobOptions.model_validate_json(options or "{}")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"invalid options: {exc}") from exc
        if opts.max_seconds <= opts.min_seconds:
            raise HTTPException(status_code=422, detail="max_seconds must be greater than min_seconds")
        if opts.style not in STYLES or opts.aspect not in ASPECTS:
            raise HTTPException(status_code=422, detail="unknown style or aspect")
        if active_jobs() >= settings.max_queued_jobs:
            raise HTTPException(status_code=429, detail="too many jobs in the queue; try again later")

        job_id = uuid.uuid4().hex[:12]
        if file is not None and file.filename:
            suffix = Path(file.filename).suffix.lower()
            if suffix not in UPLOAD_EXTENSIONS:
                raise HTTPException(status_code=415, detail=f"unsupported file type {suffix or '(none)'}")
            limit = settings.max_upload_mb * 1024 * 1024
            target = uploads_dir / f"{job_id}{suffix}"
            written = 0
            with target.open("wb") as fh:
                while chunk := await file.read(1024 * 1024):
                    written += len(chunk)
                    if written > limit:
                        fh.close()
                        target.unlink(missing_ok=True)
                        raise HTTPException(status_code=413, detail=f"file exceeds {settings.max_upload_mb} MB")
                    fh.write(chunk)
            source: str = str(target)
            source_label = file.filename
        elif url:
            source = url.strip()
            source_label = source
        else:
            raise HTTPException(status_code=422, detail="provide a YouTube url or a video file")

        store.path(job_id).mkdir(parents=True, exist_ok=True)
        job = {"id": job_id, "status": "queued", "stage": "queued", "progress": 0.0,
               "message": "Waiting for a worker", "source": source_label, "origin_path": source,
               "options": opts.model_dump(), "created": _now(), "updated": _now(),
               "error": None, "manifest": None}
        store.save(job)
        run_options = Options(**opts.model_dump())
        executor.submit(run_job, job_id, source, run_options)
        return JSONResponse({"id": job_id, "status": "queued"}, status_code=201)

    @app.get("/api/jobs", dependencies=[Depends(require_token)])
    def list_jobs() -> list[dict[str, Any]]:
        return [_public(j) for j in store.list()]

    @app.get("/api/jobs/{job_id}", dependencies=[Depends(require_token)])
    def get_job(job_id: str) -> dict[str, Any]:
        job = store.load(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return _public(job)

    @app.get("/api/jobs/{job_id}/files/{name}", dependencies=[Depends(require_token)])
    def get_file(job_id: str, name: str) -> FileResponse:
        if not JOB_ID.match(job_id) or not FILE_NAME.match(name):
            raise HTTPException(status_code=404, detail="not found")
        base = (store.path(job_id) / "clips").resolve()
        target = (base / name).resolve()
        if base not in target.parents or not target.is_file():
            raise HTTPException(status_code=404, detail="not found")
        media = "video/mp4" if target.suffix == ".mp4" else "application/octet-stream"
        return FileResponse(target, media_type=media, filename=name)

    @app.delete("/api/jobs/{job_id}", dependencies=[Depends(require_token)])
    def delete_job(job_id: str) -> dict[str, bool]:
        job = store.load(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job["status"] in ("queued", "running"):
            raise HTTPException(status_code=409, detail="cannot delete a job that is still running")
        shutil.rmtree(store.path(job_id), ignore_errors=True)
        return {"deleted": True}

    @app.get("/api/time")
    def server_time() -> dict[str, float]:
        return {"now": time.time()}

    return app


def _public(job: dict[str, Any]) -> dict[str, Any]:
    """Job view for clients: drops internal paths, keeps everything the UI needs."""
    view = {k: v for k, v in job.items() if k != "origin_path"}
    manifest = job.get("manifest")
    if manifest:
        view["manifest"] = {**manifest, "clips": [
            {**c, "url": f"/api/jobs/{job['id']}/files/{c['file']}",
             "srt_url": f"/api/jobs/{job['id']}/files/{c['srt']}"} if c.get("status") == "ok" else c
            for c in manifest.get("clips", [])
        ]}
    return view
