"""Publishing and performance stats for YouTube (Shorts), with optional scheduling.

Setup (one time): create an OAuth *Desktop* client in Google Cloud Console with the
YouTube Data API v3 enabled, download it, and point ``YOUTUBE_CLIENT_SECRETS`` at the
JSON file. Then run ``python -m clipper auth youtube``. The refresh token is stored in
``data/youtube_token.json`` (keep it private).

Uploads from an unverified API project are forced to *private* by YouTube until the
project passes an audit; that is a YouTube policy, not something this code can change.
TikTok and Instagram are not supported (they require approved developer apps).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]


class PublishError(RuntimeError):
    pass


@dataclass(frozen=True)
class UploadRequest:
    file: Path
    title: str
    description: str
    privacy: str = "private"            # private | unlisted | public
    publish_at: str | None = None       # RFC 3339 UTC; forces private until then
    tags: tuple[str, ...] = ()


def build_body(req: UploadRequest) -> dict[str, Any]:
    """The ``videos.insert`` request body (pure function, unit-tested)."""
    if req.privacy not in ("private", "unlisted", "public"):
        raise PublishError("privacy must be private, unlisted or public")
    title = req.title.strip()[:90]
    if "#shorts" not in title.lower() and "#shorts" not in req.description.lower():
        description = f"{req.description.strip()}\n\n#Shorts".strip()
    else:
        description = req.description.strip()
    status: dict[str, Any] = {"privacyStatus": req.privacy, "selfDeclaredMadeForKids": False}
    if req.publish_at:
        status.update(privacyStatus="private", publishAt=parse_schedule(req.publish_at))
    return {
        "snippet": {"title": title or "Untitled", "description": description[:4900],
                    "tags": [t.lstrip("#") for t in req.tags][:15], "categoryId": "22"},
        "status": status,
    }


def parse_schedule(value: str) -> str:
    """Normalise an ISO-8601 time to RFC 3339 UTC; it must be in the future."""
    try:
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PublishError(f"invalid schedule time {value!r}; use e.g. 2026-12-01T15:00:00Z") from exc
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    when = when.astimezone(timezone.utc)
    if when <= datetime.now(timezone.utc):
        raise PublishError("schedule time must be in the future")
    return when.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def authorize(client_secrets: Path, token_path: Path) -> None:
    """Interactive OAuth flow (opens a browser) and stores the refresh token."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not client_secrets.is_file():
        raise PublishError(f"client secrets file not found: {client_secrets}")
    creds = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES).run_local_server(port=0)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json(), encoding="utf-8")


def _service(token_path: Path):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    if not token_path.is_file():
        raise PublishError("not authorised yet: run `python -m clipper auth youtube`")
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds.valid:
        creds.refresh(Request())
        token_path.write_text(creds.to_json(), encoding="utf-8")
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def upload(req: UploadRequest, token_path: Path, service=None) -> str:
    """Resumable upload. Returns the YouTube video id."""
    from googleapiclient.http import MediaFileUpload

    if not req.file.is_file():
        raise PublishError(f"file not found: {req.file}")
    service = service or _service(token_path)
    media = MediaFileUpload(str(req.file), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
    request = service.videos().insert(part="snippet,status", body=build_body(req), media_body=media)
    response = None
    while response is None:
        _, response = request.next_chunk()
    video_id = response.get("id")
    if not video_id:
        raise PublishError(f"upload finished without a video id: {response}")
    return video_id


def record_publication(records_path: Path, entry: dict[str, Any]) -> None:
    records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.exists() else []
    records.append(entry)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch_stats(records_path: Path, token_path: Path, service=None) -> list[dict[str, Any]]:
    """Views/likes/comments for every video published through this tool, next to its predicted score."""
    if not records_path.exists():
        return []
    records = json.loads(records_path.read_text(encoding="utf-8"))
    ids = [r["video_id"] for r in records if re.fullmatch(r"[\w-]{6,20}", r.get("video_id", ""))]
    if not ids:
        return []
    service = service or _service(token_path)
    stats: dict[str, dict[str, Any]] = {}
    for i in range(0, len(ids), 50):
        resp = service.videos().list(part="statistics", id=",".join(ids[i:i + 50])).execute()
        for item in resp.get("items", []):
            stats[item["id"]] = item.get("statistics", {})
    return [{**r, "views": int(stats.get(r["video_id"], {}).get("viewCount", 0)),
             "likes": int(stats.get(r["video_id"], {}).get("likeCount", 0)),
             "comments": int(stats.get(r["video_id"], {}).get("commentCount", 0))} for r in records]


def load_clip(run_dir: Path, index: int) -> tuple[dict[str, Any], UploadRequest]:
    """Read ``manifest.json`` in ``run_dir`` and build the upload request for clip ``index``."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise PublishError(f"no manifest.json in {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clip = next((c for c in manifest["clips"] if c["index"] == index and c.get("status") == "ok"), None)
    if clip is None:
        raise PublishError(f"clip {index} not found or failed in {run_dir}")
    tags = tuple(clip.get("hashtags", []))
    description = f"{clip.get('description', '')}\n\n{' '.join(tags)}".strip()
    return clip, UploadRequest(file=run_dir / clip["file"], title=clip["title"], description=description, tags=tags)
