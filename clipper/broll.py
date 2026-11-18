"""AI B-roll: an LLM picks visual moments, Pexels supplies stock footage.

Needs ``PEXELS_API_KEY`` (free at pexels.com/api). The footage is overlaid full-frame
for a couple of seconds under the captions while the speaker's audio continues.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from .llm import LLM

log = logging.getLogger(__name__)

PEXELS_SEARCH = "https://api.pexels.com/videos/search"


class BrollItem(BaseModel):
    start: float = Field(description="Seconds from the start of the clip when the footage appears.")
    duration: float = Field(description="How long it stays on screen, between 1.5 and 3 seconds.")
    query: str = Field(description="1-3 word stock-footage search phrase showing what is being said.")


class BrollPlan(BaseModel):
    items: list[BrollItem]


@dataclass(frozen=True)
class BrollCut:
    start: float
    end: float
    path: Path


def plan_broll(llm: LLM, lines: list[tuple[float, float, str]], clip_seconds: float,
               max_items: int = 3) -> list[BrollItem]:
    """Ask the LLM where B-roll helps. ``lines`` are (start, end, text) on the clip timeline."""
    transcript = "\n".join(f"[{s:.1f}-{e:.1f}] {t}" for s, e, t in lines)
    prompt = (
        f"This is a {clip_seconds:.0f}-second short video. Pick up to {max_items} moments where stock B-roll "
        "would strengthen what is being said (concrete nouns, places, actions). Rules: never in the first "
        f"2 seconds or the last 2 seconds, at least 3 seconds apart, each 1.5-3 seconds long.\n\n{transcript}"
    )
    plan = llm.generate(prompt, BrollPlan, temperature=0.3)
    chosen: list[BrollItem] = []
    for item in sorted(plan.items, key=lambda i: i.start):
        dur = min(3.0, max(1.5, item.duration))
        ok_range = 2.0 <= item.start and item.start + dur <= clip_seconds - 2.0
        ok_gap = not chosen or item.start >= chosen[-1].start + chosen[-1].duration + 3.0
        if ok_range and ok_gap and item.query.strip():
            chosen.append(BrollItem(start=item.start, duration=dur, query=item.query.strip()[:60]))
        if len(chosen) == max_items:
            break
    return chosen


def pick_video_file(payload: dict) -> str | None:
    """Choose a download URL from a Pexels search response: HD mp4, closest to 1080 px tall, not huge."""
    best: tuple[int, str] | None = None
    for video in payload.get("videos", []):
        for f in video.get("video_files", []):
            if f.get("file_type") != "video/mp4" or not f.get("link"):
                continue
            height = f.get("height") or 0
            if height < 720:
                continue
            score = abs(height - 1920 if height > f.get("width", 0) else height - 1080)
            if best is None or score < best[0]:
                best = (score, f["link"])
    return best[1] if best else None


def fetch_broll(query: str, api_key: str, cache_dir: Path, opener=urllib.request.urlopen) -> Path | None:
    """Search Pexels and download one clip (cached). Returns None on any failure."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / (hashlib.sha1(query.lower().encode()).hexdigest()[:16] + ".mp4")
    if target.exists():
        return target
    try:
        params = urllib.parse.urlencode({"query": query, "per_page": 5, "orientation": "portrait", "size": "medium"})
        req = urllib.request.Request(f"{PEXELS_SEARCH}?{params}", headers={"Authorization": api_key})
        with opener(req, timeout=20) as resp:
            link = pick_video_file(json.loads(resp.read()))
        if not link:
            return None
        with opener(link, timeout=60) as resp:
            data = resp.read()
        tmp = target.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(target)
        return target
    except Exception as exc:  # cosmetic feature: never fail a clip over it
        log.warning("B-roll fetch failed for %r: %s", query, exc)
        return None
