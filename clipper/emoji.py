"""Emoji pop-ups: keyword-triggered Twemoji images overlaid above the captions.

libass cannot draw colour emoji, so the renderer overlays small PNGs (fetched once from
the Twemoji CDN and cached) for a short moment when a trigger word is spoken.
"""

from __future__ import annotations

import logging
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .captions import Word

log = logging.getLogger(__name__)

TWEMOJI_URL = "https://cdn.jsdelivr.net/gh/jdecked/twemoji@15.1.0/assets/72x72/{code}.png"

KEYWORDS: dict[str, str] = {
    "money": "💰", "cash": "💵", "dollars": "💵", "profit": "📈", "rich": "🤑", "million": "🤑", "millions": "🤑",
    "business": "💼", "sales": "📈", "growth": "📈", "invest": "📈", "price": "🏷️",
    "love": "❤️", "heart": "❤️", "happy": "😊", "funny": "😂", "laugh": "😂", "lol": "😂",
    "crazy": "🤯", "mind": "🤯", "insane": "🤯", "wow": "😮", "shocked": "😱", "scary": "😱", "afraid": "😱",
    "fire": "🔥", "hot": "🔥", "best": "🏆", "win": "🏆", "winner": "🏆", "success": "🏆", "goal": "🎯",
    "idea": "💡", "think": "🤔", "why": "🤔", "question": "❓", "secret": "🤫", "quiet": "🤫",
    "time": "⏰", "fast": "⚡", "quick": "⚡", "power": "⚡", "energy": "⚡",
    "stop": "🛑", "warning": "⚠️", "danger": "⚠️", "mistake": "❌", "wrong": "❌", "fail": "❌", "failed": "❌",
    "right": "✅", "correct": "✅", "yes": "✅", "check": "✅",
    "work": "🛠️", "build": "🛠️", "learn": "📚", "book": "📚", "read": "📚", "school": "🎓",
    "world": "🌍", "global": "🌍", "home": "🏠", "house": "🏠", "car": "🚗", "phone": "📱", "computer": "💻",
    "ai": "🤖", "robot": "🤖", "game": "🎮", "music": "🎵", "food": "🍔", "coffee": "☕",
    "sad": "😢", "cry": "😢", "angry": "😡", "hate": "😡", "sleep": "😴", "tired": "😴",
    "eyes": "👀", "look": "👀", "watch": "👀", "listen": "👂", "hands": "🙌", "thanks": "🙏", "please": "🙏",
}


@dataclass(frozen=True)
class EmojiEvent:
    start: float
    end: float
    emoji: str


def _codepoint(emoji: str) -> str:
    return "-".join(f"{ord(c):x}" for c in emoji if ord(c) != 0xFE0F)


def pick_emoji_events(words: list[Word], max_events: int = 5, min_gap: float = 2.5,
                      show_seconds: float = 1.4) -> list[EmojiEvent]:
    """Choose trigger words, at most ``max_events``, spaced at least ``min_gap`` seconds apart."""
    events: list[EmojiEvent] = []
    last_end = -1e9
    for w in words:
        token = re.sub(r"[^a-z]", "", w.text.lower())
        emoji = KEYWORDS.get(token)
        if emoji and w.start - last_end >= min_gap:
            events.append(EmojiEvent(w.start, w.start + show_seconds, emoji))
            last_end = w.start + show_seconds
            if len(events) >= max_events:
                break
    return events


def fetch_emoji_png(emoji: str, cache_dir: Path) -> Path | None:
    """Download (once) the Twemoji PNG for ``emoji``. Returns None when offline."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{_codepoint(emoji)}.png"
    if target.exists():
        return target
    try:
        with urllib.request.urlopen(TWEMOJI_URL.format(code=_codepoint(emoji)), timeout=15) as resp:
            data = resp.read()
        tmp = target.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(target)
        return target
    except Exception as exc:  # cosmetic feature: never fail a clip over it
        log.warning("could not fetch emoji %s: %s", emoji, exc)
        return None
