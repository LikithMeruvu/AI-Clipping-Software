"""Choose the best clip windows from a transcript.

Primary path: an LLM (any configured provider) returns structured candidates with a
virality score, a hook, a title, a description and hashtags. Long videos are split
into windows so no single prompt grows without bound.

Fallback path: a deterministic heuristic (speech density, questions, numbers and
hook words). The fallback is always reported in ``SelectionResult.method`` and
``warnings``, so a failed LLM never produces silently random clips again.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .captions import Word
from .llm import LLM, LLMError
from .transcribe import Segment, Transcript

log = logging.getLogger(__name__)

WINDOW_SECONDS = 1500  # 25-minute windows keep prompts bounded on long videos


class ClipCandidate(BaseModel):
    start: float = Field(description="Start time in seconds. Must equal the start of a transcript line.")
    end: float = Field(description="End time in seconds. Must equal the end of a transcript line.")
    score: int = Field(ge=0, le=100, description="Virality score 0-100 (hook strength, payoff, shareability).")
    hook: str = Field(description="Short on-screen hook headline, at most 8 words, no quotes.")
    title: str = Field(description="Post title, at most 70 characters.")
    description: str = Field(description="One or two sentences for the post caption.")
    hashtags: list[str] = Field(description="3 to 6 hashtags, each starting with #, no spaces.")
    reason: str = Field(description="One sentence on why this moment works.")


class ClipPlan(BaseModel):
    clips: list[ClipCandidate]


@dataclass(frozen=True)
class Clip:
    start: float
    end: float
    score: int
    hook: str
    title: str
    description: str
    hashtags: list[str]
    reason: str
    method: str  # "llm" or "heuristic"

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SelectionResult:
    method: str
    clips: list[Clip]
    warnings: list[str] = field(default_factory=list)


def _fmt_transcript(segments: list[Segment]) -> str:
    return "\n".join(f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in segments)


def _prompt(segments: list[Segment], n: int, min_d: float, max_d: float, focus: str | None,
            language: str, window: tuple[float, float]) -> str:
    focus_line = f"Creator focus (prioritise moments matching this): {focus}\n" if focus else ""
    return (
        "You are an expert short-form video editor who finds moments that go viral on TikTok, "
        "Instagram Reels and YouTube Shorts.\n"
        f"Choose up to {n} non-overlapping clips from this transcript window "
        f"({window[0]:.0f}s to {window[1]:.0f}s).\n"
        f"Each clip must be between {min_d:.0f} and {max_d:.0f} seconds long.\n"
        "Rules:\n"
        "- Start on a line that opens a complete thought, and end on the line that completes it.\n"
        "- The first 3 seconds must hook the viewer (a bold claim, question, surprise or story setup).\n"
        "- Prefer self-contained moments that make sense without the rest of the video.\n"
        "- Use only start and end times that appear in the transcript.\n"
        f"- Write titles, hooks and descriptions in the transcript language ({language}).\n"
        f"{focus_line}\nTranscript:\n{_fmt_transcript(segments)}\n"
    )


def _snap(start: float, end: float, words: list[Word], min_d: float, max_d: float,
          duration: float) -> tuple[float, float] | None:
    """Snap LLM times to real word boundaries and enforce the duration window."""
    if not words:
        return (start, end) if min_d <= end - start <= max_d else None
    starts = [w.start for w in words]
    ends = [w.end for w in words]
    near_start = min(starts, key=lambda s: abs(s - start))
    if abs(near_start - start) <= 1.5:
        start = near_start
    near_end = min(ends, key=lambda e: abs(e - end))
    if abs(near_end - end) <= 1.5:
        end = near_end
    start = max(0.0, min(start, duration))
    end = min(duration, end)
    if end - start > max_d:
        candidates = [e for e in ends if start + min_d <= e <= start + max_d]
        if not candidates:
            return None
        end = max(candidates)
    if end - start < min_d:
        candidates = [e for e in ends if e >= start + min_d and e <= min(duration, start + max_d)]
        if not candidates:
            return None
        end = min(candidates)
    return (start, end) if start < end else None


def _resolve_overlaps(clips: list[Clip], n: int) -> list[Clip]:
    chosen: list[Clip] = []
    for clip in sorted(clips, key=lambda c: c.score, reverse=True):
        if all(clip.end <= c.start or clip.start >= c.end for c in chosen):
            chosen.append(clip)
        if len(chosen) == n:
            break
    return sorted(chosen, key=lambda c: c.start)


def _windows(duration: float) -> list[tuple[float, float]]:
    if duration <= WINDOW_SECONDS * 1.25:
        return [(0.0, duration)]
    out, t = [], 0.0
    while t < duration:
        out.append((t, min(duration, t + WINDOW_SECONDS)))
        t += WINDOW_SECONDS
    return out


def select_with_llm(transcript: Transcript, llm: LLM, n: int, min_d: float, max_d: float,
                    focus: str | None = None) -> list[Clip]:
    clips: list[Clip] = []
    windows = _windows(transcript.duration)
    per_window = max(2, round(n * 1.5 / len(windows)) + 1)
    for lo, hi in windows:
        segs = [s for s in transcript.segments if s.end > lo and s.start < hi]
        if not segs:
            continue
        prompt = _prompt(segs, per_window, min_d, max_d, focus, transcript.language, (lo, hi))
        plan = llm.generate(prompt, ClipPlan, temperature=0.4)
        for cand in plan.clips:
            snapped = _snap(cand.start, cand.end, transcript.words, min_d, max_d, transcript.duration)
            if snapped is None:
                log.info("dropping candidate %.1f-%.1f: outside duration window", cand.start, cand.end)
                continue
            s, e = snapped
            clips.append(Clip(
                start=s, end=e, score=cand.score, hook=cand.hook.strip(), title=cand.title.strip(),
                description=cand.description.strip(), hashtags=[h if h.startswith("#") else f"#{h}" for h in cand.hashtags][:6],
                reason=cand.reason.strip(), method="llm",
            ))
    return _resolve_overlaps(clips, n)


_HOOK_WORDS = re.compile(r"\b(secret|never|mistake|truth|why|how|nobody|everyone|stop|biggest|worst|best|free|money)\b", re.I)


def select_heuristic(transcript: Transcript, n: int, min_d: float, max_d: float) -> list[Clip]:
    """Deterministic fallback: score sliding windows over segment boundaries."""
    segs = transcript.segments
    scored: list[tuple[float, float, float, str]] = []
    for i, first in enumerate(segs):
        j = i
        while j < len(segs) and segs[j].end - first.start <= max_d:
            if segs[j].end - first.start >= min_d:
                text = " ".join(s.text for s in segs[i : j + 1])
                words = max(1, len(text.split()))
                duration = segs[j].end - first.start
                density = words / duration
                bonus = 0.0
                bonus += 1.5 * text.count("?") + 1.0 * text.count("!")
                bonus += 0.5 * len(_HOOK_WORDS.findall(text))
                bonus += 0.5 * len(re.findall(r"\d", text)) / 2
                scored.append((density + bonus, first.start, segs[j].end, text))
            j += 1
    scored.sort(reverse=True)
    top = max((s for s, *_ in scored), default=1.0) or 1.0

    clips: list[Clip] = []
    for score, start, end, text in scored:
        if any(not (end <= c.start or start >= c.end) for c in clips):
            continue
        first_sentence = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
        clips.append(Clip(
            start=start, end=end, score=int(round(100 * score / top * 0.8)),
            hook=" ".join(first_sentence.split()[:8]), title=first_sentence[:70],
            description=text[:200], hashtags=["#shorts", "#viral"], reason="heuristic: speech density and hook words",
            method="heuristic",
        ))
        if len(clips) == n:
            break
    return sorted(clips, key=lambda c: c.start)


def select_clips(transcript: Transcript, llm: LLM | None, n: int, min_d: float, max_d: float,
                 focus: str | None = None) -> SelectionResult:
    """Run the LLM selection, falling back to the heuristic with a visible warning."""
    if llm is not None:
        try:
            clips = select_with_llm(transcript, llm, n, min_d, max_d, focus)
            if clips:
                return SelectionResult(method="llm", clips=clips)
            warning = "LLM returned no usable clips inside the duration window; used heuristic selection"
        except LLMError as exc:
            warning = f"LLM selection failed ({exc}); used heuristic selection"
        except Exception as exc:  # network/SDK failures of any kind
            warning = f"LLM selection error ({type(exc).__name__}: {exc}); used heuristic selection"
        log.warning(warning)
        return SelectionResult(method="heuristic", clips=select_heuristic(transcript, n, min_d, max_d),
                               warnings=[warning])
    return SelectionResult(method="heuristic", clips=select_heuristic(transcript, n, min_d, max_d),
                           warnings=["no LLM configured; used heuristic selection"])
