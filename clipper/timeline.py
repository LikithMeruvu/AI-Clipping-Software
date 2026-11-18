"""Timeline maths: silence removal (jump cuts) and mapping source time to output time."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .captions import Word


def keep_intervals(
    words: Sequence[Word],
    start: float,
    end: float,
    max_gap: float = 0.45,
    pad: float = 0.10,
) -> list[tuple[float, float]]:
    """Source-time intervals to keep for a clip, with long pauses removed.

    A pause longer than ``max_gap`` seconds between two spoken words is cut. Each
    kept speech block keeps ``pad`` seconds of breathing room on both sides. The
    result is always non-empty for a clip that has speech, and falls back to the
    whole range when there are no words.
    """
    inside = [w for w in words if w.end > start and w.start < end]
    if not inside:
        return [(start, end)]

    blocks: list[list[Word]] = [[inside[0]]]
    for prev, cur in zip(inside, inside[1:]):
        if cur.start - prev.end > max_gap:
            blocks.append([cur])
        else:
            blocks[-1].append(cur)

    intervals: list[tuple[float, float]] = []
    for block in blocks:
        a = max(start, block[0].start - pad)
        b = min(end, block[-1].end + pad)
        if intervals and a <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], b))
        else:
            intervals.append((a, b))
    return intervals


@dataclass(frozen=True)
class TimeMap:
    """Maps source timestamps onto the concatenated output timeline."""

    intervals: tuple[tuple[float, float], ...]

    @classmethod
    def from_intervals(cls, intervals: Sequence[tuple[float, float]]) -> TimeMap:
        return cls(tuple(intervals))

    @property
    def duration(self) -> float:
        return sum(b - a for a, b in self.intervals)

    def to_output(self, t: float) -> float:
        """Output time for source time ``t``. Times inside a removed gap snap to the next kept frame."""
        elapsed = 0.0
        for a, b in self.intervals:
            if t < a:
                return elapsed  # inside a cut: snap to the start of the next kept block
            if t <= b:
                return elapsed + (t - a)
            elapsed += b - a
        return elapsed

    def map_words(self, words: Sequence[Word]) -> list[Word]:
        """Re-time words to the output timeline, dropping any that fall entirely in a cut."""
        out: list[Word] = []
        for w in words:
            if not any(a < w.end and w.start < b for a, b in self.intervals):
                continue
            start = self.to_output(w.start)
            end = max(start, self.to_output(w.end))
            out.append(Word(start=start, end=end, text=w.text))
        return out
