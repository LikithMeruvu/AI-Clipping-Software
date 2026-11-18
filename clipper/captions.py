"""Caption generation: ASS (burned in by libass) and SRT (sidecar file).

ASS is used instead of per-frame image rendering because libass rasterises text
inside ffmpeg itself, which is an order of magnitude faster than drawing each
frame in Python. Karaoke highlighting uses ``\\k`` tags, so the active word is
highlighted without any extra events.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    start: float  # seconds, output timeline
    end: float
    text: str


@dataclass(frozen=True)
class Line:
    """One caption block. ``words`` enables karaoke; ``text`` is a plain (translated) line."""

    start: float
    end: float
    words: tuple[Word, ...] | None = None
    text: str | None = None


@dataclass(frozen=True)
class CaptionStyle:
    key: str
    label: str
    font: str
    size: int            # px at 1080 px output width
    highlight: str       # ASS colour (&HAABBGGRR) of the word being spoken
    base: str            # colour of words not yet spoken
    outline: str
    outline_width: float
    shadow: float
    box: bool            # opaque background box instead of outline
    uppercase: bool
    max_words: int
    max_chars: int


STYLES: dict[str, CaptionStyle] = {
    s.key: s
    for s in (
        CaptionStyle("hormozi", "Bold highlight (viral)", "Arial Black", 84, "&H0000FFFF", "&H00FFFFFF",
                     "&H00000000", 7, 0, False, True, 3, 22),
        CaptionStyle("clean", "Clean white", "Arial", 72, "&H00FFFFFF", "&H00FFFFFF",
                     "&H00000000", 4, 2, False, False, 5, 32),
        CaptionStyle("boxed", "Boxed", "Arial", 70, "&H0000FFFF", "&H00FFFFFF",
                     "&HA0000000", 0, 0, True, False, 4, 26),
        CaptionStyle("neon", "Neon", "Arial Black", 80, "&H00FFFF00", "&H00FF00FF",
                     "&H00200020", 6, 0, False, True, 3, 22),
        CaptionStyle("minimal", "Minimal", "Arial", 60, "&H00FFFFFF", "&H00DDDDDD",
                     "&H00000000", 2, 1, False, False, 6, 40),
    )
}

_SENTENCE_END = re.compile(r"[.!?][\"')\]]*$")


def _ass_time(seconds: float) -> str:
    cs = max(0, round(seconds * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _srt_time(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _escape(text: str) -> str:
    """Make arbitrary transcript text safe inside an ASS dialogue line."""
    text = text.replace("\\", "/").replace("{", "(").replace("}", ")")
    return re.sub(r"\s+", " ", text).strip()


def _display(text: str, style: CaptionStyle) -> str:
    return _escape(text.upper() if style.uppercase else text)


def group_words(words: Sequence[Word], style: CaptionStyle, max_gap: float = 0.6) -> list[Line]:
    """Split a word stream into on-screen caption groups.

    A group ends at a sentence boundary, a long pause, or the style's word/char limit.
    """
    lines: list[Line] = []
    current: list[Word] = []
    chars = 0

    def flush() -> None:
        if current:
            lines.append(Line(start=current[0].start, end=current[-1].end, words=tuple(current)))

    for word in words:
        if not word.text.strip():
            continue
        if current:
            too_long = len(current) >= style.max_words or chars + len(word.text) + 1 > style.max_chars
            pause = word.start - current[-1].end > max_gap
            if too_long or pause or _SENTENCE_END.search(current[-1].text):
                flush()
                current, chars = [], 0
        current.append(word)
        chars += len(word.text) + 1
    flush()
    return lines


def _wrap(text: str, width: int = 20) -> str:
    out, line = [], ""
    for token in text.split():
        if line and len(line) + 1 + len(token) > width:
            out.append(line)
            line = token
        else:
            line = f"{line} {token}".strip()
    if line:
        out.append(line)
    return "\\N".join(out)


def build_ass(
    lines: Iterable[Line],
    *,
    width: int,
    height: int,
    style: CaptionStyle,
    hook: str | None = None,
    hook_seconds: float = 3.0,
    progress_bar: bool = False,
    duration: float | None = None,
) -> str:
    """Render caption lines, an optional hook headline and an optional progress bar to ASS."""
    scale = width / 1080
    size = max(24, round(style.size * scale))
    outline = round(style.outline_width * scale, 2)
    shadow = round(style.shadow * scale, 2)
    border_style = 3 if style.box else 1
    margin_v = round(height * 0.16)
    hook_size = max(28, round(size * 0.85))

    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Caption,{style.font},{size},{style.highlight},{style.base},{style.outline},"
        f"&HA0000000,-1,0,0,0,100,100,0,0,{border_style},{outline},{shadow},2,60,60,{margin_v},1",
        f"Style: Hook,{style.font},{hook_size},&H00FFFFFF,&H00FFFFFF,{style.outline},&HB0000000,"
        f"-1,0,0,0,100,100,0,0,3,{outline},0,8,60,60,{round(height * 0.12)},1",
        f"Style: Bar,Arial,10,{style.highlight},{style.highlight},&H00000000,&H00000000,0,0,0,0,"
        "100,100,0,0,1,0,0,1,0,0,0,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    events: list[str] = []
    end_of_video = duration or 0.0

    for line in lines:
        end_of_video = max(end_of_video, line.end)
        if line.words:
            parts = []
            words = line.words
            for i, word in enumerate(words):
                next_start = words[i + 1].start if i + 1 < len(words) else word.end
                cs = max(1, round((next_start - word.start) * 100))
                suffix = " " if i + 1 < len(words) else ""
                parts.append(f"{{\\kf{cs}}}{_display(word.text, style)}{suffix}")
            text = "".join(parts)
        else:
            text = _display(line.text or "", style)
        events.append(
            f"Dialogue: 0,{_ass_time(line.start)},{_ass_time(line.end)},Caption,,0,0,0,,"
            f"{{\\fad(80,40)}}{text}"
        )

    if hook:
        events.append(
            f"Dialogue: 1,{_ass_time(0)},{_ass_time(hook_seconds)},Hook,,0,0,0,,"
            f"{{\\fad(120,200)}}{_wrap(_escape(hook))}"
        )

    if progress_bar and end_of_video > 0:
        bar_w, bar_h = width, max(6, round(height * 0.008))
        drawing = f"m 0 0 l {bar_w} 0 {bar_w} {bar_h} 0 {bar_h}"
        total_ms = round(end_of_video * 1000)
        events.append(
            f"Dialogue: 2,{_ass_time(0)},{_ass_time(end_of_video)},Bar,,0,0,0,,"
            f"{{\\an4\\pos(0,{height - bar_h})\\p1\\fscx0\\t(0,{total_ms},\\fscx100)}}{drawing}"
        )

    return "\n".join(header + events) + "\n"


def build_srt(lines: Iterable[Line], style: CaptionStyle | None = None) -> str:
    """Sidecar SRT with one cue per caption line (word-level lines are joined)."""
    blocks = []
    for index, line in enumerate(lines, 1):
        if line.words:
            text = " ".join(w.text for w in line.words)
        else:
            text = line.text or ""
        if style and style.uppercase:
            text = text.upper()
        blocks.append(f"{index}\n{_srt_time(line.start)} --> {_srt_time(line.end)}\n{text.strip()}\n")
    return "\n".join(blocks)
