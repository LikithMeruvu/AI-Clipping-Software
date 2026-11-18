import re

from clipper.captions import STYLES, Word, build_ass, build_srt, group_words
from clipper.timeline import TimeMap, keep_intervals


def words_from(spec):
    return [Word(start=s, end=e, text=t) for s, e, t in spec]


# ---------- silence removal and time mapping ----------

def test_keep_intervals_cuts_long_pauses_and_keeps_padding():
    ws = words_from([(0.0, 0.5, "hello"), (0.6, 1.0, "there"), (4.0, 4.4, "again")])
    intervals = keep_intervals(ws, 0.0, 10.0, max_gap=0.45, pad=0.1)
    assert intervals == [(0.0, 1.1), (3.9, 4.5)]


def test_keep_intervals_without_words_keeps_whole_range():
    assert keep_intervals([], 5.0, 9.0) == [(5.0, 9.0)]


def test_time_map_duration_equals_sum_of_kept_intervals():
    tm = TimeMap.from_intervals([(0.0, 1.1), (3.9, 4.5)])
    assert abs(tm.duration - 1.7) < 1e-9


def test_time_map_shifts_later_times_by_removed_gap():
    tm = TimeMap.from_intervals([(0.0, 1.1), (3.9, 4.5)])
    assert tm.to_output(0.5) == 0.5
    assert abs(tm.to_output(4.2) - (1.1 + 0.3)) < 1e-9


def test_time_map_drops_words_entirely_inside_a_cut():
    tm = TimeMap.from_intervals([(0.0, 1.1), (3.9, 4.5)])
    ws = words_from([(0.0, 0.5, "a"), (2.0, 2.5, "gone"), (4.0, 4.4, "b")])
    mapped = tm.map_words(ws)
    assert [w.text for w in mapped] == ["a", "b"]
    assert mapped[1].start >= mapped[0].end


# ---------- captions ----------

def test_group_words_breaks_on_sentence_end_and_pause():
    style = STYLES["hormozi"]
    ws = words_from([(0, 0.3, "one"), (0.3, 0.6, "two."), (0.6, 0.9, "three"),
                     (2.5, 2.8, "four")])
    groups = group_words(ws, style)
    assert [len(g.words) for g in groups] == [2, 1, 1]


def test_group_words_respects_max_words_and_chars():
    style = STYLES["hormozi"]  # max 3 words
    ws = words_from([(i * 0.2, i * 0.2 + 0.2, f"w{i}") for i in range(7)])
    assert all(len(g.words) <= style.max_words for g in group_words(ws, style))


def test_ass_has_play_resolution_and_karaoke_tags_that_sum_to_group_length():
    style = STYLES["hormozi"]
    ws = words_from([(0.0, 0.4, "hello"), (0.4, 0.9, "world")])
    lines = group_words(ws, style)
    ass = build_ass(lines, width=1080, height=1920, style=style)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    karaoke = [int(x) for x in re.findall(r"\\kf(\d+)", ass)]
    assert karaoke == [40, 50]  # centiseconds, one tag per word


def test_ass_escapes_braces_and_newlines_in_transcript_text():
    style = STYLES["clean"]
    ws = words_from([(0.0, 0.5, "{evil}")])
    ass = build_ass(group_words(ws, style), width=1080, height=1920, style=style)
    dialogue = next(ln for ln in ass.splitlines() if ln.startswith("Dialogue"))
    assert "{evil}" not in dialogue and "(evil)" in dialogue


def test_hook_and_progress_bar_are_emitted_when_requested():
    style = STYLES["boxed"]
    ass = build_ass([], width=1080, height=1920, style=style, hook="Stop scrolling now",
                    progress_bar=True, duration=12.0)
    assert ",Hook," in ass and "Stop scrolling now" in ass
    assert ",Bar," in ass and "\\t(0,12000," in ass


def test_srt_uses_comma_milliseconds_and_numbers_cues():
    style = STYLES["minimal"]
    ws = words_from([(1.0, 1.5, "hi"), (1.5, 2.25, "there")])
    srt = build_srt(group_words(ws, style), style)
    assert srt.startswith("1\n00:00:01,000 --> 00:00:02,250\nHI THERE".replace("HI THERE", "hi there")) or \
        "00:00:01,000 --> 00:00:02,250" in srt
    assert srt.splitlines()[0] == "1"
