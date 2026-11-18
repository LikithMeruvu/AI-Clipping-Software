import pytest
from pydantic import BaseModel

from clipper import llm as llm_mod
from clipper.captions import Word
from clipper.pipeline import IngestError, validate_youtube_url
from clipper.reframe import FaceSample, crop_path, pick_track, sendcmd_script, smooth_track
from clipper.selection import ClipPlan, _snap, select_clips, select_heuristic
from clipper.transcribe import Segment, Transcript

# ---------- reframe ----------

def test_smooth_track_dead_zone_suppresses_jitter():
    track = [(i * 0.25, 0.5 + (0.005 if i % 2 else -0.005)) for i in range(20)]
    out = smooth_track(track)
    assert len({round(x, 6) for _, x in out}) == 1


def test_pick_track_prefers_large_confident_face_and_holds_when_missing():
    samples = [FaceSample(0.0, 0.3, 0.5, 0.1, 0.9), FaceSample(0.0, 0.7, 0.5, 0.4, 0.9)]
    track = pick_track(samples, duration=1.0)
    assert track[0][1] == 0.7
    assert all(x == 0.7 for _, x in track)  # no detections later: camera holds position


def test_crop_path_stays_inside_frame_and_even():
    track = [(0.0, 0.0), (1.0, 1.0)]
    crop_w, pts = crop_path(track, 1920, 1080, 9 / 16, fps=30, duration=1.0)
    assert crop_w == 608
    assert all(0 <= x <= 1920 - crop_w and x % 2 == 0 for _, x in pts)
    assert len(pts) == 30


def test_crop_path_narrow_source_needs_no_crop():
    crop_w, pts = crop_path([(0.0, 0.5)], 600, 1080, 9 / 16, 30, 1.0)
    assert crop_w == 600 and pts == [(0.0, 0)]


def test_sendcmd_script_format():
    assert sendcmd_script([(0.0, 10), (0.0333, 12)]) == "0.0000 crop x 10;\n0.0333 crop x 12;\n"


# ---------- selection ----------

def words(n=60, step=1.0):
    return [Word(i * step, i * step + 0.8, f"w{i}") for i in range(n)]


def transcript(n_seg=20):
    segs = [Segment(i * 5.0, i * 5.0 + 4.5, f"Why is this the biggest mistake {i}? Never do it!") for i in range(n_seg)]
    return Transcript("en", n_seg * 5.0, segs, words(n_seg * 5))


def test_snap_moves_to_word_boundaries_and_enforces_window():
    ws = words()
    s, e = _snap(10.2, 40.9, ws, 20, 60, 60.0)
    assert s == 10.0 and e == pytest.approx(40.8)
    assert _snap(10, 15, ws, 20, 60, 60.0) is not None  # extended to reach min duration


def test_heuristic_clips_do_not_overlap_and_respect_duration():
    clips = select_heuristic(transcript(), 3, 15, 30)
    assert 1 <= len(clips) <= 3
    for a, b in zip(clips, clips[1:]):
        assert a.end <= b.start
    assert all(15 <= c.duration <= 30 for c in clips)


class FakeLLM:
    name = "fake"

    def __init__(self, plan=None, error=None):
        self.plan, self.error = plan, error

    def generate(self, prompt, schema, temperature=0.3):
        if self.error:
            raise self.error
        return self.plan


def cand(start, end, score):
    return {"start": start, "end": end, "score": score, "hook": "h", "title": "t",
            "description": "d", "hashtags": ["x"], "reason": "r"}


def test_llm_selection_used_and_overlaps_resolved_by_score():
    plan = ClipPlan.model_validate({"clips": [cand(0, 25, 70), cand(10, 35, 90), cand(50, 75, 60)]})
    res = select_clips(transcript(), FakeLLM(plan), 3, 15, 30)
    assert res.method == "llm" and not res.warnings
    assert [(c.start, c.score) for c in res.clips] == [(10.0, 90), (50.0, 60)]
    assert res.clips[0].hashtags == ["#x"]


def test_llm_failure_falls_back_visibly_not_silently():
    res = select_clips(transcript(), FakeLLM(error=llm_mod.LLMError("model not found")), 2, 15, 30)
    assert res.method == "heuristic"
    assert res.warnings and "model not found" in res.warnings[0]
    assert res.clips


# ---------- llm retry loop ----------

class Reply(BaseModel):
    n: int


class Scripted(llm_mod._ValidatingMixin):
    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)

    def _call(self, prompt, schema, temperature):
        return self.replies.pop(0)


def test_invalid_reply_is_retried_then_accepted():
    assert Scripted(["not json", '```json\n{"n": 3}\n```']).generate("p", Reply).n == 3


def test_never_valid_raises_llm_error():
    with pytest.raises(llm_mod.LLMError):
        Scripted(['{"n": "x"}'] * 3).generate("p", Reply)


def test_build_llm_requires_key():
    from clipper.config import Settings
    with pytest.raises(llm_mod.LLMError):
        llm_mod.build_llm(Settings(_env_file=None, gemini_api_key=None))


# ---------- ingest validation ----------

@pytest.mark.parametrize("url,vid", [
    ("https://www.youtube.com/watch?v=ZalTrOoh4sc", "ZalTrOoh4sc"),
    ("https://youtu.be/ZalTrOoh4sc?t=5", "ZalTrOoh4sc"),
    ("https://www.youtube.com/shorts/ZalTrOoh4sc", "ZalTrOoh4sc"),
])
def test_valid_youtube_urls(url, vid):
    assert validate_youtube_url(url) == vid


@pytest.mark.parametrize("url", [
    "https://evil.com/watch?v=ZalTrOoh4sc", "file:///etc/passwd", "https://youtube.com.evil.com/watch?v=abcdefghij",
    "https://www.youtube.com/watch", "ftp://youtu.be/abcdefghij",
])
def test_rejected_urls(url):
    with pytest.raises(IngestError):
        validate_youtube_url(url)


# ---------- active speaker and layout ----------

def _two_faces(t, mouth_left, mouth_right):
    return [FaceSample(t, 0.25, 0.5, 0.3, 1.0, mouth_left), FaceSample(t, 0.75, 0.5, 0.3, 1.0, mouth_right)]


def test_active_speaker_camera_follows_whoever_talks_with_dwell_time():
    from clipper.reframe import pick_track
    samples = []
    for i in range(40):  # 10 s at 0.25 s steps: left talks first half, right second half
        t = i * 0.25
        samples += _two_faces(t, 0.5 if t < 5 else 0.0, 0.0 if t < 5 else 0.5)
    track = dict(pick_track(samples, 10.0))
    assert track[2.0] == 0.25 and track[9.0] == 0.75
    switches = sum(1 for a, b in zip(list(track.values()), list(track.values())[1:]) if a != b)
    assert switches == 1  # no ping-pong


def test_decide_layout_split_only_for_persistent_side_by_side_faces():
    from clipper.reframe import decide_layout, pick_pair_tracks
    both = [s for i in range(40) for s in _two_faces(i * 0.25, 0.1, 0.1)]
    assert decide_layout(both, 10.0) == "split"
    one = [FaceSample(i * 0.25, 0.5, 0.5, 0.3, 1.0) for i in range(40)]
    assert decide_layout(one, 10.0) == "single"
    left, right = pick_pair_tracks(both, 10.0)
    assert left[5][1] == 0.25 and right[5][1] == 0.75


def test_sendcmd_script_supports_named_targets():
    assert sendcmd_script([(0.0, 4)], "crop@a") == "0.0000 crop@a x 4;\n"


# ---------- emoji and dubbing helpers ----------

def test_emoji_events_respect_spacing_and_cap():
    from clipper.emoji import pick_emoji_events
    ws = [Word(1, 1.2, "Money!"), Word(1.5, 1.7, "crazy"), Word(5, 5.2, "wow"), Word(9, 9.2, "fire")]
    ev = pick_emoji_events(ws, max_events=2, min_gap=2.5)
    assert [e.start for e in ev] == [1, 5]  # "crazy" too close to "money"; cap of 2 honoured


def test_atempo_chain_splits_ratios_outside_supported_range():
    from clipper.dub import atempo_chain
    assert atempo_chain(1.25) == "atempo=1.2500"
    assert atempo_chain(3.0).startswith("atempo=2.0,")


# ---------- B-roll ----------

def test_pick_video_file_prefers_hd_portrait_mp4():
    from clipper.broll import pick_video_file
    payload = {"videos": [{"video_files": [
        {"file_type": "video/mp4", "width": 360, "height": 640, "link": "low"},
        {"file_type": "video/mp4", "width": 1080, "height": 1920, "link": "hd"},
        {"file_type": "video/webm", "width": 1080, "height": 1920, "link": "webm"}]}]}
    assert pick_video_file(payload) == "hd"
    assert pick_video_file({"videos": []}) is None


def test_plan_broll_enforces_rules():
    from clipper.broll import BrollPlan, plan_broll
    plan = BrollPlan.model_validate({"items": [
        {"start": 0.5, "duration": 2, "query": "too early"},
        {"start": 5, "duration": 9, "query": "city"},          # duration clamped to 3
        {"start": 6, "duration": 2, "query": "overlaps"},       # < 3 s after the previous one
        {"start": 28, "duration": 2, "query": "too late"}]})
    got = plan_broll(FakeLLM(plan), [(0, 30, "x")], clip_seconds=30)
    assert [(g.query, g.duration) for g in got] == [("city", 3.0)]


def test_fetch_broll_downloads_once_and_survives_failure(tmp_path):
    import io
    import json

    from clipper.broll import fetch_broll

    calls = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=0):
        url = getattr(req, "full_url", req)
        calls.append(url)
        if "api.pexels" in url:
            return Resp(json.dumps({"videos": [{"video_files": [
                {"file_type": "video/mp4", "width": 1080, "height": 1920, "link": "https://cdn/x.mp4"}]}]}).encode())
        return Resp(b"VIDEO")

    path = fetch_broll("city", "key", tmp_path, opener)
    assert path and path.read_bytes() == b"VIDEO"
    assert fetch_broll("city", "key", tmp_path, opener) == path and len(calls) == 2  # cached second time

    def boom(*a, **k):
        raise OSError("offline")

    assert fetch_broll("other", "key", tmp_path, boom) is None
