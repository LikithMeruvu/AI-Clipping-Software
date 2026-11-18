"""Smart reframing: speaker tracking, active-speaker switching and split-screen layout.

Pipeline
1. ``sample_faces``  reads the clip with OpenCV and runs MediaPipe's FaceLandmarker on a
   sparse, down-scaled sample of frames. Besides the face box it reports the ``jawOpen``
   blendshape, which rises when someone speaks (the active-speaker signal).
2. ``pick_track``    follows identities over time and switches to whoever is talking, with a
   minimum dwell time so the camera does not ping-pong.
3. ``decide_layout`` chooses single-speaker framing, or a stacked split-screen when two
   people are persistently on screen side by side.
4. ``smooth_track`` / ``crop_path`` turn raw positions into gentle per-frame crop offsets
   that the renderer applies through ffmpeg's ``sendcmd`` + ``crop`` filters.

OpenCV and MediaPipe are imported lazily, so the numeric parts stay cheap to test.
"""

from __future__ import annotations

import logging
import os
import statistics
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
LANDMARKER_NAME = "face_landmarker.task"
_model_lock = threading.Lock()


@dataclass(frozen=True)
class FaceSample:
    t: float          # seconds, relative to the clip start
    cx: float         # normalised centre x (0..1)
    cy: float         # normalised centre y (0..1)
    size: float       # normalised face height (0..1)
    score: float      # detector confidence (0..1)
    mouth: float = 0.0  # jawOpen blendshape (0..1): speech activity


@dataclass(frozen=True)
class LayoutPlan:
    mode: str                                # "single" or "split"
    crop_w: int
    src_h: int
    points: list[tuple[float, int]]          # crop x per frame (top panel in split mode)
    points2: list[tuple[float, int]] | None = None  # bottom panel in split mode


def ensure_face_model(models_dir: Path) -> Path:
    """Download the MediaPipe FaceLandmarker model once, atomically and thread-safely."""
    with _model_lock:  # parallel clip renders must not download it concurrently
        models_dir.mkdir(parents=True, exist_ok=True)
        target = models_dir / LANDMARKER_NAME
        if target.exists() and target.stat().st_size > 1_000_000:
            return target
        log.info("Downloading face landmarker model to %s", target)
        fd, tmp = tempfile.mkstemp(dir=models_dir, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as fh, urllib.request.urlopen(LANDMARKER_URL, timeout=60) as resp:
                fh.write(resp.read())
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return target


def sample_faces(video: Path, model_path: Path, samples_per_second: float = 6.0,
                 max_width: int = 640) -> list[FaceSample]:
    """Run the face landmarker over sparse frames of ``video``."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=4,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
    )
    samples: list[FaceSample] = []
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(fps / samples_per_second))

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        index = 0
        while cap.grab():  # grab() decodes without the colour conversion of read()
            if index % stride == 0:
                ok, frame = cap.retrieve()
                if ok:
                    h, w = frame.shape[:2]
                    scale = min(1.0, max_width / w)
                    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                    for i, pts in enumerate(result.face_landmarks):
                        xs = [p.x for p in pts]
                        ys = [p.y for p in pts]
                        mouth = 0.0
                        if result.face_blendshapes and i < len(result.face_blendshapes):
                            mouth = next((c.score for c in result.face_blendshapes[i]
                                          if c.category_name == "jawOpen"), 0.0)
                        samples.append(FaceSample(
                            t=index / fps, cx=(min(xs) + max(xs)) / 2, cy=(min(ys) + max(ys)) / 2,
                            size=max(ys) - min(ys), score=1.0, mouth=float(mouth),
                        ))
            index += 1
    cap.release()
    return samples


def _cluster(cands: list[FaceSample], radius: float = 0.1) -> list[FaceSample]:
    """Merge samples of the same face inside one time step (keep the most active)."""
    merged: list[FaceSample] = []
    for s in sorted(cands, key=lambda s: s.cx):
        if merged and abs(s.cx - merged[-1].cx) < radius:
            if s.mouth > merged[-1].mouth:
                merged[-1] = s
        else:
            merged.append(s)
    return merged


def pick_track(samples: list[FaceSample], duration: float, step: float = 0.25,
               min_dwell: float = 1.0, margin: float = 0.05) -> list[tuple[float, float]]:
    """Follow the active speaker. Returns ``[(t, cx)]`` on a regular grid.

    Each visible face is an identity with a smoothed speech-activity score. The camera
    stays on the current speaker and switches only when another face is clearly more
    active and the dwell time has passed. With no activity signal, the larger face wins.
    """
    by_time: dict[int, list[FaceSample]] = {}
    for s in samples:
        by_time.setdefault(round(s.t / step), []).append(s)

    identities: list[dict] = []  # {"cx", "ema", "size"}
    current: int | None = None
    last_switch = -1e9
    last_cx = 0.5
    track: list[tuple[float, float]] = []

    def weight(ident: dict) -> float:
        return ident["ema"] * 2.0 + ident["size"]

    for i in range(max(1, int(duration / step) + 1)):
        t = i * step
        seen: list[int] = []
        for face in _cluster(by_time.get(i, [])):
            match = min(range(len(identities)), key=lambda k: abs(identities[k]["cx"] - face.cx), default=None)
            if match is None or abs(identities[match]["cx"] - face.cx) > 0.15:
                identities.append({"cx": face.cx, "ema": face.mouth, "size": face.size})
                match = len(identities) - 1
            else:
                ident = identities[match]
                ident.update(cx=face.cx, size=face.size, ema=0.7 * ident["ema"] + 0.3 * face.mouth)
            seen.append(match)
        if seen:
            best = max(seen, key=lambda k: weight(identities[k]))
            if current is None or current not in seen:
                current, last_switch = best, t
            elif best != current and t - last_switch >= min_dwell \
                    and weight(identities[best]) > weight(identities[current]) + margin:
                current, last_switch = best, t
            last_cx = identities[current]["cx"]
        track.append((t, last_cx))
    return track


def decide_layout(samples: list[FaceSample], duration: float, step: float = 0.25,
                  min_separation: float = 0.25, min_ratio: float = 0.35) -> str:
    """``"split"`` when two people are persistently on screen side by side, else ``"single"``."""
    by_time: dict[int, list[FaceSample]] = {}
    for s in samples:
        by_time.setdefault(round(s.t / step), []).append(s)
    total = max(1, int(duration / step) + 1)
    pairs = 0
    for cands in by_time.values():
        faces = _cluster(cands)
        if len(faces) >= 2 and faces[-1].cx - faces[0].cx >= min_separation \
                and min(faces[0].size, faces[-1].size) > 0.08:
            pairs += 1
    return "split" if pairs >= 4 and pairs / total >= min_ratio else "single"


def pick_pair_tracks(samples: list[FaceSample], duration: float,
                     step: float = 0.25) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Left and right face positions on a regular grid (hold last known when a face drops out)."""
    by_time: dict[int, list[FaceSample]] = {}
    for s in samples:
        by_time.setdefault(round(s.t / step), []).append(s)
    left_x, right_x = 0.3, 0.7
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    for i in range(max(1, int(duration / step) + 1)):
        faces = _cluster(by_time.get(i, []))
        if len(faces) >= 2:
            left_x, right_x = faces[0].cx, faces[-1].cx
        elif len(faces) == 1:  # keep the face on its own side
            if abs(faces[0].cx - left_x) <= abs(faces[0].cx - right_x):
                left_x = faces[0].cx
            else:
                right_x = faces[0].cx
        left.append((i * step, left_x))
        right.append((i * step, right_x))
    return left, right


def smooth_track(track: list[tuple[float, float]], median_window: int = 5,
                 mean_window_s: float = 1.2, step: float = 0.25,
                 dead_zone: float = 0.02) -> list[tuple[float, float]]:
    """Median filter (kills outliers), moving average (smooth pans), dead-zone (no jitter)."""
    if not track:
        return track
    xs = [x for _, x in track]
    half_m = median_window // 2
    med = [statistics.median(xs[max(0, i - half_m): i + half_m + 1]) for i in range(len(xs))]
    half = max(1, round(mean_window_s / step / 2))
    smooth = []
    for i in range(len(med)):
        window = med[max(0, i - half): i + half + 1]
        smooth.append(sum(window) / len(window))
    out, held = [], smooth[0]
    for (t, _), value in zip(track, smooth):
        if abs(value - held) > dead_zone:
            held = value
        out.append((t, held))
    return out


def crop_path(track: list[tuple[float, float]], src_w: int, src_h: int, out_aspect: float,
              fps: float, duration: float) -> tuple[int, list[tuple[float, int]]]:
    """Convert a smoothed track into ``(crop_w, [(t, x_left)])`` per output frame.

    When the source is narrower than the target aspect the caller should letterbox:
    crop_w == src_w and every x_left is 0.
    """
    crop_w = int(round(src_h * out_aspect))
    crop_w -= crop_w % 2
    if crop_w >= src_w:
        return src_w - (src_w % 2), [(0.0, 0)]
    if not track:
        track = [(0.0, 0.5)]
    times = [t for t, _ in track]
    values = [v for _, v in track]
    points: list[tuple[float, int]] = []
    for f in range(max(1, int(duration * fps))):
        t = f / fps
        cx = _interp(t, times, values)
        x_left = max(0, min(src_w - crop_w, int(round(cx * src_w - crop_w / 2))))
        points.append((t, x_left - x_left % 2))
    return crop_w, points


def _interp(t: float, times: list[float], values: list[float]) -> float:
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    lo, hi = 0, len(times) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if times[mid] <= t:
            lo = mid
        else:
            hi = mid
    span = times[hi] - times[lo]
    alpha = 0.0 if span == 0 else (t - times[lo]) / span
    return values[lo] + (values[hi] - values[lo]) * alpha


def sendcmd_script(points: list[tuple[float, int]], target: str = "crop") -> str:
    """ffmpeg sendcmd script that moves a crop origin per frame, e.g. ``0.033 crop x 120;``."""
    return "\n".join(f"{t:.4f} {target} x {x};" for t, x in points) + "\n"


def analyse_layout(video: Path, duration: float, out_w: int, out_h: int, fps: float,
                   models_dir: Path, layout: str = "auto") -> LayoutPlan | None:
    """Plan the framing for a clip. ``None`` means the source needs no cropping (letterbox)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    aspect = out_w / out_h
    if src_w / src_h <= aspect:
        return None

    samples = sample_faces(video, ensure_face_model(models_dir))
    mode = layout if layout in ("single", "split") else decide_layout(samples, duration)
    if mode == "split":
        half_aspect = out_w / (out_h / 2)
        if round(src_h * half_aspect) < src_w:  # a split panel must still be a crop, not a pad
            left, right = pick_pair_tracks(samples, duration)
            crop_w, p1 = crop_path(smooth_track(left), src_w, src_h, half_aspect, fps, duration)
            _, p2 = crop_path(smooth_track(right), src_w, src_h, half_aspect, fps, duration)
            return LayoutPlan("split", crop_w, src_h, p1, p2)
    crop_w, points = crop_path(smooth_track(pick_track(samples, duration)), src_w, src_h, aspect, fps, duration)
    return LayoutPlan("single", crop_w, src_h, points)
