"""Smart 9:16 (or any aspect) reframing that keeps the speaker centred.

Pipeline:
1. ``sample_faces`` reads the clip sequentially with OpenCV and runs MediaPipe's
   Tasks face detector on a sparse, down-scaled sample of frames (~4 per second).
2. ``pick_track`` chooses one face per sample, preferring large, confident faces
   and sticking to the previously tracked face to avoid ping-ponging.
3. ``smooth_track`` applies a median filter, a moving average and a dead-zone so
   the virtual camera pans gently instead of jittering.
4. ``crop_path`` converts the smoothed track into per-frame crop offsets, which
   the renderer feeds to ffmpeg's ``sendcmd`` + ``crop`` filters.

Detection is the only part that needs MediaPipe or OpenCV, and both are imported
lazily so the rest of the module stays cheap to test.
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

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
FACE_MODEL_NAME = "blaze_face_short_range.tflite"
_model_lock = threading.Lock()


@dataclass(frozen=True)
class FaceSample:
    t: float          # seconds, relative to the clip start
    cx: float         # normalised centre x (0..1)
    cy: float         # normalised centre y (0..1)
    size: float       # normalised face height (0..1)
    score: float      # detector confidence (0..1)


def ensure_face_model(models_dir: Path) -> Path:
    """Download the MediaPipe face detector model once, atomically."""
    with _model_lock:  # parallel clip renders must not download the model concurrently
        return _ensure_face_model(models_dir)


def _ensure_face_model(models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / FACE_MODEL_NAME
    if target.exists() and target.stat().st_size > 100_000:
        return target
    log.info("Downloading face detector model to %s", target)
    fd, tmp = tempfile.mkstemp(dir=models_dir, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh, urllib.request.urlopen(FACE_MODEL_URL, timeout=60) as resp:
            fh.write(resp.read())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return target


def sample_faces(video: Path, model_path: Path, samples_per_second: float = 4.0,
                 max_width: int = 640) -> list[FaceSample]:
    """Run the face detector over sparse frames of ``video``."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )
    samples: list[FaceSample] = []
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(fps / samples_per_second))

    with vision.FaceDetector.create_from_options(options) as detector:
        index = 0
        while True:
            ok = cap.grab()  # grab() decodes without the colour conversion of read()
            if not ok:
                break
            if index % stride == 0:
                ok, frame = cap.retrieve()
                if ok:
                    h, w = frame.shape[:2]
                    scale = min(1.0, max_width / w)
                    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = detector.detect(image)
                    sh, sw = rgb.shape[:2]
                    for det in result.detections:
                        box = det.bounding_box
                        score = det.categories[0].score if det.categories else 0.0
                        samples.append(FaceSample(
                            t=index / fps,
                            cx=(box.origin_x + box.width / 2) / sw,
                            cy=(box.origin_y + box.height / 2) / sh,
                            size=box.height / sh,
                            score=float(score),
                        ))
            index += 1
    cap.release()
    return samples


def pick_track(samples: list[FaceSample], duration: float, step: float = 0.25,
               stickiness: float = 0.2) -> list[tuple[float, float]]:
    """Choose one speaker per time step. Returns ``[(t, cx)]`` on a regular grid.

    Scoring favours large, confident faces. A face near the previously tracked
    position gets a bonus, so the camera does not swap between two people on
    every detection.
    """
    by_time: dict[int, list[FaceSample]] = {}
    for s in samples:
        by_time.setdefault(round(s.t / step), []).append(s)

    n_steps = max(1, int(duration / step) + 1)
    track: list[tuple[float, float]] = []
    last_cx = 0.5
    for i in range(n_steps):
        candidates = by_time.get(i, [])
        if candidates:
            def rank(s: FaceSample, anchor: float = last_cx) -> float:
                continuity = 1.0 - min(1.0, abs(s.cx - anchor) / 0.5)
                return s.score * (s.size ** 2) * (1.0 + stickiness * continuity)
            best = max(candidates, key=rank)
            last_cx = best.cx
        track.append((i * step, last_cx))
    return track


def smooth_track(track: list[tuple[float, float]], median_window: int = 5,
                 mean_window_s: float = 1.2, step: float = 0.25,
                 dead_zone: float = 0.02) -> list[tuple[float, float]]:
    """Median filter (kills detector outliers), moving average (smooth pans), dead-zone (no jitter)."""
    if not track:
        return track
    xs = [x for _, x in track]

    def median_at(i: int) -> float:
        half = median_window // 2
        return statistics.median(xs[max(0, i - half): i + half + 1])

    med = [median_at(i) for i in range(len(xs))]
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
    """Convert a smoothed track into crop geometry.

    Returns ``(crop_w, [(t, x_left)])`` where crop_w is the source-pixel crop width
    and ``x_left`` is the crop origin for each output frame time. When the source is
    narrower than the target aspect, the caller should letterbox instead, so
    crop_w == src_w and every x_left is 0.
    """
    crop_w = int(round(src_h * out_aspect))
    crop_w -= crop_w % 2
    if crop_w >= src_w:
        return src_w - (src_w % 2), [(0.0, 0)]

    if not track:
        track = [(0.0, 0.5)]
    frames = max(1, int(duration * fps))
    times = [t for t, _ in track]
    values = [v for _, v in track]
    points: list[tuple[float, int]] = []
    for f in range(frames):
        t = f / fps
        cx = _interp(t, times, values)
        x_left = int(round(cx * src_w - crop_w / 2))
        x_left = max(0, min(src_w - crop_w, x_left))
        x_left -= x_left % 2
        points.append((t, x_left))
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


def sendcmd_script(points: list[tuple[float, int]]) -> str:
    """ffmpeg sendcmd script that moves the crop origin per frame, e.g. ``0.033 crop x 120;``."""
    lines = [f"{t:.4f} crop x {x};" for t, x in points]
    return "\n".join(lines) + "\n"


def analyse_speaker_path(video: Path, duration: float, out_aspect: float, fps: float,
                         models_dir: Path) -> tuple[int, int, int, list[tuple[float, int]]] | None:
    """End-to-end helper. Returns ``(src_w, crop_w, src_h, points)`` or None when no tracking is needed."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if src_w / src_h <= out_aspect:
        return None  # already narrower than the target: letterbox, nothing to track

    model = ensure_face_model(models_dir)
    samples = sample_faces(video, model)
    track = smooth_track(pick_track(samples, duration))
    crop_w, points = crop_path(track, src_w, src_h, out_aspect, fps, duration)
    return src_w, crop_w, src_h, points
