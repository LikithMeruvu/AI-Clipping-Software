"""Speech-to-text with word timestamps (faster-whisper), cached on disk.

The model is loaded once per process. CUDA is used when ctranslate2 can see a GPU
and the model loads; otherwise the loader falls back to int8 on CPU with a warning.
Transcripts are cached by source key, so re-running a clip with new settings does
not transcribe the video again.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from .captions import Word

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    language: str
    duration: float
    segments: list[Segment]
    words: list[Word]

    def to_json(self) -> str:
        return json.dumps({
            "language": self.language,
            "duration": self.duration,
            "segments": [asdict(s) for s in self.segments],
            "words": [asdict(w) for w in self.words],
        })

    @classmethod
    def from_json(cls, raw: str) -> Transcript:
        data = json.loads(raw)
        return cls(
            language=data["language"],
            duration=data["duration"],
            segments=[Segment(**s) for s in data["segments"]],
            words=[Word(**w) for w in data["words"]],
        )


class Transcriber:
    _lock = threading.Lock()
    _models: dict[tuple[str, str], object] = {}

    def __init__(self, model_name: str, device: str = "auto", models_dir: Path | None = None):
        self.model_name = model_name
        self.device_pref = device
        self.models_dir = models_dir

    def _load(self):
        import ctranslate2
        from faster_whisper import WhisperModel

        want_cuda = self.device_pref in ("auto", "cuda") and ctranslate2.get_cuda_device_count() > 0
        key = (self.model_name, "cuda" if want_cuda else "cpu")
        with self._lock:
            if key in self._models:
                return self._models[key]
            download_root = str(self.models_dir) if self.models_dir else None
            if want_cuda:
                try:
                    model = WhisperModel(self.model_name, device="cuda", compute_type="float16",
                                         download_root=download_root)
                    log.info("whisper %s loaded on CUDA (float16)", self.model_name)
                    self._models[key] = model
                    return model
                except Exception as exc:  # e.g. missing cuBLAS/cuDNN DLLs
                    log.warning("CUDA whisper load failed (%s); using CPU int8", exc)
            key = (self.model_name, "cpu")
            if key not in self._models:
                log.info("whisper %s loading on CPU (int8)", self.model_name)
                self._models[key] = WhisperModel(self.model_name, device="cpu", compute_type="int8",
                                                 cpu_threads=max(1, os.cpu_count() or 4),
                                                 download_root=download_root)
            return self._models[key]

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcript:
        try:
            return self._transcribe(audio_path, language)
        except RuntimeError as exc:
            # CUDA can load yet fail at decode time (e.g. cublas64_12.dll missing). Retry on CPU once.
            if self.device_pref != "cpu" and any(k in str(exc).lower() for k in ("cublas", "cudnn", "cuda")):
                log.warning("CUDA unusable at decode time (%s); retrying on CPU", exc)
                self.device_pref = "cpu"
                return self._transcribe(audio_path, language)
            raise

    def _transcribe(self, audio_path: Path, language: str | None = None) -> Transcript:
        model = self._load()
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            beam_size=5,
            condition_on_previous_text=True,
        )
        segments: list[Segment] = []
        words: list[Word] = []
        for seg in segments_iter:  # generator: decoding happens here
            text = seg.text.strip()
            if text:
                segments.append(Segment(start=float(seg.start), end=float(seg.end), text=text))
            for w in seg.words or []:
                token = w.word.strip()
                if token:
                    words.append(Word(start=float(w.start), end=float(w.end), text=token))
        duration = float(getattr(info, "duration", 0.0) or (segments[-1].end if segments else 0.0))
        return Transcript(language=info.language, duration=duration, segments=segments, words=words)


def cached_transcribe(transcriber: Transcriber, audio_path: Path, cache_key: str, cache_dir: Path,
                      language: str | None = None) -> Transcript:
    """Return a cached transcript when present, otherwise transcribe and store it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in cache_key)
    path = cache_dir / f"{safe}.{transcriber.model_name}.{language or 'auto'}.json"
    if path.exists():
        try:
            return Transcript.from_json(path.read_text(encoding="utf-8"))
        except (ValueError, KeyError, TypeError):
            log.warning("ignoring corrupt transcript cache %s", path.name)
    transcript = transcriber.transcribe(audio_path, language=language)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(transcript.to_json(), encoding="utf-8")
    tmp.replace(path)
    return transcript
