"""AI dubbing: synthesise translated speech (Gemini TTS) and fit it to the clip length.

The original voice is kept underneath at low volume (see ``render``), so the result
sounds like a dub rather than a replacement. Dubbing needs the Gemini provider.
"""

from __future__ import annotations

import wave
from pathlib import Path

from .config import Settings
from .llm import LLMError, _with_retries
from .media import run_ffmpeg

TTS_RATE = 24000
MAX_SPEEDUP = 1.4


def _write_wav(pcm: bytes, path: Path) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TTS_RATE)
        w.writeframes(pcm)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def atempo_chain(ratio: float) -> str:
    """ffmpeg atempo filter for a speed ratio (each atempo stage accepts 0.5-2.0)."""
    stages = []
    while ratio > 2.0:
        stages.append("atempo=2.0")
        ratio /= 2.0
    while ratio < 0.5:
        stages.append("atempo=0.5")
        ratio /= 0.5
    stages.append(f"atempo={ratio:.4f}")
    return ",".join(stages)


def synthesize(settings: Settings, text: str, language: str, out: Path, target_seconds: float,
               warnings: list[str]) -> Path:
    """Create ``out`` (wav) with ``text`` spoken in ``language``, sped up if needed to fit ``target_seconds``."""
    if settings.llm_provider != "gemini" or not settings.gemini_api_key:
        raise LLMError("dubbing requires LLM_PROVIDER=gemini and GEMINI_API_KEY (uses Gemini TTS)")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value(),
                          http_options=types.HttpOptions(api_version="v1beta", timeout=120_000))
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=settings.tts_voice))),
    )
    prompt = f"Read this naturally and expressively in {language}, at a brisk, energetic pace: {text}"
    response = _with_retries(lambda: client.models.generate_content(
        model=settings.tts_model, contents=prompt, config=config))
    try:
        pcm = response.candidates[0].content.parts[0].inline_data.data
    except (AttributeError, IndexError, TypeError) as exc:
        raise LLMError("TTS returned no audio") from exc

    raw = out.with_suffix(".raw.wav")
    _write_wav(pcm, raw)
    ratio = wav_duration(raw) / target_seconds if target_seconds > 0 else 1.0
    if ratio > 1.0:
        if ratio > MAX_SPEEDUP:
            warnings.append(f"dub is {ratio:.2f}x longer than the clip; sped up {MAX_SPEEDUP}x and trimmed")
        run_ffmpeg([settings.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
                    "-af", atempo_chain(min(ratio, MAX_SPEEDUP)), "-t", f"{target_seconds:.3f}", str(out)])
        raw.unlink(missing_ok=True)
    else:
        raw.replace(out)
    return out
