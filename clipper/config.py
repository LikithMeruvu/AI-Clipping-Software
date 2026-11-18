"""Typed application settings, loaded from environment variables and ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (clip selection, hooks, titles, translation) ---
    llm_provider: Literal["gemini", "openai"] = "gemini"
    llm_model: str = "gemini-2.5-flash"
    llm_base_url: str | None = None  # openai provider only, e.g. https://api.groq.com/openai/v1
    llm_api_key: SecretStr | None = None  # openai provider; optional for local servers
    gemini_api_key: SecretStr | None = None  # GEMINI_API_KEY

    pexels_api_key: SecretStr | None = None  # PEXELS_API_KEY, enables --broll

    # --- transcription ---
    whisper_model: str = "large-v3-turbo"  # use "small" or "base" on CPU
    whisper_device: Literal["auto", "cuda", "cpu"] = "auto"

    tts_model: str = "gemini-2.5-flash-preview-tts"  # used by AI dubbing
    tts_voice: str = "Kore"

    # --- rendering ---
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    encoder: Literal["auto", "h264_nvenc", "libx264"] = "auto"
    render_workers: int = Field(default=2, ge=1, le=8)
    fonts_dir: Path | None = None  # defaults to the OS font directory when unset

    # --- storage ---
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")

    # --- web service ---
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    api_token: SecretStr | None = None  # when set, /api/* requires "Authorization: Bearer <token>"
    max_upload_mb: int = Field(default=2048, ge=1)
    max_queued_jobs: int = Field(default=20, ge=1)

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
