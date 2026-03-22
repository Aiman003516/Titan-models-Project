"""
Titan Agent Configuration

All settings loaded from environment variables / .env file.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── RunPod (Titan fine-tuned models) ──────────────────────────────────
    runpod_api_key: str = Field(
        ..., description="RunPod API key (rpa_...)"
    )
    titan_backend_endpoint: str = Field(
        ..., description="RunPod endpoint URL for Titan-Backend"
    )
    titan_ui_endpoint: str = Field(
        ..., description="RunPod endpoint URL for Titan-UI"
    )
    titan_temperature: float = Field(default=0.6)
    titan_max_tokens: int = Field(default=16384)

    # ── Groq (Tier-2 debugging LLM) ──────────────────────────────────────
    groq_api_key: str = Field(
        ..., description="Groq API key (gsk_...)"
    )
    groq_model: str = Field(
        default="llama-3.1-70b-versatile",
        description="Groq model for Tier-2 debugging",
    )
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
    )

    # ── Debug loop ────────────────────────────────────────────────────────
    debug_temperature: float = Field(default=0.3)
    debug_max_tokens: int = Field(default=8192)
    max_debug_retries: int = Field(default=3)

    # ── Server ────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
