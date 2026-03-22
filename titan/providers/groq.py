"""
Groq Provider — Tier-2 debugging LLM

Uses Groq's OpenAI-compatible API for fast, free code fixing.
Llama 3.1 70B at ~2-5 second response times.
"""

import logging
from typing import Optional

import httpx

from titan.config import settings

logger = logging.getLogger(__name__)


class GroqProvider:
    """Fast, free LLM via Groq for Tier-2 code debugging."""

    def __init__(self):
        self.api_key = settings.groq_api_key
        self.base_url = settings.groq_base_url.rstrip("/")
        self.model = settings.groq_model
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    async def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a chat completion to Groq. Returns the response text.
        """
        client = await self._get_client()

        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": all_messages,
            "temperature": temperature or settings.debug_temperature,
            "max_tokens": max_tokens or settings.debug_max_tokens,
        }

        resp = await client.post("/chat/completions", json=payload)
        if resp.status_code != 200:
            error_text = resp.text[:500]
            raise Exception(f"Groq error ({resp.status_code}): {error_text}")

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        logger.info(
            f"Groq response: {usage.get('prompt_tokens', 0)} in / "
            f"{usage.get('completion_tokens', 0)} out tokens"
        )
        return content

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
