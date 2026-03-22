"""
RunPod Serverless Provider

Handles async job submission and polling for Titan fine-tuned models.
Supports the vLLM worker format with sampling_params.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from titan.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RunPodResponse:
    """Response from a RunPod inference call."""
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    execution_time_ms: int = 0
    job_id: str = ""


class RunPodProvider:
    """
    Async-poll client for RunPod Serverless vLLM endpoints.

    POST {endpoint}/run   → submit job
    GET  {endpoint}/status/{id} → poll until COMPLETED
    """

    POLL_INTERVAL = 2.0     # seconds between polls
    MAX_POLL_TIME = 600.0   # 10 min max (cold start + generation)

    def __init__(self):
        self.api_key = settings.runpod_api_key
        self.backend_endpoint = settings.titan_backend_endpoint.rstrip("/")
        self.ui_endpoint = settings.titan_ui_endpoint.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(60.0, connect=15.0),
            )
        return self._client

    def _resolve_endpoint(self, model: str) -> str:
        if "ui" in model.lower() or "frontend" in model.lower():
            return self.ui_endpoint
        return self.backend_endpoint

    async def chat(
        self,
        messages: list[dict],
        model: str = "titan-backend",
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> RunPodResponse:
        """
        Send a chat completion to a Titan model via RunPod Serverless.

        Uses the vLLM worker format with sampling_params (critical fix from diagnostic).
        """
        endpoint = self._resolve_endpoint(model)
        temp = temperature or settings.titan_temperature
        tokens = max_tokens or settings.titan_max_tokens

        # Build messages list with optional system prompt
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        # vLLM worker format — sampling_params is REQUIRED (top-level max_tokens is ignored)
        payload = {
            "input": {
                "messages": all_messages,
                "sampling_params": {
                    "temperature": temp,
                    "max_tokens": tokens,
                },
            }
        }

        start_time = time.time()
        job = await self._submit_and_poll(endpoint, payload)
        elapsed_ms = int((time.time() - start_time) * 1000)

        content, input_tok, output_tok = self._extract_content(job)

        return RunPodResponse(
            content=content,
            input_tokens=input_tok,
            output_tokens=output_tok,
            execution_time_ms=elapsed_ms,
            job_id=job.get("id", ""),
        )

    async def _submit_and_poll(self, endpoint: str, payload: dict) -> dict:
        client = await self._get_client()

        # Submit
        run_url = f"{endpoint}/run"
        resp = await client.post(run_url, json=payload)
        if resp.status_code != 200:
            raise Exception(f"RunPod submit error ({resp.status_code}): {resp.text}")

        job = resp.json()
        job_id = job.get("id")
        status = job.get("status", "UNKNOWN")

        if not job_id:
            raise Exception(f"RunPod returned no job ID: {job}")
        if status == "COMPLETED":
            return job

        # Poll
        status_url = f"{endpoint}/status/{job_id}"
        elapsed = 0.0
        while elapsed < self.MAX_POLL_TIME:
            await asyncio.sleep(self.POLL_INTERVAL)
            elapsed += self.POLL_INTERVAL

            resp = await client.get(status_url)
            if resp.status_code != 200:
                logger.warning(f"Poll error ({resp.status_code}): {resp.text}")
                continue

            job = resp.json()
            status = job.get("status", "UNKNOWN")

            if status == "COMPLETED":
                return job
            if status == "FAILED":
                raise Exception(f"RunPod job failed: {job.get('error', 'Unknown')}")

            if elapsed % 30 < self.POLL_INTERVAL:
                logger.info(f"RunPod {job_id}: {status} ({elapsed:.0f}s)")

        raise Exception(f"RunPod job {job_id} timed out after {self.MAX_POLL_TIME}s")

    def _extract_content(self, job: dict) -> tuple[str, int, int]:
        output = job.get("output")
        if not output:
            raise Exception(f"RunPod response has no output: {job}")

        if isinstance(output, list):
            output = output[0]

        # vLLM format: output.choices[0].tokens or output.choices[0].message.content
        if isinstance(output, dict):
            choices = output.get("choices", [])
            if choices:
                choice = choices[0]
                # Try message.content first
                msg = choice.get("message", {})
                if msg.get("content"):
                    content = msg["content"]
                else:
                    # Fall back to tokens list
                    tokens = choice.get("tokens", [])
                    content = "".join(tokens) if isinstance(tokens, list) else str(tokens)
            elif "text" in output:
                content = output["text"]
            else:
                content = str(output)

            usage = output.get("usage", {})
            return (
                content,
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            )

        # Plain string output
        return str(output), 0, 0

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
