"""AI compression using Claude Haiku via Anthropic SDK with structured outputs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from anthropic import APIConnectionError, APIStatusError, AsyncAnthropic, RateLimitError

from claude_mem_lite.storage.models import CompressedObservation, FunctionChangeRecord
from claude_mem_lite.worker.exceptions import NonRetryableError, RetryableError
from claude_mem_lite.worker.prompts import COMPRESSION_SCHEMA, build_compression_prompt

if TYPE_CHECKING:
    from claude_mem_lite.config import Config


class Compressor:
    """Compress raw tool outputs into structured observations via Claude API."""

    def __init__(self, config: Config) -> None:
        self.client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env
        self.model = config.compression_model

    async def compress(
        self,
        raw_output: str,
        tool_name: str,
        files_touched: str,
    ) -> CompressedObservation:
        """Compress raw tool output into structured observation.

        Uses Anthropic Structured Outputs API for guaranteed
        schema-valid JSON.
        """
        prompt = build_compression_prompt(raw_output, tool_name, files_touched)

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                extra_body={
                    "output_config": {
                        "format": {
                            "type": "json_schema",
                            "schema": COMPRESSION_SCHEMA,
                        }
                    }
                },
            )
        except (APIConnectionError, RateLimitError) as e:
            msg = str(e)
            raise RetryableError(msg) from e
        except APIStatusError as e:
            msg = str(e)
            if e.status_code >= 500:
                raise RetryableError(msg) from e
            raise NonRetryableError(msg) from e

        text_block = response.content[0]
        assert hasattr(text_block, "text"), f"Unexpected content block type: {type(text_block)}"
        return self._parse_response(text_block.text, response.usage)

    def _parse_response(self, text: str, usage) -> CompressedObservation:
        """Parse JSON response from Claude.

        With structured outputs, the response is guaranteed to be valid JSON.
        Fallback parsing handles rare edge cases as defense-in-depth.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = self._extract_json_fallback(text)

        functions_changed = [FunctionChangeRecord(**fc) for fc in data.get("functions_changed", [])]

        return CompressedObservation(
            title=data.get("title", "")[:200],
            summary=data.get("summary", "")[:1000],
            detail=data.get("detail"),
            files_touched=data.get("files_touched", []),
            functions_changed=functions_changed,
            tokens_in=usage.input_tokens,
            tokens_out=usage.output_tokens,
        )

    def _extract_json_fallback(self, text: str) -> dict:
        """Last-resort JSON extraction when structured outputs fails."""
        cleaned = text.strip()

        # Strip markdown fences
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        # Find JSON object boundaries
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                result: dict = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
            else:
                return result

        msg = f"Could not extract JSON from response: {text[:200]}"
        raise RetryableError(msg)

    async def close(self) -> None:
        """Clean up HTTP client."""
        await self.client.close()
