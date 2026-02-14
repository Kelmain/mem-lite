"""Tests for AI compression (compressor + prompts)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from anthropic import APIConnectionError, APIStatusError

from claude_mem_lite.config import Config
from claude_mem_lite.storage.models import CompressedObservation
from claude_mem_lite.worker.compressor import Compressor
from claude_mem_lite.worker.exceptions import NonRetryableError, RetryableError
from claude_mem_lite.worker.prompts import MAX_RAW_CHARS, build_compression_prompt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(text: str, input_tokens: int = 100, output_tokens: int = 50):
    """Create a mock Anthropic API response."""
    content_block = MagicMock()
    content_block.text = text
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    response = MagicMock()
    response.content = [content_block]
    response.usage = usage
    return response


# ---------------------------------------------------------------------------
# Compressor tests
# ---------------------------------------------------------------------------


class TestCompressor:
    """Tests for the Compressor class."""

    @patch("claude_mem_lite.worker.compressor.AsyncAnthropic")
    async def test_compress_valid_structured_output(self, mock_anthropic_cls):
        """Valid JSON from the API produces a CompressedObservation with correct fields."""
        payload = {
            "title": "Add user auth middleware",
            "summary": "Added JWT-based auth middleware to protect API routes.",
            "detail": None,
            "files_touched": ["src/auth.py", "src/routes.py"],
            "functions_changed": [],
        }
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = _make_mock_response(
            json.dumps(payload), input_tokens=200, output_tokens=80
        )
        mock_anthropic_cls.return_value = mock_client

        compressor = Compressor(Config())
        result = await compressor.compress("raw output", "Write", "src/auth.py")

        assert isinstance(result, CompressedObservation)
        assert result.title == "Add user auth middleware"
        assert result.summary == "Added JWT-based auth middleware to protect API routes."
        assert result.detail is None
        assert result.files_touched == ["src/auth.py", "src/routes.py"]
        assert result.functions_changed == []
        assert result.tokens_in == 200
        assert result.tokens_out == 80

    @patch("claude_mem_lite.worker.compressor.AsyncAnthropic")
    async def test_compress_with_functions_changed(self, mock_anthropic_cls):
        """functions_changed array is parsed into FunctionChangeRecord objects."""
        payload = {
            "title": "Refactor login handler",
            "summary": "Refactored login to use dependency injection.",
            "detail": "Replaced global db reference with injected session.",
            "files_touched": ["src/auth.py"],
            "functions_changed": [
                {"file": "src/auth.py", "name": "login", "action": "modified"},
                {"file": "src/auth.py", "name": "get_session", "action": "new"},
            ],
        }
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = _make_mock_response(json.dumps(payload))
        mock_anthropic_cls.return_value = mock_client

        compressor = Compressor(Config())
        result = await compressor.compress("raw output", "Write", "src/auth.py")

        assert len(result.functions_changed) == 2
        assert result.functions_changed[0].file == "src/auth.py"
        assert result.functions_changed[0].name == "login"
        assert result.functions_changed[0].action == "modified"
        assert result.functions_changed[1].name == "get_session"
        assert result.functions_changed[1].action == "new"

    @patch("claude_mem_lite.worker.compressor.AsyncAnthropic")
    async def test_compress_json_with_markdown_fences(self, mock_anthropic_cls):
        """Fallback parser strips markdown fences and extracts JSON."""
        payload = {
            "title": "Update config",
            "summary": "Changed default port.",
            "detail": None,
            "files_touched": ["config.py"],
            "functions_changed": [],
        }
        # First json.loads will fail on the fenced text, triggering fallback
        fenced = f"```json\n{json.dumps(payload)}\n```"

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = _make_mock_response(fenced)
        mock_anthropic_cls.return_value = mock_client

        compressor = Compressor(Config())
        result = await compressor.compress("raw output", "Write", "config.py")

        assert result.title == "Update config"
        assert result.files_touched == ["config.py"]

    @patch("claude_mem_lite.worker.compressor.AsyncAnthropic")
    async def test_compress_json_with_trailing_text(self, mock_anthropic_cls):
        """Fallback parser extracts JSON even when followed by trailing text."""
        payload = {
            "title": "Fix bug in parser",
            "summary": "Resolved off-by-one error.",
            "detail": None,
            "files_touched": ["parser.py"],
            "functions_changed": [],
        }
        text_with_trailing = json.dumps(payload) + " Hope this helps!"

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = _make_mock_response(text_with_trailing)
        mock_anthropic_cls.return_value = mock_client

        compressor = Compressor(Config())
        result = await compressor.compress("raw output", "Write", "parser.py")

        assert result.title == "Fix bug in parser"
        assert result.files_touched == ["parser.py"]

    @patch("claude_mem_lite.worker.compressor.AsyncAnthropic")
    async def test_compress_api_connection_error_raises_retryable(self, mock_anthropic_cls):
        """APIConnectionError is wrapped in RetryableError."""
        mock_client = AsyncMock()
        mock_client.messages.create.side_effect = APIConnectionError(
            request=httpx.Request("POST", "https://api.anthropic.com")
        )
        mock_anthropic_cls.return_value = mock_client

        compressor = Compressor(Config())
        with pytest.raises(RetryableError):
            await compressor.compress("raw output", "Write", "file.py")

    @patch("claude_mem_lite.worker.compressor.AsyncAnthropic")
    async def test_compress_api_status_error_500_raises_retryable(self, mock_anthropic_cls):
        """APIStatusError with status 500 is wrapped in RetryableError."""
        mock_client = AsyncMock()
        mock_response = httpx.Response(
            status_code=500,
            request=httpx.Request("POST", "https://api.anthropic.com"),
        )
        mock_client.messages.create.side_effect = APIStatusError(
            message="Server Error", response=mock_response, body=None
        )
        mock_anthropic_cls.return_value = mock_client

        compressor = Compressor(Config())
        with pytest.raises(RetryableError):
            await compressor.compress("raw output", "Write", "file.py")

    @patch("claude_mem_lite.worker.compressor.AsyncAnthropic")
    async def test_compress_api_status_error_400_raises_non_retryable(self, mock_anthropic_cls):
        """APIStatusError with status 400 is wrapped in NonRetryableError."""
        mock_client = AsyncMock()
        mock_response = httpx.Response(
            status_code=400,
            request=httpx.Request("POST", "https://api.anthropic.com"),
        )
        mock_client.messages.create.side_effect = APIStatusError(
            message="Bad Request", response=mock_response, body=None
        )
        mock_anthropic_cls.return_value = mock_client

        compressor = Compressor(Config())
        with pytest.raises(NonRetryableError):
            await compressor.compress("raw output", "Write", "file.py")


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------


class TestBuildCompressionPrompt:
    """Tests for the build_compression_prompt function."""

    def test_compress_truncation_large_input(self):
        """Input exceeding MAX_RAW_CHARS is truncated with a marker preserving head and tail."""
        large_input = "x" * (MAX_RAW_CHARS + 10_000)

        result = build_compression_prompt(large_input, "Write", "big_file.py")

        # Truncation marker must be present
        assert "[... truncated" in result
        assert "chars ...]" in result

        # Head and tail of original content must be preserved
        half = MAX_RAW_CHARS // 2
        assert "x" * min(100, half) in result  # head chars present
        # Full original should NOT be present (it was truncated)
        assert large_input not in result
