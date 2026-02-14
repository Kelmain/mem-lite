"""Compression prompt templates and schemas."""

from __future__ import annotations

# JSON Schema for structured outputs API -- mirrors CompressedObservation model
COMPRESSION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "Brief action summary, 5-10 words, imperative mood",
        },
        "summary": {
            "type": "string",
            "description": "What happened and why, 1-3 sentences",
        },
        "detail": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Technical details worth remembering. Null if summary is sufficient.",
        },
        "files_touched": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of file paths actually modified",
        },
        "functions_changed": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "name": {"type": "string"},
                    "action": {"type": "string", "enum": ["new", "modified", "deleted"]},
                },
                "required": ["file", "name", "action"],
                "additionalProperties": False,
            },
            "description": "Functions that were created, modified, or deleted",
        },
    },
    "required": ["title", "summary", "files_touched", "functions_changed"],
    "additionalProperties": False,
}

SUMMARY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_files": {"type": "array", "items": {"type": "string"}},
        "key_decisions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "key_files", "key_decisions"],
    "additionalProperties": False,
}

COMPRESSION_PROMPT_V1 = """\
Compress this Claude Code tool output into a structured observation.

Tool: {tool_name}
Files: {files_touched}

<tool_output>
{raw_output}
</tool_output>

Return a JSON object with these fields:
{{
  "title": "Brief action summary, 5-10 words. Example: 'Added JWT auth middleware'",
  "summary": "What happened and why, 1-3 sentences. Include key decisions made.",
  "detail": "Technical details worth remembering across sessions. Null if summary is sufficient.",
  "files_touched": ["list", "of", "file/paths"],
  "functions_changed": [
    {{"file": "path/to/file.py", "name": "function_name", "action": "new|modified|deleted"}}
  ]
}}

Rules:
- title: imperative mood, no period. Focus on WHAT changed, not tool mechanics.
- summary: preserve WHY decisions were made, not just WHAT happened.
- detail: only include if there's non-obvious information. Omit for trivial operations.
- files_touched: only files actually modified, not just read.
- functions_changed: only if identifiable from the output. Empty array if unclear.
- If the tool output is a Read operation with no changes, set title to describe what was \
examined and summary to key findings.
"""

MAX_RAW_CHARS = 32_000


def build_compression_prompt(
    raw_output: str,
    tool_name: str,
    files_touched: str,
) -> str:
    """Build compression prompt, truncating raw_output if too large."""
    truncated = raw_output
    if len(raw_output) > MAX_RAW_CHARS:
        half = MAX_RAW_CHARS // 2
        truncated = (
            raw_output[:half]
            + f"\n\n[... truncated {len(raw_output) - MAX_RAW_CHARS} chars ...]\n\n"
            + raw_output[-half:]
        )

    return COMPRESSION_PROMPT_V1.format(
        tool_name=tool_name,
        files_touched=files_touched,
        raw_output=truncated,
    )
