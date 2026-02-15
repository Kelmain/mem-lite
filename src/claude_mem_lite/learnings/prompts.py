"""Extraction prompt for learning extraction from session summaries."""

from __future__ import annotations

LEARNING_EXTRACTION_SYSTEM = """\
You extract reusable project knowledge from development session summaries.

Categories (pick exactly one per learning):
- architecture: System design decisions, module structure, API patterns
- convention: Coding style rules, naming conventions, testing patterns
- gotcha: Non-obvious bugs, silent failures, surprising behaviors
- dependency: Package versions, compatibility notes, API quirks
- pattern: Recurring implementation patterns, common workflows

Rules:
1. Only extract things useful across MULTIPLE sessions
2. Do NOT extract session-specific actions ("fixed bug in auth.py")
3. DO extract patterns ("auth module uses JWT with HttpOnly cookies")
4. Each learning should be 1-2 sentences, specific and actionable
5. If a new learning CONTRADICTS an existing one listed below, set "contradicts" \
to the existing learning's content
6. confidence: 0.3-0.7 based on how confident you are this is a real pattern \
(not a one-off)

Respond with a JSON array ONLY (no markdown fences, no preamble):
[{"category": "...", "content": "...", "confidence": 0.5, "contradicts": null}]

Return empty array [] if no learnings are worth extracting.
"""

LEARNING_EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "learnings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "architecture",
                            "convention",
                            "gotcha",
                            "dependency",
                            "pattern",
                        ],
                    },
                    "content": {"type": "string"},
                    "confidence": {"type": "number"},
                    "contradicts": {"type": ["string", "null"]},
                },
                "required": ["category", "content", "confidence", "contradicts"],
            },
        }
    },
    "required": ["learnings"],
}


def build_extraction_prompt(
    summary: str,
    observations: list[dict],
    existing_learnings: list[dict],
) -> str:
    """Build user message for learning extraction."""
    parts = [f"## Session Summary\n{summary}\n"]

    if observations:
        parts.append("## Key Observations")
        for obs in observations[:8]:
            title = obs.get("title", "untitled")
            obs_summary = obs.get("summary", "")
            parts.append(f"- {title}: {obs_summary}")
        parts.append("")

    if existing_learnings:
        parts.append("## Existing Project Learnings (do NOT re-extract these)")
        parts.append("Learnings marked [low-confidence] have been contradicted or are unconfirmed.")
        parts.append(
            "Do NOT re-extract [low-confidence] learnings unless you have strong new evidence.\n"
        )
        for learning in existing_learnings[:30]:
            cat = learning.get("category", "unknown")
            content_text = learning.get("content", "")
            conf = learning.get("confidence", 0.5)
            prefix = "[low-confidence] " if conf < 0.5 else ""
            parts.append(f"- {prefix}[{cat}] {content_text}")
        parts.append("")

    return "\n".join(parts)
