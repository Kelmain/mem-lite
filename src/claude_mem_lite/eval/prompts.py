"""QAG prompt template for information preservation evaluation."""

from __future__ import annotations

QAG_PROMPT = """\
You are evaluating whether a compressed observation preserves key information \
from the original tool output.

<raw_tool_output>
{raw_output}
</raw_tool_output>

<compressed_observation>
Title: {title}
Summary: {summary}
Detail: {detail}
Files: {files_touched}
Functions: {functions_changed}
</compressed_observation>

Tasks:
1. Generate UP TO 5 factual questions that a developer continuing this work
   tomorrow would need answered.

   FOCUS ON:
   - Architectural decisions and their rationale
   - Behavioral changes to public APIs or interfaces
   - New dependencies or breaking changes
   - Key trade-offs or alternatives that were considered
   - Error handling or edge case changes

   DO NOT ask about:
   - Specific line numbers or column positions
   - Formatting, whitespace, or import ordering changes
   - Variable or parameter renaming (unless it signals a semantic change)
   - Internal helper functions or private implementation details
   - Trivial boilerplate (license headers, auto-generated code)

   If the change is small (e.g., a single bug fix), generate FEWER questions
   rather than padding with low-value ones. Minimum 2, maximum 5.

2. For each question, determine if the compressed observation contains enough
   information to answer it correctly.

3. Also assess: does the summary explain the RATIONALE (why something was done),
   not just the action (what was done)?

Return ONLY a JSON object (no markdown fences):
{{
  "questions": [
    {{"question": "...", "answerable": true/false, "evidence": "brief quote or null"}}
  ],
  "decision_rationale_present": true/false,
  "rationale_note": "brief explanation of why rationale is/isn't present"
}}
"""
