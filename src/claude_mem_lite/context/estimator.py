"""Token estimation for context budget management."""


def estimate_tokens(text: str) -> int:
    """Estimate Claude token count from text.

    Uses character-based heuristic: ~3.5 characters per token.
    Accuracy: ±10-15% vs Claude's actual tokenizer.
    Sufficient for context budget management where slight
    overestimation is preferable to exceeding the budget.
    """
    return max(1, int(len(text) / 3.5))
