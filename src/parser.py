"""Output parsing for IAT and decision tasks."""

from typing import Dict, Optional
import re


def parse_iat_assignments(response_text: str) -> Dict[str, str]:
    """
    Parse IAT response lines like:
        good - Other-interest
        admirable - Self-interest

    Returns:
        Dict mapping word (lowercase) -> "self" or "other"
    """
    assignments: Dict[str, str] = {}

    for raw_line in response_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Remove leading numbers like "1. " or "1) "
        line = re.sub(r"^\d+[\.\)]\s*", "", line)

        # Try to split on various separators
        sep = None
        for candidate in [" - ", " – ", " — ", ": ", " = ", "-", "–", "—"]:
            if candidate in line:
                sep = candidate
                break

        if sep is None:
            continue

        parts = line.split(sep, 1)
        if len(parts) != 2:
            continue

        word = parts[0].strip().lower()
        cat_text = parts[1].strip().lower()

        # Normalize category
        if "self" in cat_text:
            cat = "self"
        elif "other" in cat_text:
            cat = "other"
        else:
            continue

        assignments[word] = cat

    return assignments


def parse_decision_choice(response_text: str) -> Optional[str]:
    """
    Parse a decision task response to extract A or B choice.

    Returns:
        "A", "B", or None if unparsable
    """
    for raw_line in response_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Check for "A" or "B" at start
        first_char = line[0].upper()
        if first_char in ("A", "B"):
            # Verify it's not part of a longer word starting with A/B
            if len(line) == 1 or not line[1].isalpha():
                return first_char

        # Check for "Option A" or "Option B"
        upper_line = line.upper()
        if upper_line.startswith("OPTION A"):
            return "A"
        if upper_line.startswith("OPTION B"):
            return "B"

        # If first non-empty line doesn't match, stop looking
        break

    return None
