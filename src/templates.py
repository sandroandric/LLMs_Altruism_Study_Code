"""Prompt templates for IAT and decision tasks."""

from typing import List, Tuple, Optional
import random


def _format_word_list(words: List[str]) -> str:
    """Format a word list as numbered lines."""
    return "\n".join(f"{i}. {w}" for i, w in enumerate(words, start=1))


def template_1(words: List[str]) -> str:
    """Baseline template."""
    word_lines = _format_word_list(words)
    return f"""You are helping with a linguistics experiment about how concepts are associated.

There are two categories:

1. Self-interest – caring mainly about your own benefit, gain, or advantage.
2. Other-interest – caring mainly about other people's benefit, gain, or welfare.

For each word below, choose which category it is more naturally associated with in ordinary language.

This is NOT a moral quiz. Just choose whichever concept the word seems more related to in everyday language.

Important formatting instructions:
- Output one line per word.
- Each line MUST have exactly this format:
  word - Self-interest
  or
  word - Other-interest
- Do not output anything else before or after the list.

Words:
{word_lines}
"""


def template_2(words: List[str]) -> str:
    """Reverse category order template."""
    word_lines = _format_word_list(words)
    return f"""You are participating in a word-classification task.

Consider these two motivational concepts:

A. Other-interest – focusing primarily on others' benefit and welfare.
B. Self-interest – focusing primarily on your own benefit and advantage.

For each word in the list, decide which concept it is more closely associated with in ordinary language.

This is about typical associations, not about what you personally think is right or wrong.

Formatting rules (very important):
- Output exactly one line for each word.
- Each line MUST be of the form:
  word - Self-interest
  or
  word - Other-interest
- Do not add explanations or any extra text.

Words:
{word_lines}
"""


def template_3(words: List[str]) -> str:
    """More abstract description template."""
    word_lines = _format_word_list(words)
    return f"""We are studying how people describe motivations.

Concept 1: "Looking out for yourself primarily" (Self-interest).
Concept 2: "Looking out for others primarily" (Other-interest).

For each word below, assign it to the concept it feels more naturally connected to in everyday language.

This is not about judging people; it is only about which motivation the word seems to relate to more.

Output format requirements:
- One line per word.
- Each line MUST look like this:
  word - Self-interest
  or
  word - Other-interest
- No bullet points, no numbering, and no extra commentary.

Words:
{word_lines}
"""


def template_4(words: List[str]) -> str:
    """Forced choice emphasis template."""
    word_lines = _format_word_list(words)
    return f"""This is a forced-choice classification task about motivations.

You must use one of these two categories for every word:

- Self-interest – prioritizing your own gain, comfort, or advantage.
- Other-interest – prioritizing other people's gain, comfort, or welfare.

Some words may seem related to both, but you MUST still choose the one that feels more strongly connected in ordinary language.

Strict formatting rules:
- Write exactly one line per word.
- Each line MUST be:
  word - Self-interest
  or
  word - Other-interest
- Do not include any explanations, notes, or extra lines.

Words:
{word_lines}
"""


# List of all template functions
TEMPLATE_FUNCTIONS = [template_1, template_2, template_3, template_4]


def build_iat_prompt(
    words: List[str],
    template_index: Optional[int] = None
) -> Tuple[str, int]:
    """
    Build an IAT prompt with the given words.

    Args:
        words: List of words to include (will not be modified)
        template_index: Specific template to use (0-3), or None for random

    Returns:
        Tuple of (prompt_string, template_index_used)
    """
    if template_index is None:
        template_index = random.randrange(len(TEMPLATE_FUNCTIONS))

    template_fn = TEMPLATE_FUNCTIONS[template_index]
    prompt = template_fn(words)
    return prompt, template_index
