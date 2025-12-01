"""IAT-style word association task runner."""

import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime

from src.word_lists import ALL_WORDS
from src.templates import build_iat_prompt
from src.parser import parse_iat_assignments
from src.scoring import compute_altruism_bias
from src.api_client import call_model


@dataclass
class IATTrialResult:
    """Result of a single IAT trial."""
    model_name: str
    template_index: int
    word_order: List[str]
    raw_response: str
    assignments: Dict[str, str]
    altruism_bias: float
    timestamp: str
    n_parsed: int
    n_expected: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def run_single_iat_trial(
    model_name: str,
    template_index: Optional[int] = None,
) -> IATTrialResult:
    """
    Run a single IAT trial.

    Args:
        model_name: OpenRouter model identifier
        template_index: Specific template (0-3) or None for random

    Returns:
        IATTrialResult with all trial data
    """
    # Randomize word order
    words = ALL_WORDS.copy()
    random.shuffle(words)

    # Build prompt
    prompt, used_template = build_iat_prompt(words, template_index)

    # Query model
    response = call_model(model_name, prompt)

    # Parse and score
    assignments = parse_iat_assignments(response)
    bias = compute_altruism_bias(assignments)

    return IATTrialResult(
        model_name=model_name,
        template_index=used_template,
        word_order=words,
        raw_response=response,
        assignments=assignments,
        altruism_bias=bias,
        timestamp=datetime.utcnow().isoformat(),
        n_parsed=len(assignments),
        n_expected=len(ALL_WORDS),
    )


def run_iat_experiment(
    model_name: str,
    n_trials: int,
    progress_callback: Optional[callable] = None,
) -> List[IATTrialResult]:
    """
    Run multiple IAT trials for a model.

    Args:
        model_name: OpenRouter model identifier
        n_trials: Number of trials to run
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        List of IATTrialResult objects
    """
    results: List[IATTrialResult] = []

    for i in range(n_trials):
        trial = run_single_iat_trial(model_name)
        results.append(trial)

        if progress_callback:
            progress_callback(i + 1, n_trials)

    return results


def summarize_iat_results(results: List[IATTrialResult]) -> Dict[str, Any]:
    """
    Compute summary statistics for IAT results.

    Returns:
        Dict with n_trials, mean_bias, stdev, min, max, valid_trials, parse_rate
    """
    import math
    import statistics

    biases = [r.altruism_bias for r in results if not math.isnan(r.altruism_bias)]

    if not biases:
        return {
            "n_trials": len(results),
            "valid_trials": 0,
            "mean_altruism_bias": float("nan"),
            "stdev_altruism_bias": float("nan"),
            "min_bias": float("nan"),
            "max_bias": float("nan"),
            "parse_rate": 0.0,
        }

    # Calculate parse rate
    total_expected = sum(r.n_expected for r in results)
    total_parsed = sum(r.n_parsed for r in results)
    parse_rate = total_parsed / total_expected if total_expected > 0 else 0.0

    return {
        "n_trials": len(results),
        "valid_trials": len(biases),
        "mean_altruism_bias": statistics.mean(biases),
        "stdev_altruism_bias": statistics.pstdev(biases) if len(biases) > 1 else 0.0,
        "min_bias": min(biases),
        "max_bias": max(biases),
        "parse_rate": parse_rate,
    }
