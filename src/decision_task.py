"""Decision-altruism task runner."""

import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime

from src.scenarios import DECISION_SCENARIOS, DecisionScenario, build_decision_prompt
from src.parser import parse_decision_choice
from src.api_client import call_model


@dataclass
class DecisionTrialResult:
    """Result of a single decision trial."""
    model_name: str
    scenario_id: str
    raw_response: str
    chosen_option: Optional[str]  # "A", "B", or None
    altruistic_option: str
    is_altruistic: Optional[bool]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def run_single_decision_trial(
    model_name: str,
    scenario: DecisionScenario,
) -> DecisionTrialResult:
    """
    Run a single decision trial.

    Args:
        model_name: OpenRouter model identifier
        scenario: The decision scenario to present

    Returns:
        DecisionTrialResult with trial data
    """
    prompt = build_decision_prompt(scenario)
    response = call_model(model_name, prompt)
    choice = parse_decision_choice(response)

    if choice is None:
        is_alt = None
    else:
        is_alt = (choice == scenario.altruistic_option)

    return DecisionTrialResult(
        model_name=model_name,
        scenario_id=scenario.id,
        raw_response=response,
        chosen_option=choice,
        altruistic_option=scenario.altruistic_option,
        is_altruistic=is_alt,
        timestamp=datetime.utcnow().isoformat(),
    )


def run_decision_experiment(
    model_name: str,
    n_repeats: int = 1,
    shuffle_order: bool = True,
    progress_callback: Optional[callable] = None,
) -> List[DecisionTrialResult]:
    """
    Run the decision-altruism task for a model.

    Args:
        model_name: OpenRouter model identifier
        n_repeats: Number of times to run each scenario
        shuffle_order: Whether to randomize scenario order each repeat
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        List of DecisionTrialResult objects
    """
    results: List[DecisionTrialResult] = []
    total = n_repeats * len(DECISION_SCENARIOS)
    current = 0

    for _ in range(n_repeats):
        scenarios = DECISION_SCENARIOS.copy()
        if shuffle_order:
            random.shuffle(scenarios)

        for scenario in scenarios:
            trial = run_single_decision_trial(model_name, scenario)
            results.append(trial)
            current += 1

            if progress_callback:
                progress_callback(current, total)

    return results


def summarize_decision_results(results: List[DecisionTrialResult]) -> Dict[str, Any]:
    """
    Compute summary statistics for decision results.

    Returns:
        Dict with n_trials, decision_altruism_score, stdev, parse_rate, per_scenario breakdown
    """
    import statistics

    # Overall stats
    valid_choices = [r.is_altruistic for r in results if r.is_altruistic is not None]

    if not valid_choices:
        return {
            "n_trials": len(results),
            "valid_trials": 0,
            "decision_altruism_score": float("nan"),
            "stdev_decision_score": float("nan"),
            "parse_rate": 0.0,
            "per_scenario": {},
        }

    scores = [1.0 if c else 0.0 for c in valid_choices]
    parse_rate = len(valid_choices) / len(results) if results else 0.0

    # Per-scenario breakdown
    per_scenario: Dict[str, Dict[str, Any]] = {}
    for scenario in DECISION_SCENARIOS:
        scenario_results = [r for r in results if r.scenario_id == scenario.id]
        scenario_valid = [r.is_altruistic for r in scenario_results if r.is_altruistic is not None]

        if scenario_valid:
            per_scenario[scenario.id] = {
                "n_trials": len(scenario_results),
                "valid_trials": len(scenario_valid),
                "altruism_rate": sum(1 for v in scenario_valid if v) / len(scenario_valid),
            }
        else:
            per_scenario[scenario.id] = {
                "n_trials": len(scenario_results),
                "valid_trials": 0,
                "altruism_rate": float("nan"),
            }

    return {
        "n_trials": len(results),
        "valid_trials": len(valid_choices),
        "decision_altruism_score": statistics.mean(scores),
        "stdev_decision_score": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "parse_rate": parse_rate,
        "per_scenario": per_scenario,
    }
