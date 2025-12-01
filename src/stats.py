"""Statistical analysis for the LLM Implicit Altruism Test study."""

import math
import statistics
from typing import List, Dict, Any, Tuple
from scipy import stats as scipy_stats


def pearson_correlation(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """
    Compute Pearson correlation coefficient and p-value.

    Args:
        xs: First variable values
        ys: Second variable values (same length as xs)

    Returns:
        Tuple of (correlation_r, p_value)
    """
    if len(xs) != len(ys):
        raise ValueError("Lists must be the same length for correlation.")
    if len(xs) < 2:
        return (float("nan"), float("nan"))

    r, p = scipy_stats.pearsonr(xs, ys)
    return (r, p)


def one_sample_ttest(values: List[float], null_mean: float = 0.0) -> Dict[str, float]:
    """
    One-sample t-test against a null hypothesis mean.

    Args:
        values: Sample values
        null_mean: Hypothesized population mean (default 0)

    Returns:
        Dict with t_statistic, p_value, mean, std_error, ci_lower, ci_upper
    """
    if len(values) < 2:
        return {
            "t_statistic": float("nan"),
            "p_value": float("nan"),
            "mean": values[0] if values else float("nan"),
            "std_error": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }

    t_stat, p_val = scipy_stats.ttest_1samp(values, null_mean)
    mean = statistics.mean(values)
    std_error = statistics.stdev(values) / math.sqrt(len(values))

    # 95% CI
    ci = scipy_stats.t.interval(
        0.95,
        df=len(values) - 1,
        loc=mean,
        scale=std_error
    )

    return {
        "t_statistic": t_stat,
        "p_value": p_val,
        "mean": mean,
        "std_error": std_error,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
    }


def independent_ttest(
    group1: List[float],
    group2: List[float]
) -> Dict[str, float]:
    """
    Independent samples t-test between two groups.

    Args:
        group1: First group values
        group2: Second group values

    Returns:
        Dict with t_statistic, p_value, mean_diff, cohens_d
    """
    if len(group1) < 2 or len(group2) < 2:
        return {
            "t_statistic": float("nan"),
            "p_value": float("nan"),
            "mean_diff": float("nan"),
            "cohens_d": float("nan"),
        }

    t_stat, p_val = scipy_stats.ttest_ind(group1, group2)
    mean_diff = statistics.mean(group1) - statistics.mean(group2)

    # Cohen's d (pooled standard deviation)
    n1, n2 = len(group1), len(group2)
    var1 = statistics.variance(group1)
    var2 = statistics.variance(group2)
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else float("nan")

    return {
        "t_statistic": t_stat,
        "p_value": p_val,
        "mean_diff": mean_diff,
        "cohens_d": cohens_d,
    }


def anova_between_models(
    model_scores: Dict[str, List[float]]
) -> Dict[str, Any]:
    """
    One-way ANOVA comparing scores across models.

    Args:
        model_scores: Dict mapping model_name -> list of scores

    Returns:
        Dict with f_statistic, p_value, df_between, df_within
    """
    groups = list(model_scores.values())

    if len(groups) < 2:
        return {
            "f_statistic": float("nan"),
            "p_value": float("nan"),
            "df_between": 0,
            "df_within": 0,
        }

    # Filter out empty groups
    groups = [g for g in groups if len(g) > 0]

    if len(groups) < 2:
        return {
            "f_statistic": float("nan"),
            "p_value": float("nan"),
            "df_between": 0,
            "df_within": 0,
        }

    f_stat, p_val = scipy_stats.f_oneway(*groups)

    df_between = len(groups) - 1
    df_within = sum(len(g) - 1 for g in groups)

    return {
        "f_statistic": f_stat,
        "p_value": p_val,
        "df_between": df_between,
        "df_within": df_within,
    }


def correlate_iat_and_decision(
    iat_summaries: Dict[str, Dict[str, float]],
    decision_summaries: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Correlate IAT altruism bias with decision altruism score across models.

    Args:
        iat_summaries: Dict model_name -> summary dict with 'mean_altruism_bias'
        decision_summaries: Dict model_name -> summary dict with 'decision_altruism_score'

    Returns:
        Dict with correlation results and model data
    """
    shared_models = sorted(
        set(iat_summaries.keys()) & set(decision_summaries.keys())
    )

    if len(shared_models) < 2:
        return {
            "models": shared_models,
            "n_models": len(shared_models),
            "r": float("nan"),
            "p_value": float("nan"),
            "iat_values": [],
            "decision_values": [],
        }

    xs = []
    ys = []
    for m in shared_models:
        iat_val = iat_summaries[m].get("mean_altruism_bias")
        dec_val = decision_summaries[m].get("decision_altruism_score")

        if iat_val is None or dec_val is None:
            continue
        if math.isnan(iat_val) or math.isnan(dec_val):
            continue

        xs.append(iat_val)
        ys.append(dec_val)

    if len(xs) < 2:
        return {
            "models": shared_models,
            "n_models": len(shared_models),
            "r": float("nan"),
            "p_value": float("nan"),
            "iat_values": xs,
            "decision_values": ys,
        }

    r, p = pearson_correlation(xs, ys)

    return {
        "models": shared_models,
        "n_models": len(xs),
        "r": r,
        "p_value": p,
        "iat_mean": statistics.mean(xs),
        "decision_mean": statistics.mean(ys),
        "iat_values": xs,
        "decision_values": ys,
    }


def generate_results_table(
    iat_summaries: Dict[str, Dict[str, float]],
    decision_summaries: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    """
    Generate a results table combining IAT and decision data for all models.

    Returns:
        List of dicts, one per model, with all summary statistics
    """
    all_models = sorted(
        set(iat_summaries.keys()) | set(decision_summaries.keys())
    )

    table = []
    for model in all_models:
        row = {"model": model}

        iat = iat_summaries.get(model, {})
        row["iat_mean"] = iat.get("mean_altruism_bias", float("nan"))
        row["iat_stdev"] = iat.get("stdev_altruism_bias", float("nan"))
        row["iat_n"] = iat.get("valid_trials", 0)
        row["iat_ci_lower"] = iat.get("ci_lower", float("nan"))
        row["iat_ci_upper"] = iat.get("ci_upper", float("nan"))

        dec = decision_summaries.get(model, {})
        row["decision_mean"] = dec.get("decision_altruism_score", float("nan"))
        row["decision_stdev"] = dec.get("stdev_decision_score", float("nan"))
        row["decision_n"] = dec.get("valid_trials", 0)

        table.append(row)

    return table
