#!/usr/bin/env python3
"""
Power analysis for LLM Altruism Study.

Determines required sample size (N models) for detecting correlations.
"""

import math
from scipy import stats


def power_for_correlation(n: int, r: float, alpha: float = 0.05) -> float:
    """
    Calculate statistical power for detecting a correlation.

    Args:
        n: Sample size
        r: Expected correlation coefficient
        alpha: Significance level (default 0.05)

    Returns:
        Statistical power (0-1)
    """
    # Fisher's z transformation
    z_r = 0.5 * math.log((1 + r) / (1 - r))

    # Standard error
    se = 1 / math.sqrt(n - 3)

    # Critical z for alpha
    z_alpha = stats.norm.ppf(1 - alpha / 2)

    # Non-centrality parameter
    ncp = z_r / se

    # Power = P(reject H0 | H1 true)
    power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)

    return power


def required_n_for_power(r: float, power: float = 0.80, alpha: float = 0.05) -> int:
    """
    Calculate required N for desired power.

    Args:
        r: Expected correlation coefficient
        power: Desired power (default 0.80)
        alpha: Significance level (default 0.05)

    Returns:
        Required sample size
    """
    for n in range(4, 500):
        if power_for_correlation(n, r, alpha) >= power:
            return n
    return 500


def main():
    print("=" * 60)
    print("POWER ANALYSIS FOR LLM ALTRUISM STUDY")
    print("=" * 60)

    # Effect size from pilot study
    pilot_r = 0.63
    conservative_r = 0.50

    print(f"\nPilot study effect size: r = {pilot_r}")
    print(f"Conservative estimate:   r = {conservative_r}")

    print("\n" + "-" * 60)
    print("Required N for Different Effect Sizes (power = 0.80, α = 0.05)")
    print("-" * 60)

    for r in [0.40, 0.50, 0.60, 0.70, 0.80]:
        n = required_n_for_power(r, power=0.80)
        print(f"  r = {r:.2f}  →  N = {n}")

    print("\n" + "-" * 60)
    print("Power for Different Sample Sizes (r = 0.50, α = 0.05)")
    print("-" * 60)

    for n in [10, 15, 20, 24, 30, 40, 50]:
        power = power_for_correlation(n, r=0.50)
        print(f"  N = {n:2d}  →  Power = {power:.2f}")

    print("\n" + "-" * 60)
    print("RECOMMENDATION")
    print("-" * 60)

    target_n = 24
    power_conservative = power_for_correlation(target_n, r=0.50)
    power_pilot = power_for_correlation(target_n, r=0.63)

    print(f"\n  With N = {target_n} models:")
    print(f"    - Power to detect r = 0.50: {power_conservative:.2f}")
    print(f"    - Power to detect r = 0.63: {power_pilot:.2f}")

    if power_conservative >= 0.70:
        print(f"\n  ✓ N = {target_n} provides adequate power (≥0.70)")
    else:
        needed = required_n_for_power(0.50, power=0.80)
        print(f"\n  ✗ Consider increasing to N = {needed} for 0.80 power")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
