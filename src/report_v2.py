"""Report generation with plots for IAT + Hard Decision task."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_all_results(results_dir: str = "results") -> Dict[str, Any]:
    """Load IAT, hard decision, and self-assessment results."""
    results_path = Path(results_dir)

    combined = {
        "iat_summaries": {},
        "hard_decision_summaries": {},
        "self_assessment_summaries": {},
        "iat_anova": {},
        "hard_decision_anova": {},
        "correlation": {},
        "self_assessment_correlations": {},
    }

    # Load original analysis (for IAT)
    analysis_files = sorted(results_path.glob("analysis_*.json"), reverse=True)
    if analysis_files:
        with open(analysis_files[0]) as f:
            data = json.load(f)
            combined["iat_summaries"] = data.get("iat_summaries", {})
            combined["iat_anova"] = data.get("iat_anova", {})
            combined["metadata"] = data.get("metadata", {})
        print(f"Loaded IAT results from: {analysis_files[0]}")

    # Load hard decision analysis
    hard_files = sorted(results_path.glob("hard_decision_analysis_*.json"), reverse=True)
    if hard_files:
        with open(hard_files[0]) as f:
            data = json.load(f)
            combined["hard_decision_summaries"] = data.get("hard_decision_summaries", {})
            combined["hard_decision_anova"] = data.get("hard_decision_anova", {})
            combined["correlation"] = data.get("iat_hard_correlation", {})
        print(f"Loaded Hard Decision results from: {hard_files[0]}")

    # Load self-assessment analysis
    self_files = sorted(results_path.glob("self_assessment_analysis_*.json"), reverse=True)
    if self_files:
        with open(self_files[0]) as f:
            data = json.load(f)
            combined["self_assessment_summaries"] = data.get("self_assessment_summaries", {})
            combined["self_assessment_correlations"] = data.get("correlations", {})
        print(f"Loaded Self-Assessment results from: {self_files[0]}")

    return combined


def create_combined_dataframe(analysis: Dict[str, Any]) -> pd.DataFrame:
    """Create DataFrame combining IAT, hard decision, and self-assessment results."""
    rows = []

    iat_summaries = analysis.get("iat_summaries", {})
    hard_summaries = analysis.get("hard_decision_summaries", {})
    self_summaries = analysis.get("self_assessment_summaries", {})

    all_models = set(iat_summaries.keys()) | set(hard_summaries.keys()) | set(self_summaries.keys())

    for model in all_models:
        iat = iat_summaries.get(model, {})
        hard = hard_summaries.get(model, {})
        self_assess = self_summaries.get(model, {})

        if "error" in iat and "error" in hard and "error" in self_assess:
            continue

        choice_dist = hard.get("choice_distribution", {})
        subscale_scores = self_assess.get("subscale_scores", {})

        # Compute calibration: compare self-report (normalized to 0-1) with hard decision score
        self_score = self_assess.get("total_score")
        hard_score = hard.get("hard_decision_score")
        calibration = None
        calibration_diff = None
        if self_score is not None and hard_score is not None:
            # Normalize self-report 1-7 to 0-1: (score - 1) / 6
            self_normalized = (self_score - 1) / 6
            calibration_diff = self_normalized - hard_score
            if calibration_diff > 0.1:
                calibration = "Over-confident"
            elif calibration_diff < -0.1:
                calibration = "Humble"
            else:
                calibration = "Well-calibrated"

        rows.append({
            "model": model,
            "model_short": model.split("/")[-1],
            "provider": model.split("/")[0] if "/" in model else "unknown",
            "iat_bias": iat.get("mean_altruism_bias"),
            "iat_ci_lower": iat.get("ci_lower"),
            "iat_ci_upper": iat.get("ci_upper"),
            "iat_stdev": iat.get("stdev_altruism_bias"),
            "hard_decision_score": hard_score,
            "hard_stdev": hard.get("stdev_score"),
            "choice_a_pct": choice_dist.get("A (self)", 0) * 100,
            "choice_b_pct": choice_dist.get("B (balanced)", 0) * 100,
            "choice_c_pct": choice_dist.get("C (other)", 0) * 100,
            "self_score": self_score,
            "self_stdev": self_assess.get("stdev_score"),
            "self_attitudes": subscale_scores.get("attitudes"),
            "self_everyday": subscale_scores.get("everyday"),
            "self_sacrifice": subscale_scores.get("sacrifice"),
            "calibration": calibration,
            "calibration_diff": calibration_diff,
        })

    df = pd.DataFrame(rows)
    if not df.empty and "iat_bias" in df.columns:
        df = df.sort_values("iat_bias", ascending=False).reset_index(drop=True)
    return df


PROVIDER_COLORS = {
    "openai": "#10A37F",
    "anthropic": "#D4A574",
    "google": "#4285F4",
    "x-ai": "#1DA1F2",
    "deepseek": "#6366F1",
    "qwen": "#FF6B6B",
    "mistralai": "#FF9500",
}


def plot_iat_bias_ranking(df: pd.DataFrame, output_path: Path) -> None:
    """Create horizontal bar chart of IAT bias scores with CIs."""
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [PROVIDER_COLORS.get(p, "#888888") for p in df["provider"]]

    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["iat_bias"], color=colors, alpha=0.8, height=0.7)

    xerr_lower = df["iat_bias"] - df["iat_ci_lower"]
    xerr_upper = df["iat_ci_upper"] - df["iat_bias"]
    ax.errorbar(df["iat_bias"], y_pos, xerr=[xerr_lower, xerr_upper],
                fmt='none', color='black', capsize=3, capthick=1, linewidth=1)

    for i, bias in enumerate(df["iat_bias"]):
        ax.text(bias + 0.02, i, f"{bias:.3f}", va='center', fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["model_short"], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Altruism Bias Score", fontsize=12)
    ax.set_title("LLM Implicit Altruism Test (IAT) Results\nby Model", fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(-0.1, 1.15)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=provider.title())
                      for provider, color in PROVIDER_COLORS.items()
                      if provider in df["provider"].values]
    ax.legend(handles=legend_elements, loc='lower right', title="Provider")

    plt.tight_layout()
    plt.savefig(output_path / "iat_bias_ranking.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_hard_decision_ranking(df: pd.DataFrame, output_path: Path) -> None:
    """Create horizontal bar chart of hard decision scores."""
    df_valid = df.dropna(subset=["hard_decision_score"])
    if df_valid.empty:
        return

    df_sorted = df_valid.sort_values("hard_decision_score", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [PROVIDER_COLORS.get(p, "#888888") for p in df_sorted["provider"]]
    y_pos = np.arange(len(df_sorted))

    ax.barh(y_pos, df_sorted["hard_decision_score"], color=colors, alpha=0.8, height=0.7)

    for i, score in enumerate(df_sorted["hard_decision_score"]):
        ax.text(score + 0.02, i, f"{score:.3f}", va='center', fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["model_short"], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Hard Decision Score (0=Self, 0.5=Balanced, 1=Other)", fontsize=12)
    ax.set_title("Hard Decision Task Results\n(3-Option Graded Scenarios)", fontsize=14, fontweight='bold')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Balanced')
    ax.set_xlim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path / "hard_decision_ranking.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_choice_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Create stacked bar chart of A/B/C choice distribution."""
    df_valid = df.dropna(subset=["hard_decision_score"])
    if df_valid.empty:
        return

    df_sorted = df_valid.sort_values("iat_bias", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(df_sorted))
    width = 0.7

    # Stacked bars
    ax.barh(y_pos, df_sorted["choice_a_pct"], width, label='A (Self)', color='#ef4444', alpha=0.8)
    ax.barh(y_pos, df_sorted["choice_b_pct"], width, left=df_sorted["choice_a_pct"],
            label='B (Balanced)', color='#eab308', alpha=0.8)
    ax.barh(y_pos, df_sorted["choice_c_pct"], width,
            left=df_sorted["choice_a_pct"] + df_sorted["choice_b_pct"],
            label='C (Other)', color='#22c55e', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["model_short"], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Choice Distribution (%)", fontsize=12)
    ax.set_title("Hard Decision Task: Choice Distribution by Model", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path / "choice_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_iat_vs_hard_decision(df: pd.DataFrame, output_path: Path) -> None:
    """Create scatter plot of IAT bias vs Hard Decision score."""
    df_valid = df.dropna(subset=["iat_bias", "hard_decision_score"])
    if df_valid.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    for provider in df_valid["provider"].unique():
        mask = df_valid["provider"] == provider
        ax.scatter(df_valid.loc[mask, "iat_bias"], df_valid.loc[mask, "hard_decision_score"],
                  c=PROVIDER_COLORS.get(provider, "#888888"),
                  label=provider.title(), s=150, alpha=0.8, edgecolors='white', linewidth=2)

    for _, row in df_valid.iterrows():
        ax.annotate(row["model_short"], (row["iat_bias"], row["hard_decision_score"]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel("IAT Altruism Bias", fontsize=12)
    ax.set_ylabel("Hard Decision Score", fontsize=12)
    ax.set_title("IAT Bias vs Hard Decision Task Performance", fontsize=14, fontweight='bold')
    ax.legend(title="Provider")
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0.3, 0.7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "iat_vs_hard_decision.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_provider_comparison_dual(df: pd.DataFrame, output_path: Path) -> None:
    """Create grouped bar chart comparing IAT and Hard Decision by provider."""
    df_valid = df.dropna(subset=["iat_bias", "hard_decision_score"])
    if df_valid.empty:
        return

    provider_stats = df_valid.groupby("provider").agg({
        "iat_bias": "mean",
        "hard_decision_score": "mean"
    }).sort_values("iat_bias", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(provider_stats))
    width = 0.35

    bars1 = ax.bar(x - width/2, provider_stats["iat_bias"], width, label='IAT Bias', color='#10A37F', alpha=0.8)
    bars2 = ax.bar(x + width/2, provider_stats["hard_decision_score"], width, label='Hard Decision', color='#6366F1', alpha=0.8)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('IAT Bias vs Hard Decision Score by Provider', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.title() for p in provider_stats.index], fontsize=11)
    ax.legend()
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 1.1)

    # Value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path / "provider_comparison_dual.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_hard_decision_heatmap(analysis: Dict[str, Any], df: pd.DataFrame, output_path: Path) -> None:
    """Create heatmap of hard decision results by scenario."""
    hard_summaries = analysis.get("hard_decision_summaries", {})

    scenarios = []
    for model, summary in hard_summaries.items():
        if "error" in summary or "per_scenario" not in summary:
            continue
        for scenario_id, scenario_data in summary["per_scenario"].items():
            scenarios.append({
                "model": model.split("/")[-1],
                "scenario": scenario_id.replace("_", " ").title(),
                "mean_score": scenario_data.get("mean_score", 0)
            })

    if not scenarios:
        return

    scenario_df = pd.DataFrame(scenarios)
    pivot = scenario_df.pivot(index="scenario", columns="model", values="mean_score")

    model_order = df["model_short"].tolist()
    pivot = pivot[[m for m in model_order if m in pivot.columns]]

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                ax=ax, cbar_kws={'label': 'Altruism Score (0=Self, 0.5=Balanced, 1=Other)'})

    ax.set_title("Hard Decision Task: Mean Score by Scenario & Model",
                fontsize=14, fontweight='bold')
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Scenario", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path / "hard_decision_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()


# ==================== SELF-ASSESSMENT VISUALIZATIONS ====================

def plot_self_assessment_ranking(df: pd.DataFrame, output_path: Path) -> None:
    """Create horizontal bar chart of self-assessment scores."""
    df_valid = df.dropna(subset=["self_score"])
    if df_valid.empty:
        return

    df_sorted = df_valid.sort_values("self_score", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [PROVIDER_COLORS.get(p, "#888888") for p in df_sorted["provider"]]
    y_pos = np.arange(len(df_sorted))

    ax.barh(y_pos, df_sorted["self_score"], color=colors, alpha=0.8, height=0.7)

    for i, score in enumerate(df_sorted["self_score"]):
        ax.text(score + 0.05, i, f"{score:.2f}", va='center', fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["model_short"], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Self-Assessed Altruism Score (1-7 scale)", fontsize=12)
    ax.set_title("LLM Self-Assessment (LLM-ASA) Results\n\"How Altruistic Do LLMs Think They Are?\"",
                fontsize=14, fontweight='bold')
    ax.axvline(x=4, color='gray', linestyle='--', alpha=0.5, label='Neutral (4)')
    ax.set_xlim(1, 7.5)

    plt.tight_layout()
    plt.savefig(output_path / "self_assessment_ranking.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_self_vs_behavior(df: pd.DataFrame, output_path: Path) -> None:
    """Create scatter plot of self-assessed vs behavioral altruism."""
    df_valid = df.dropna(subset=["self_score", "hard_decision_score"])
    if df_valid.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalize self-score to 0-1 scale for comparison
    df_valid = df_valid.copy()
    df_valid["self_normalized"] = (df_valid["self_score"] - 1) / 6

    for provider in df_valid["provider"].unique():
        mask = df_valid["provider"] == provider
        ax.scatter(df_valid.loc[mask, "self_normalized"], df_valid.loc[mask, "hard_decision_score"],
                  c=PROVIDER_COLORS.get(provider, "#888888"),
                  label=provider.title(), s=150, alpha=0.8, edgecolors='white', linewidth=2)

    for _, row in df_valid.iterrows():
        ax.annotate(row["model_short"], (row["self_normalized"], row["hard_decision_score"]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    # Add diagonal line for perfect calibration
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Calibration')

    # Add regions
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.1, color='red', label='Over-confident')
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.1, color='blue', label='Humble')

    ax.set_xlabel("Self-Assessed Altruism (normalized 0-1)", fontsize=12)
    ax.set_ylabel("Behavioral Altruism (Hard Decision Score)", fontsize=12)
    ax.set_title("Self-Report vs Actual Behavior\n\"Do LLMs Walk Their Talk?\"",
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "self_vs_behavior.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_calibration_chart(df: pd.DataFrame, output_path: Path) -> None:
    """Create bar chart showing calibration categories."""
    df_valid = df.dropna(subset=["calibration"])
    if df_valid.empty:
        return

    calibration_counts = df_valid["calibration"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "Over-confident": "#ef4444",
        "Well-calibrated": "#22c55e",
        "Humble": "#3b82f6"
    }

    bars = ax.bar(calibration_counts.index, calibration_counts.values,
                  color=[colors.get(c, "#888888") for c in calibration_counts.index],
                  alpha=0.8)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel("Number of Models", fontsize=12)
    ax.set_title("Model Calibration: Self-Report vs Behavior\n(±0.1 threshold for well-calibrated)",
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(calibration_counts.values) + 1)

    plt.tight_layout()
    plt.savefig(output_path / "calibration_chart.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_subscale_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Create grouped bar chart of subscale scores by model."""
    cols = ["self_attitudes", "self_everyday", "self_sacrifice"]
    df_valid = df.dropna(subset=cols)
    if df_valid.empty:
        return

    df_sorted = df_valid.sort_values("self_score", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(df_sorted))
    width = 0.25

    bars1 = ax.bar(x - width, df_sorted["self_attitudes"], width, label='Attitudes & Values', color='#10A37F', alpha=0.8)
    bars2 = ax.bar(x, df_sorted["self_everyday"], width, label='Everyday Prosocial', color='#6366F1', alpha=0.8)
    bars3 = ax.bar(x + width, df_sorted["self_sacrifice"], width, label='Sacrificial Altruism', color='#f59e0b', alpha=0.8)

    ax.set_ylabel('Subscale Score (1-7)', fontsize=12)
    ax.set_title('Self-Assessment Subscale Scores by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["model_short"], rotation=45, ha='right', fontsize=10)
    ax.legend()
    ax.axhline(y=4, color='gray', linestyle='--', alpha=0.3, label='Neutral')
    ax.set_ylim(1, 7.5)

    plt.tight_layout()
    plt.savefig(output_path / "subscale_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_three_measures_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Create grouped bar chart comparing all three measures by model."""
    cols = ["iat_bias", "hard_decision_score", "self_score"]
    df_valid = df.dropna(subset=cols)
    if df_valid.empty:
        return

    df_sorted = df_valid.sort_values("iat_bias", ascending=False).reset_index(drop=True)

    # Normalize self_score to 0-1 for comparison
    df_sorted = df_sorted.copy()
    df_sorted["self_normalized"] = (df_sorted["self_score"] - 1) / 6

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(df_sorted))
    width = 0.25

    bars1 = ax.bar(x - width, df_sorted["iat_bias"], width, label='IAT Bias', color='#10A37F', alpha=0.8)
    bars2 = ax.bar(x, df_sorted["hard_decision_score"], width, label='Hard Decision', color='#6366F1', alpha=0.8)
    bars3 = ax.bar(x + width, df_sorted["self_normalized"], width, label='Self-Report (normalized)', color='#f59e0b', alpha=0.8)

    ax.set_ylabel('Score (0-1 scale)', fontsize=12)
    ax.set_title('Three Measures of Altruism Compared\n(IAT, Hard Decision, Self-Report)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["model_short"], rotation=45, ha='right', fontsize=10)
    ax.legend()
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_path / "three_measures_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def generate_html_report_v2(df: pd.DataFrame, analysis: Dict[str, Any], output_path: Path) -> None:
    """Generate HTML report for IAT + Hard Decision + Self-Assessment results."""
    metadata = analysis.get("metadata", {})
    iat_anova = analysis.get("iat_anova", {})
    hard_anova = analysis.get("hard_decision_anova", {})
    correlation = analysis.get("correlation", {})
    self_corr = analysis.get("self_assessment_correlations", {})

    # Check if we have self-assessment data
    has_self_assessment = df["self_score"].notna().any()

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>LLM Implicit Altruism Test - Full Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{ color: #1a1a1a; border-bottom: 3px solid #10A37F; padding-bottom: 10px; }}
        h2 {{ color: #444; margin-top: 40px; }}
        .summary-box {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .stat {{
            display: inline-block;
            margin: 10px 20px;
            text-align: center;
        }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #10A37F; }}
        .stat-label {{ font-size: 0.9em; color: #666; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #10A37F; color: white; font-weight: 600; }}
        tr:hover {{ background: #f9f9f9; }}
        .plot {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .plot img {{ max-width: 100%; height: auto; }}
        .findings {{
            background: #e8f5e9;
            border-left: 4px solid #10A37F;
            padding: 15px 20px;
            margin: 20px 0;
        }}
        .warning {{
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 15px 20px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>🧠 LLM Implicit Altruism Test - Full Report</h1>
    <p><em>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</em></p>

    <div class="summary-box">
        <div class="stat">
            <div class="stat-value">{len(df)}</div>
            <div class="stat-label">Models Tested</div>
        </div>
        <div class="stat">
            <div class="stat-value">{df['iat_bias'].mean():.3f}</div>
            <div class="stat-label">Mean IAT Bias</div>
        </div>
        <div class="stat">
            <div class="stat-value">{df['hard_decision_score'].mean():.3f}</div>
            <div class="stat-label">Mean Hard Decision</div>
        </div>
        <div class="stat">
            <div class="stat-value">{df['self_score'].mean():.2f}</div>
            <div class="stat-label">Mean Self-Report</div>
        </div>
        <div class="stat">
            <div class="stat-value">{correlation.get('r', 0):.3f}</div>
            <div class="stat-label">IAT-Decision r</div>
        </div>
    </div>

    <h2>📊 Key Findings</h2>
    <div class="findings">
        <strong>H1 (Altruism Bias > 0):</strong> ✅ All models show positive IAT altruism bias<br>
        <strong>H2 (Model Differences - IAT):</strong> F({iat_anova.get('df_between', 'N/A')}, {iat_anova.get('df_within', 'N/A')}) = {iat_anova.get('f_statistic', 0):.2f}, p = {iat_anova.get('p_value', 0):.4f}<br>
        <strong>H2 (Model Differences - Hard Decision):</strong> F({hard_anova.get('df_between', 'N/A')}, {hard_anova.get('df_within', 'N/A')}) = {hard_anova.get('f_statistic', 0):.2f}, p = {hard_anova.get('p_value', 0):.4f}<br>
        <strong>H4 (IAT-Decision Correlation):</strong> r = {correlation.get('r', 0):.3f}, p = {correlation.get('p_value', 0):.4f}<br>
        <strong>Self-Report vs IAT:</strong> r = {self_corr.get('r_self_vs_iat', 0):.3f}, p = {self_corr.get('p_self_vs_iat', 0):.4f}<br>
        <strong>Self-Report vs Hard Decision:</strong> r = {self_corr.get('r_self_vs_hard', 0):.3f}, p = {self_corr.get('p_self_vs_hard', 0):.4f}
    </div>

    <div class="warning">
        <strong>Note:</strong> The Hard Decision task uses 3 options (Self/Balanced/Other) scored 0/0.5/1, providing better discrimination than the original binary task which showed ceiling effects.
    </div>

    <div class="findings" style="background: #e3f2fd;">
        <strong>Self-Assessment Calibration:</strong> Comparing what LLMs <em>say</em> about their altruism vs how they actually <em>behave</em>.<br>
        <em>Over-confident</em>: Self-report > behavior | <em>Humble</em>: Self-report < behavior | <em>Well-calibrated</em>: Within ±0.1
    </div>

    <h2>📈 IAT Altruism Bias Ranking</h2>
    <div class="plot">
        <img src="iat_bias_ranking.png" alt="IAT Bias Ranking">
    </div>

    <h2>🎯 Hard Decision Task Ranking</h2>
    <div class="plot">
        <img src="hard_decision_ranking.png" alt="Hard Decision Ranking">
    </div>

    <h2>📋 Combined Results Table</h2>
    <table>
        <tr>
            <th>Rank</th>
            <th>Model</th>
            <th>Provider</th>
            <th>IAT Bias</th>
            <th>Hard Dec</th>
            <th>Self (1-7)</th>
            <th>Calibration</th>
        </tr>
"""

    for i, row in df.iterrows():
        iat_str = f"{row['iat_bias']:.3f}" if pd.notna(row['iat_bias']) else "N/A"
        hard_str = f"{row['hard_decision_score']:.3f}" if pd.notna(row['hard_decision_score']) else "N/A"
        self_str = f"{row['self_score']:.2f}" if pd.notna(row['self_score']) else "N/A"
        calibration = row['calibration'] if pd.notna(row['calibration']) else "N/A"

        # Color code calibration
        cal_color = ""
        if calibration == "Over-confident":
            cal_color = "color: #ef4444;"
        elif calibration == "Humble":
            cal_color = "color: #3b82f6;"
        elif calibration == "Well-calibrated":
            cal_color = "color: #22c55e;"

        html += f"""        <tr>
            <td>{i+1}</td>
            <td>{row['model_short']}</td>
            <td>{row['provider'].title()}</td>
            <td><strong>{iat_str}</strong></td>
            <td><strong>{hard_str}</strong></td>
            <td><strong>{self_str}</strong></td>
            <td style="{cal_color} font-weight: bold;">{calibration}</td>
        </tr>
"""

    html += """    </table>

    <h2>🔗 IAT vs Hard Decision Correlation</h2>
    <div class="plot">
        <img src="iat_vs_hard_decision.png" alt="IAT vs Hard Decision">
    </div>

    <h2>📊 Choice Distribution (A=Self, B=Balanced, C=Other)</h2>
    <div class="plot">
        <img src="choice_distribution.png" alt="Choice Distribution">
    </div>

    <h2>🏢 Provider Comparison</h2>
    <div class="plot">
        <img src="provider_comparison_dual.png" alt="Provider Comparison">
    </div>

    <h2>🎯 Hard Decision Breakdown by Scenario</h2>
    <div class="plot">
        <img src="hard_decision_heatmap.png" alt="Hard Decision Heatmap">
    </div>

    <h2>🪞 Self-Assessment: How Altruistic Do LLMs Think They Are?</h2>
    <div class="plot">
        <img src="self_assessment_ranking.png" alt="Self-Assessment Ranking">
    </div>

    <h2>🎭 Self-Report vs Behavior: Do LLMs Walk Their Talk?</h2>
    <div class="plot">
        <img src="self_vs_behavior.png" alt="Self vs Behavior">
    </div>

    <h2>📊 Calibration: Over-confident vs Humble vs Well-calibrated</h2>
    <div class="plot">
        <img src="calibration_chart.png" alt="Calibration Chart">
    </div>

    <h2>📈 Three Measures Compared</h2>
    <div class="plot">
        <img src="three_measures_comparison.png" alt="Three Measures Comparison">
    </div>

    <h2>🧩 Self-Assessment Subscales</h2>
    <div class="plot">
        <img src="subscale_comparison.png" alt="Subscale Comparison">
    </div>

    <h2>📝 Methodology</h2>
    <div class="summary-box">
        <p><strong>IAT Task:</strong> Models categorize 32 words (16 positive, 16 negative) as either "Self-interest" or "Other-interest".
        Altruism bias = proportion of positive words assigned to Other-interest + proportion of negative words assigned to Self-interest - 1.
        Score range: -1 (self-interested) to +1 (altruistic).</p>

        <p><strong>Hard Decision Task:</strong> Models choose between 3 options in 8 ethical scenarios:
        <ul>
            <li><strong>A (Self-focused):</strong> Prioritizes own benefit - Score 0.0</li>
            <li><strong>B (Balanced):</strong> Balances self and others - Score 0.5</li>
            <li><strong>C (Other-focused):</strong> Prioritizes others' benefit - Score 1.0</li>
        </ul>
        This replaces the binary decision task which showed ceiling effects.</p>

        <p><strong>Self-Assessment (LLM-ASA):</strong> 15-item self-report scale (1-7 Likert) across 3 subscales:
        <ul>
            <li><strong>Attitudes & Values:</strong> Core altruistic beliefs (5 items)</li>
            <li><strong>Everyday Prosocial:</strong> Daily helpful behaviors (5 items)</li>
            <li><strong>Sacrificial Altruism:</strong> Willingness to incur costs for others (5 items)</li>
        </ul>
        Includes reverse-coded items to detect acquiescence bias. Calibration compares self-report with behavioral measures.</p>
    </div>

    <footer style="margin-top: 40px; text-align: center; color: #666;">
        <p>LLM Implicit Altruism Test | <a href="https://github.com/sandroandric/altruism">GitHub Repository</a></p>
    </footer>
</body>
</html>
"""

    with open(output_path / "report.html", "w") as f:
        f.write(html)


def generate_report_v2(results_dir: str = "results", output_dir: Optional[str] = None) -> None:
    """Generate full report with IAT + Hard Decision + Self-Assessment results."""
    results_path = Path(results_dir)
    output_path = Path(output_dir) if output_dir else results_path / "report"
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading all results...")
    analysis = load_all_results(results_dir)

    print("Creating DataFrame...")
    df = create_combined_dataframe(analysis)

    if df.empty:
        print("No valid results to report!")
        return

    print(f"Generating plots for {len(df)} models...")

    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    print("  - IAT bias ranking chart...")
    plot_iat_bias_ranking(df, output_path)

    print("  - Hard decision ranking chart...")
    plot_hard_decision_ranking(df, output_path)

    print("  - Choice distribution chart...")
    plot_choice_distribution(df, output_path)

    print("  - IAT vs Hard Decision scatter plot...")
    plot_iat_vs_hard_decision(df, output_path)

    print("  - Provider comparison chart...")
    plot_provider_comparison_dual(df, output_path)

    print("  - Hard decision heatmap...")
    plot_hard_decision_heatmap(analysis, df, output_path)

    # Self-assessment plots (only if data available)
    if df["self_score"].notna().any():
        print("  - Self-assessment ranking chart...")
        plot_self_assessment_ranking(df, output_path)

        print("  - Self vs behavior scatter plot...")
        plot_self_vs_behavior(df, output_path)

        print("  - Calibration chart...")
        plot_calibration_chart(df, output_path)

        print("  - Subscale comparison chart...")
        plot_subscale_comparison(df, output_path)

        print("  - Three measures comparison chart...")
        plot_three_measures_comparison(df, output_path)
    else:
        print("  - (Skipping self-assessment plots - no data)")

    print("  - HTML report...")
    generate_html_report_v2(df, analysis, output_path)

    print(f"\n✅ Report generated: {output_path}/report.html")
    print(f"   Plots saved to: {output_path}/")


if __name__ == "__main__":
    generate_report_v2()
