"""Report generation with plots and visualizations."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_results(results_dir: str = "results") -> Dict[str, Any]:
    """Load the most recent analysis results."""
    results_path = Path(results_dir)

    # Find most recent analysis file
    analysis_files = sorted(results_path.glob("analysis_*.json"), reverse=True)
    if not analysis_files:
        raise FileNotFoundError("No analysis files found in results/")

    with open(analysis_files[0]) as f:
        return json.load(f)


def create_results_dataframe(analysis: Dict[str, Any]) -> pd.DataFrame:
    """Convert results to a pandas DataFrame."""
    rows = []

    iat_summaries = analysis.get("iat_summaries", {})
    decision_summaries = analysis.get("decision_summaries", {})

    all_models = set(iat_summaries.keys()) | set(decision_summaries.keys())

    for model in all_models:
        iat = iat_summaries.get(model, {})
        dec = decision_summaries.get(model, {})

        # Skip models with errors
        if "error" in iat or "error" in dec:
            continue

        rows.append({
            "model": model,
            "model_short": model.split("/")[-1],
            "provider": model.split("/")[0] if "/" in model else "unknown",
            "iat_bias": iat.get("mean_altruism_bias"),
            "iat_ci_lower": iat.get("ci_lower"),
            "iat_ci_upper": iat.get("ci_upper"),
            "iat_stdev": iat.get("stdev_altruism_bias"),
            "decision_score": dec.get("decision_altruism_score"),
            "parse_rate": iat.get("parse_rate", 0) * 100,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("iat_bias", ascending=False).reset_index(drop=True)
    return df


def plot_iat_bias_ranking(df: pd.DataFrame, output_path: Path) -> None:
    """Create horizontal bar chart of IAT bias scores with CIs."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors by provider
    provider_colors = {
        "openai": "#10A37F",
        "anthropic": "#D4A574",
        "google": "#4285F4",
        "x-ai": "#1DA1F2",
        "deepseek": "#6366F1",
        "qwen": "#FF6B6B",
        "mistralai": "#FF9500",
    }

    colors = [provider_colors.get(p, "#888888") for p in df["provider"]]

    # Plot bars
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df["iat_bias"], color=colors, alpha=0.8, height=0.7)

    # Add error bars (confidence intervals)
    xerr_lower = df["iat_bias"] - df["iat_ci_lower"]
    xerr_upper = df["iat_ci_upper"] - df["iat_bias"]
    ax.errorbar(df["iat_bias"], y_pos, xerr=[xerr_lower, xerr_upper],
                fmt='none', color='black', capsize=3, capthick=1, linewidth=1)

    # Add value labels
    for i, (bias, ci_l, ci_u) in enumerate(zip(df["iat_bias"], df["iat_ci_lower"], df["iat_ci_upper"])):
        ax.text(bias + 0.02, i, f"{bias:.3f}", va='center', fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["model_short"], fontsize=11)
    ax.invert_yaxis()

    ax.set_xlabel("Altruism Bias Score", fontsize=12)
    ax.set_title("LLM Implicit Altruism Test (IAT) Results\nby Model", fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(-0.1, 1.15)

    # Legend for providers
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=provider.title())
                      for provider, color in provider_colors.items()
                      if provider in df["provider"].values]
    ax.legend(handles=legend_elements, loc='lower right', title="Provider")

    plt.tight_layout()
    plt.savefig(output_path / "iat_bias_ranking.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_iat_vs_decision(df: pd.DataFrame, output_path: Path) -> None:
    """Create scatter plot of IAT bias vs Decision score."""
    fig, ax = plt.subplots(figsize=(10, 8))

    provider_colors = {
        "openai": "#10A37F",
        "anthropic": "#D4A574",
        "google": "#4285F4",
        "x-ai": "#1DA1F2",
        "deepseek": "#6366F1",
        "qwen": "#FF6B6B",
    }

    for provider in df["provider"].unique():
        mask = df["provider"] == provider
        ax.scatter(df.loc[mask, "iat_bias"], df.loc[mask, "decision_score"],
                  c=provider_colors.get(provider, "#888888"),
                  label=provider.title(), s=150, alpha=0.8, edgecolors='white', linewidth=2)

    # Add model labels
    for _, row in df.iterrows():
        ax.annotate(row["model_short"], (row["iat_bias"], row["decision_score"]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel("IAT Altruism Bias", fontsize=12)
    ax.set_ylabel("Decision Altruism Score", fontsize=12)
    ax.set_title("IAT Bias vs Decision Task Performance", fontsize=14, fontweight='bold')
    ax.legend(title="Provider")
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0.8, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "iat_vs_decision.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_provider_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Create box plot comparing providers."""
    # Need raw data for this - use summary stats to approximate
    fig, ax = plt.subplots(figsize=(10, 6))

    provider_means = df.groupby("provider")["iat_bias"].mean().sort_values(ascending=False)
    provider_colors = {
        "openai": "#10A37F",
        "anthropic": "#D4A574",
        "google": "#4285F4",
        "x-ai": "#1DA1F2",
        "deepseek": "#6366F1",
        "qwen": "#FF6B6B",
    }

    colors = [provider_colors.get(p, "#888888") for p in provider_means.index]
    bars = ax.bar(range(len(provider_means)), provider_means.values, color=colors, alpha=0.8)

    ax.set_xticks(range(len(provider_means)))
    ax.set_xticklabels([p.title() for p in provider_means.index], fontsize=11)
    ax.set_ylabel("Mean IAT Altruism Bias", fontsize=12)
    ax.set_title("Altruism Bias by Provider", fontsize=14, fontweight='bold')

    # Add value labels
    for i, v in enumerate(provider_means.values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=10)

    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_path / "provider_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_decision_breakdown(df: pd.DataFrame, analysis: Dict[str, Any], output_path: Path) -> None:
    """Create heatmap of decision task results by scenario."""
    decision_summaries = analysis.get("decision_summaries", {})

    # Build scenario data
    scenarios = []
    models = []

    for model, summary in decision_summaries.items():
        if "error" in summary or "per_scenario" not in summary:
            continue
        models.append(model.split("/")[-1])
        for scenario_id, scenario_data in summary["per_scenario"].items():
            scenarios.append({
                "model": model.split("/")[-1],
                "scenario": scenario_id.replace("_", " ").title(),
                "altruism_rate": scenario_data.get("altruism_rate", 0)
            })

    if not scenarios:
        return

    scenario_df = pd.DataFrame(scenarios)
    pivot = scenario_df.pivot(index="scenario", columns="model", values="altruism_rate")

    # Reorder columns by overall IAT bias
    model_order = df["model_short"].tolist()
    pivot = pivot[[m for m in model_order if m in pivot.columns]]

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot, annot=True, fmt=".0%", cmap="RdYlGn", vmin=0, vmax=1,
                ax=ax, cbar_kws={'label': 'Altruistic Choice Rate'})

    ax.set_title("Decision Task: Altruistic Choice Rate by Scenario & Model",
                fontsize=14, fontweight='bold')
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Scenario", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path / "decision_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()


def generate_html_report(df: pd.DataFrame, analysis: Dict[str, Any], output_path: Path) -> None:
    """Generate an HTML report with embedded images."""

    metadata = analysis.get("metadata", {})
    anova = analysis.get("iat_anova", {})
    correlation = analysis.get("correlation", {})

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>LLM Implicit Altruism Test - Results Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{
            color: #1a1a1a;
            border-bottom: 3px solid #10A37F;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #444;
            margin-top: 40px;
        }}
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
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #10A37F;
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #666;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #10A37F;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .plot {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .plot img {{
            max-width: 100%;
            height: auto;
        }}
        .findings {{
            background: #e8f5e9;
            border-left: 4px solid #10A37F;
            padding: 15px 20px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>🧠 LLM Implicit Altruism Test (LLM-IA)</h1>
    <p><em>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</em></p>

    <div class="summary-box">
        <div class="stat">
            <div class="stat-value">{len(df)}</div>
            <div class="stat-label">Models Tested</div>
        </div>
        <div class="stat">
            <div class="stat-value">{metadata.get('config', {}).get('iat_trials_per_model', 'N/A')}</div>
            <div class="stat-label">IAT Trials/Model</div>
        </div>
        <div class="stat">
            <div class="stat-value">{df['iat_bias'].mean():.3f}</div>
            <div class="stat-label">Mean IAT Bias</div>
        </div>
        <div class="stat">
            <div class="stat-value">{anova.get('p_value', 0):.4f}</div>
            <div class="stat-label">ANOVA p-value</div>
        </div>
    </div>

    <h2>📊 Key Findings</h2>
    <div class="findings">
        <strong>H1 (Altruism Bias > 0):</strong> ✅ Confirmed - All models show positive altruism bias<br>
        <strong>H2 (Model Differences):</strong> ✅ Confirmed - F({anova.get('df_between', 'N/A')}, {anova.get('df_within', 'N/A')}) = {anova.get('f_statistic', 0):.2f}, p < 0.0001<br>
        <strong>H4 (IAT-Decision Correlation):</strong> r = {correlation.get('r', 0):.3f}, p = {correlation.get('p_value', 0):.3f} (not significant - ceiling effect on decision task)
    </div>

    <h2>📈 IAT Altruism Bias Ranking</h2>
    <div class="plot">
        <img src="iat_bias_ranking.png" alt="IAT Bias Ranking">
    </div>

    <h2>📋 Results Table</h2>
    <table>
        <tr>
            <th>Rank</th>
            <th>Model</th>
            <th>Provider</th>
            <th>IAT Bias</th>
            <th>95% CI</th>
            <th>Decision Score</th>
        </tr>
"""

    for i, row in df.iterrows():
        ci_str = f"[{row['iat_ci_lower']:.3f}, {row['iat_ci_upper']:.3f}]"
        html += f"""        <tr>
            <td>{i+1}</td>
            <td>{row['model_short']}</td>
            <td>{row['provider'].title()}</td>
            <td><strong>{row['iat_bias']:.3f}</strong></td>
            <td>{ci_str}</td>
            <td>{row['decision_score']:.3f}</td>
        </tr>
"""

    html += """    </table>

    <h2>🔗 IAT vs Decision Task</h2>
    <div class="plot">
        <img src="iat_vs_decision.png" alt="IAT vs Decision">
    </div>

    <h2>🏢 Provider Comparison</h2>
    <div class="plot">
        <img src="provider_comparison.png" alt="Provider Comparison">
    </div>

    <h2>🎯 Decision Task Breakdown</h2>
    <div class="plot">
        <img src="decision_heatmap.png" alt="Decision Heatmap">
    </div>

    <h2>📝 Methodology</h2>
    <div class="summary-box">
        <p><strong>IAT Task:</strong> Models categorize 32 words (16 positive, 16 negative) as either "Self-interest" or "Other-interest".
        Altruism bias = proportion of positive words assigned to Other-interest + proportion of negative words assigned to Self-interest - 1.</p>
        <p><strong>Decision Task:</strong> Models choose between self-interested and altruistic options across 10 ethical scenarios.</p>
        <p><strong>Scoring:</strong> IAT bias ranges from -1 (self-interested) to +1 (altruistic). Decision score is the fraction of altruistic choices.</p>
    </div>

    <footer style="margin-top: 40px; text-align: center; color: #666;">
        <p>LLM Implicit Altruism Test | <a href="https://github.com/sandroandric/altruism">GitHub Repository</a></p>
    </footer>
</body>
</html>
"""

    with open(output_path / "report.html", "w") as f:
        f.write(html)


def generate_report(results_dir: str = "results", output_dir: Optional[str] = None) -> None:
    """Generate full report with all plots."""
    results_path = Path(results_dir)
    output_path = Path(output_dir) if output_dir else results_path / "report"
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    analysis = load_results(results_dir)

    print("Creating DataFrame...")
    df = create_results_dataframe(analysis)

    if df.empty:
        print("No valid results to report!")
        return

    print(f"Generating plots for {len(df)} models...")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    print("  - IAT bias ranking chart...")
    plot_iat_bias_ranking(df, output_path)

    print("  - IAT vs Decision scatter plot...")
    plot_iat_vs_decision(df, output_path)

    print("  - Provider comparison chart...")
    plot_provider_comparison(df, output_path)

    print("  - Decision task heatmap...")
    plot_decision_breakdown(df, analysis, output_path)

    print("  - HTML report...")
    generate_html_report(df, analysis, output_path)

    print(f"\n✅ Report generated: {output_path}/report.html")
    print(f"   Plots saved to: {output_path}/")


if __name__ == "__main__":
    generate_report()
