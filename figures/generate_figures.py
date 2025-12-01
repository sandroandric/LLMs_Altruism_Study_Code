#!/usr/bin/env python3
"""Generate publication-quality figures for the research paper."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from adjustText import adjust_text

# Set publication style - LARGER FONTS for readability
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
    'axes.titleweight': 'bold',
})

# Provider colors
PROVIDER_COLORS = {
    'openai': '#10A37F',
    'anthropic': '#D97706',
    'google': '#4285F4',
    'meta-llama': '#0668E1',
    'mistralai': '#FF7000',
    'x-ai': '#000000',
    'microsoft': '#00BCF2',
    'ibm-granite': '#054ADA',
    'z-ai': '#6B7280',
    'deepseek': '#6366F1',
}

def load_data():
    results_dir = Path("../../results/full_study")

    with open(results_dir / "iat_checkpoint.json") as f:
        iat_checkpoint = json.load(f)
    with open(results_dir / "forced_choice_checkpoint.json") as f:
        fc_checkpoint = json.load(f)
    with open(results_dir / "self_assessment_checkpoint.json") as f:
        sa_checkpoint = json.load(f)

    iat_data = {"iat_summaries": iat_checkpoint.get("summaries", {})}
    fc_data = {"forced_choice_summaries": fc_checkpoint.get("summaries", {})}
    sa_data = {"self_assessment_summaries": sa_checkpoint.get("summaries", {})}

    return iat_data, fc_data, sa_data


def create_combined_df(iat_data, fc_data, sa_data):
    """Create combined DataFrame for all models."""
    iat_sum = iat_data.get('iat_summaries', {})
    fc_sum = fc_data.get('forced_choice_summaries', {})
    sa_sum = sa_data.get('self_assessment_summaries', {})

    rows = []
    for model in set(iat_sum.keys()) | set(fc_sum.keys()) | set(sa_sum.keys()):
        iat = iat_sum.get(model, {})
        fc = fc_sum.get(model, {})
        sa = sa_sum.get(model, {})

        if 'error' in iat or 'error' in fc:
            continue

        rows.append({
            'model': model,
            'model_short': model.split('/')[-1],
            'provider': model.split('/')[0],
            'iat_bias': iat.get('mean_altruism_bias'),
            'fc_altruism': fc.get('altruism_rate'),
            'self_score': sa.get('total_score'),
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=['iat_bias', 'fc_altruism'])
    return df


def fig1_iat_ranking(df, output_dir):
    """Figure 1: IAT Altruism Bias by Model."""
    df_sorted = df.sort_values('iat_bias', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    colors = [PROVIDER_COLORS.get(p, '#888888') for p in df_sorted['provider']]
    y_pos = np.arange(len(df_sorted))

    bars = ax.barh(y_pos, df_sorted['iat_bias'], color=colors, alpha=0.85, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['model_short'], fontsize=11)
    ax.set_xlabel('Implicit Altruism Bias (IAT Score)', fontsize=14)
    ax.set_title('Implicit Altruism Bias by Model (N=24)', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1.1)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_sorted['iat_bias'])):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                va='center', fontsize=10)

    # Add vertical line at mean
    mean_iat = df_sorted['iat_bias'].mean()
    ax.axvline(x=mean_iat, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_iat:.2f}')
    ax.legend(loc='lower right', fontsize=12)

    # Provider legend
    providers = df_sorted['provider'].unique()
    handles = [mpatches.Patch(color=PROVIDER_COLORS.get(p, '#888'), label=p) for p in sorted(providers)]
    ax.legend(handles=handles, loc='lower right', fontsize=10, title='Provider')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_iat_ranking.pdf')
    plt.savefig(output_dir / 'fig1_iat_ranking.png')
    plt.close()
    print("Generated: fig1_iat_ranking")


def fig2_forced_choice_ranking(df, output_dir):
    """Figure 2: Behavioral Altruism by Model."""
    df_sorted = df.sort_values('fc_altruism', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    colors = [PROVIDER_COLORS.get(p, '#888888') for p in df_sorted['provider']]
    y_pos = np.arange(len(df_sorted))

    bars = ax.barh(y_pos, df_sorted['fc_altruism'] * 100, color=colors, alpha=0.85, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['model_short'], fontsize=11)
    ax.set_xlabel('Behavioral Altruism Rate (%)', fontsize=14)
    ax.set_title('Behavioral Altruism by Model (N=24)', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 100)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_sorted['fc_altruism'])):
        ax.text(val*100 + 1, bar.get_y() + bar.get_height()/2, f'{val*100:.1f}%',
                va='center', fontsize=10)

    # Add vertical lines
    ax.axvline(x=50, color='gray', linestyle=':', linewidth=2, label='Chance (50%)')
    mean_fc = df_sorted['fc_altruism'].mean() * 100
    ax.axvline(x=mean_fc, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_fc:.1f}%')
    ax.legend(loc='lower right', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_forced_choice_ranking.pdf')
    plt.savefig(output_dir / 'fig2_forced_choice_ranking.png')
    plt.close()
    print("Generated: fig2_forced_choice_ranking")


def fig3_iat_vs_behavior(df, output_dir):
    """Figure 3: IAT vs Behavioral Altruism Scatter."""
    fig, ax = plt.subplots(figsize=(12, 9))

    colors = [PROVIDER_COLORS.get(p, '#888888') for p in df['provider']]

    scatter = ax.scatter(df['iat_bias'], df['fc_altruism'] * 100,
                         c=colors, s=150, alpha=0.7, edgecolors='white', linewidth=2)

    # Collect text objects for adjustment
    texts = []
    for _, row in df.iterrows():
        short_name = row['model_short'][:12]
        txt = ax.text(row['iat_bias'], row['fc_altruism']*100, short_name,
                      fontsize=9, alpha=0.9)
        texts.append(txt)

    # Adjust text positions to avoid overlaps
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, lw=0.5),
                expand_points=(1.5, 1.5), force_points=(0.5, 0.5))

    # Correlation line
    r, p = stats.pearsonr(df['iat_bias'], df['fc_altruism'])
    z = np.polyfit(df['iat_bias'], df['fc_altruism'] * 100, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(df['iat_bias'].min(), df['iat_bias'].max(), 100)
    ax.plot(x_line, p_line(x_line), 'r--', linewidth=2, alpha=0.7)

    ax.set_xlabel('Implicit Altruism Bias (IAT)', fontsize=14)
    ax.set_ylabel('Behavioral Altruism Rate (%)', fontsize=14)
    ax.set_title(f'IAT vs Behavior (r = {r:.2f}, p = {p:.3f})', fontsize=16, fontweight='bold')

    # Add reference lines
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Chance (50%)')

    # Provider legend
    providers = df['provider'].unique()
    handles = [mpatches.Patch(color=PROVIDER_COLORS.get(p, '#888'), label=p) for p in sorted(providers)]
    ax.legend(handles=handles, loc='upper left', fontsize=10, title='Provider')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_iat_vs_behavior.pdf')
    plt.savefig(output_dir / 'fig3_iat_vs_behavior.png')
    plt.close()
    print("Generated: fig3_iat_vs_behavior")


def fig4_category_breakdown(df, output_dir):
    """Figure 4: Provider comparison."""
    provider_stats = df.groupby('provider').agg({
        'fc_altruism': ['mean', 'std', 'count']
    }).reset_index()
    provider_stats.columns = ['provider', 'mean', 'std', 'count']
    provider_stats = provider_stats[provider_stats['count'] >= 1]
    provider_stats = provider_stats.sort_values('mean', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(provider_stats))
    colors = [PROVIDER_COLORS.get(p, '#888888') for p in provider_stats['provider']]

    bars = ax.barh(y_pos, provider_stats['mean'] * 100,
                   xerr=provider_stats['std'] * 100,
                   color=colors, alpha=0.85, height=0.6, capsize=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{p} (n={int(c)})" for p, c in zip(provider_stats['provider'], provider_stats['count'])],
                       fontsize=12)
    ax.set_xlabel('Behavioral Altruism Rate (%)', fontsize=14)
    ax.set_title('Behavioral Altruism by Provider', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 100)

    # Add value labels
    for bar, mean in zip(bars, provider_stats['mean']):
        ax.text(mean*100 + 3, bar.get_y() + bar.get_height()/2, f'{mean*100:.1f}%',
                va='center', fontsize=11)

    ax.axvline(x=50, color='gray', linestyle=':', linewidth=2, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_category_breakdown.pdf')
    plt.savefig(output_dir / 'fig4_category_breakdown.png')
    plt.close()
    print("Generated: fig4_category_breakdown")


def fig5_calibration(df, output_dir):
    """Figure 5: Self-Report vs Behavior (Calibration)."""
    df = df.dropna(subset=['self_score', 'fc_altruism'])

    fig, ax = plt.subplots(figsize=(12, 9))

    # Normalize self-report to 0-100%
    self_norm = (df['self_score'] - 1) / 6 * 100
    behavior = df['fc_altruism'] * 100

    colors = [PROVIDER_COLORS.get(p, '#888888') for p in df['provider']]

    ax.scatter(behavior, self_norm, c=colors, s=150, alpha=0.7, edgecolors='white', linewidth=2)

    # Collect text objects for adjustment
    texts = []
    for _, row in df.iterrows():
        s_norm = (row['self_score'] - 1) / 6 * 100
        b = row['fc_altruism'] * 100
        short_name = row['model_short'][:12]
        txt = ax.text(b, s_norm, short_name, fontsize=9, alpha=0.9)
        texts.append(txt)

    # Adjust text positions to avoid overlaps
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, lw=0.5),
                expand_points=(1.5, 1.5), force_points=(0.5, 0.5))

    # Diagonal line (perfect calibration)
    ax.plot([40, 100], [40, 100], 'k--', linewidth=2, alpha=0.5, label='Perfect Calibration')

    # Fill regions
    ax.fill_between([40, 100], [40, 100], [100, 100], alpha=0.1, color='red', label='Overconfident')
    ax.fill_between([40, 100], [40, 40], [40, 100], alpha=0.1, color='blue', label='Underconfident')

    ax.set_xlabel('Actual Behavioral Altruism (%)', fontsize=14)
    ax.set_ylabel('Self-Reported Altruism (%)', fontsize=14)
    ax.set_title('Calibration: Self-Report vs Behavior', fontsize=16, fontweight='bold')
    ax.set_xlim(40, 100)
    ax.set_ylim(40, 100)

    ax.legend(loc='lower right', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_calibration.pdf')
    plt.savefig(output_dir / 'fig5_calibration.png')
    plt.close()
    print("Generated: fig5_calibration")


def fig6_three_measures(df, output_dir):
    """Figure 6: All three measures comparison."""
    df = df.dropna(subset=['iat_bias', 'fc_altruism', 'self_score'])
    df_sorted = df.sort_values('fc_altruism', ascending=False)

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(df_sorted))
    width = 0.25

    # Normalize all to 0-1 scale
    iat_vals = df_sorted['iat_bias']
    fc_vals = df_sorted['fc_altruism']
    sa_vals = (df_sorted['self_score'] - 1) / 6

    bars1 = ax.bar(x - width, iat_vals, width, label='IAT (Implicit)', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, fc_vals, width, label='Behavior', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, sa_vals, width, label='Self-Report', color='#e74c3c', alpha=0.8)

    ax.set_ylabel('Score (Normalized 0-1)', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_title('Three Measures of Altruism Compared', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted['model_short'], rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_ylim(0, 1.1)

    # Add horizontal line at 0.5
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_three_measures.pdf')
    plt.savefig(output_dir / 'fig6_three_measures.png')
    plt.close()
    print("Generated: fig6_three_measures")


def fig7_quadrant(df, output_dir):
    """Figure 7: Behavior vs Calibration Quadrant Plot."""
    df = df.dropna(subset=['fc_altruism', 'self_score'])

    fig, ax = plt.subplots(figsize=(12, 10))

    # Turn off the grid for this plot - it conflicts with quadrant axes
    ax.grid(False)

    behavior = df['fc_altruism']
    self_norm = (df['self_score'] - 1) / 6
    calibration = self_norm - behavior  # Positive = overconfident

    colors = [PROVIDER_COLORS.get(p, '#888888') for p in df['provider']]

    # Quadrant boundaries
    beh_mid = 0.65
    cal_mid = 0

    # Quadrant dividing lines (thicker and more prominent)
    ax.axhline(y=cal_mid, color='#333333', linestyle='-', linewidth=1.5, alpha=0.8)
    ax.axvline(x=beh_mid, color='#333333', linestyle='-', linewidth=1.5, alpha=0.8)

    # Quadrant labels
    ax.text(0.52, 0.28, 'Low Altruism\nOverconfident', fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffcccc', alpha=0.4))
    ax.text(0.78, 0.28, 'High Altruism\nOverconfident', fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffcc', alpha=0.4))
    ax.text(0.52, -0.12, 'Low Altruism\nWell-Calibrated', fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#cccccc', alpha=0.4))
    ax.text(0.78, -0.12, 'High Altruism\nWell-Calibrated\n(IDEAL)', fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ccffcc', alpha=0.4))

    # Scatter
    scatter = ax.scatter(behavior, calibration, c=colors, s=180, alpha=0.8,
                        edgecolors='white', linewidth=2)

    # Collect text objects for adjustment
    texts = []
    for _, row in df.iterrows():
        b = row['fc_altruism']
        s_norm = (row['self_score'] - 1) / 6
        c = s_norm - b
        short_name = row['model_short'][:15]
        txt = ax.text(b, c, short_name, fontsize=9, alpha=0.9)
        texts.append(txt)

    # Adjust text positions to avoid overlaps
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, lw=0.5),
                expand_points=(1.5, 1.5), force_points=(0.5, 0.5))

    ax.set_xlabel('Behavioral Altruism Rate', fontsize=14)
    ax.set_ylabel('Calibration Error\n(Self-Report - Behavior)', fontsize=14)
    ax.set_title('Behavior vs Calibration Quadrant Analysis', fontsize=16, fontweight='bold')

    ax.set_xlim(0.42, 0.92)
    ax.set_ylim(-0.25, 0.40)

    # Format axes
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:+.0f}%'))

    # Provider legend
    providers = df['provider'].unique()
    handles = [mpatches.Patch(color=PROVIDER_COLORS.get(p, '#888'), label=p) for p in sorted(providers)]
    ax.legend(handles=handles, loc='upper left', fontsize=10, title='Provider')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_quadrant_analysis.pdf')
    plt.savefig(output_dir / 'fig7_quadrant_analysis.png')
    plt.close()
    print("Generated: fig7_quadrant_analysis")


def main():
    print("Loading data...")
    iat_data, fc_data, sa_data = load_data()

    print("Creating combined DataFrame...")
    df = create_combined_df(iat_data, fc_data, sa_data)
    print(f"Generating figures for {len(df)} models...")

    output_dir = Path(__file__).parent

    fig1_iat_ranking(df, output_dir)
    fig2_forced_choice_ranking(df, output_dir)
    fig3_iat_vs_behavior(df, output_dir)
    fig4_category_breakdown(df, output_dir)
    fig5_calibration(df, output_dir)
    fig6_three_measures(df, output_dir)
    fig7_quadrant(df, output_dir)

    print(f"\nAll figures saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
