#!/usr/bin/env python3
"""
Master script to run the complete LLM Altruism Study.

This script runs all three experiments in sequence:
1. IAT (Implicit Association Test)
2. Forced Binary Choice Task
3. Self-Assessment (LLM-ASA)

Then generates analysis and figures for the paper.

Usage:
    python run_full_study.py                    # Run with default 24 models
    python run_full_study.py --test             # Test run with 2 models
    python run_full_study.py --models-file models.txt  # Custom model list
    python run_full_study.py --resume           # Resume from checkpoint
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Import full study model list (N=46 for excellent statistical power)
from src.config_full_study import FULL_STUDY_MODELS

TEST_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
]


def print_banner(text: str) -> None:
    """Print a banner."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def verify_models(models: List[str]) -> List[str]:
    """Verify models are accessible and return valid ones."""
    from src.api_client import call_model

    print_banner("VERIFYING MODEL ACCESS")
    valid_models = []

    for model in models:
        print(f"  Testing {model}...", end=" ", flush=True)
        try:
            response = call_model(model, "Say 'OK' if you can read this.")
            if response and len(response) > 0:
                print("OK")
                valid_models.append(model)
            else:
                print("EMPTY RESPONSE - skipping")
        except Exception as e:
            print(f"ERROR: {e} - skipping")

    print(f"\n  Valid models: {len(valid_models)}/{len(models)}")
    return valid_models


def run_iat_experiment(models: List[str], results_dir: Path, n_trials: int = 30, resume: bool = False) -> Dict[str, Any]:
    """Run IAT experiment for all models."""
    from src.iat_task import run_iat_experiment, summarize_iat_results

    print_banner(f"EXPERIMENT 1: IMPLICIT ASSOCIATION TEST (N={len(models)})")

    # Load checkpoint if resuming
    if resume:
        results, summaries, completed = _load_checkpoint(results_dir, "iat")
        if completed:
            print(f"  Resuming: {len(completed)} models already completed")
    else:
        results = {}
        summaries = {}
        completed = []

    for i, model in enumerate(models, 1):
        if model in completed:
            print(f"\n[{i}/{len(models)}] {model} - SKIPPED (already completed)")
            continue
        print(f"\n[{i}/{len(models)}] {model}")

        try:
            model_results = run_iat_experiment(
                model_name=model,
                n_trials=n_trials,
                progress_callback=lambda c, t: print(f"  Progress: {c}/{t}", end="\r")
            )
            results[model] = model_results
            summary = summarize_iat_results(model_results)
            summaries[model] = summary
            print(f"  IAT Bias: {summary['mean_altruism_bias']:.3f} (n={summary['valid_trials']})")

            # Checkpoint
            _save_checkpoint(results_dir, "iat", results, summaries)

        except Exception as e:
            print(f"  ERROR: {e}")
            summaries[model] = {"error": str(e)}

    return {"results": results, "summaries": summaries}


def run_forced_choice_experiment(models: List[str], results_dir: Path, n_repeats: int = 3, resume: bool = False) -> Dict[str, Any]:
    """Run Forced Choice experiment for all models."""
    from src.forced_choice_task import run_forced_choice_experiment, summarize_forced_choice_results

    print_banner(f"EXPERIMENT 2: FORCED BINARY CHOICE (N={len(models)})")

    # Load checkpoint if resuming
    if resume:
        results, summaries, completed = _load_checkpoint(results_dir, "forced_choice")
        if completed:
            print(f"  Resuming: {len(completed)} models already completed")
    else:
        results = {}
        summaries = {}
        completed = []

    for i, model in enumerate(models, 1):
        if model in completed:
            print(f"\n[{i}/{len(models)}] {model} - SKIPPED (already completed)")
            continue
        print(f"\n[{i}/{len(models)}] {model}")

        try:
            model_results = run_forced_choice_experiment(
                model_name=model,
                n_repeats=n_repeats,
                progress_callback=lambda c, t: print(f"  Progress: {c}/{t}", end="\r")
            )
            results[model] = model_results
            summary = summarize_forced_choice_results(model_results)
            summaries[model] = summary
            print(f"  Altruism Rate: {summary['altruism_rate']*100:.1f}%")

            # Checkpoint
            _save_checkpoint(results_dir, "forced_choice", results, summaries)

        except Exception as e:
            print(f"  ERROR: {e}")
            summaries[model] = {"error": str(e)}

    return {"results": results, "summaries": summaries}


def run_self_assessment_experiment(models: List[str], results_dir: Path, n_repeats: int = 3, resume: bool = False) -> Dict[str, Any]:
    """Run Self-Assessment experiment for all models."""
    from src.self_assessment_task import run_self_assessment_experiment, summarize_self_assessment_results

    print_banner(f"EXPERIMENT 3: SELF-ASSESSMENT (N={len(models)})")

    # Load checkpoint if resuming
    if resume:
        results, summaries, completed = _load_checkpoint(results_dir, "self_assessment")
        if completed:
            print(f"  Resuming: {len(completed)} models already completed")
    else:
        results = {}
        summaries = {}
        completed = []

    for i, model in enumerate(models, 1):
        if model in completed:
            print(f"\n[{i}/{len(models)}] {model} - SKIPPED (already completed)")
            continue
        print(f"\n[{i}/{len(models)}] {model}")

        try:
            model_results = run_self_assessment_experiment(
                model_name=model,
                n_repeats=n_repeats,
                progress_callback=lambda c, t: print(f"  Progress: {c}/{t}", end="\r")
            )
            results[model] = model_results
            summary = summarize_self_assessment_results(model_results)
            summaries[model] = summary
            print(f"  Self-Report: {summary['total_score']:.2f}/7")

            # Checkpoint
            _save_checkpoint(results_dir, "self_assessment", results, summaries)

        except Exception as e:
            print(f"  ERROR: {e}")
            summaries[model] = {"error": str(e)}

    return {"results": results, "summaries": summaries}


def _load_checkpoint(results_dir: Path, experiment: str) -> tuple[Dict, Dict, List[str]]:
    """Load checkpoint if exists. Returns (results, summaries, completed_models)."""
    checkpoint_path = results_dir / f"{experiment}_checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        results = checkpoint.get("raw_results", {})
        summaries = checkpoint.get("summaries", {})
        completed = checkpoint.get("models_completed", [])
        return results, summaries, completed
    return {}, {}, []


def _save_checkpoint(results_dir: Path, experiment: str, results: Dict, summaries: Dict) -> None:
    """Save checkpoint after each model - includes FULL raw data for recovery."""
    checkpoint = {
        "timestamp": datetime.utcnow().isoformat(),
        "experiment": experiment,
        "models_completed": list(results.keys()),
        "summaries": summaries,
        "raw_results": results,  # Save full raw data for recovery
    }

    checkpoint_path = results_dir / f"{experiment}_checkpoint.json"

    # Write to temp file first, then rename (atomic write for crash safety)
    temp_path = results_dir / f"{experiment}_checkpoint.tmp"
    with open(temp_path, "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)
    temp_path.rename(checkpoint_path)

    # Also save per-model backup
    backup_dir = results_dir / "model_backups" / experiment
    backup_dir.mkdir(parents=True, exist_ok=True)
    for model, data in results.items():
        safe_name = model.replace("/", "_")
        backup_path = backup_dir / f"{safe_name}.json"
        with open(backup_path, "w") as f:
            json.dump({"model": model, "summary": summaries.get(model), "raw": data}, f, indent=2, default=str)


def run_statistical_analysis(results_dir: Path, iat_data: Dict, fc_data: Dict, sa_data: Dict) -> Dict[str, Any]:
    """Run all statistical analyses."""
    from src.stats import pearson_correlation, one_sample_ttest
    import math

    print_banner("STATISTICAL ANALYSIS")

    iat_sum = iat_data["summaries"]
    fc_sum = fc_data["summaries"]
    sa_sum = sa_data["summaries"]

    # Get shared models with valid data
    shared_models = sorted(
        set(iat_sum.keys()) & set(fc_sum.keys()) & set(sa_sum.keys())
    )
    shared_models = [m for m in shared_models if "error" not in iat_sum.get(m, {})
                     and "error" not in fc_sum.get(m, {})]

    print(f"  Models with complete data: {len(shared_models)}")

    # Extract scores
    iat_scores = [iat_sum[m]["mean_altruism_bias"] for m in shared_models]
    fc_scores = [fc_sum[m]["altruism_rate"] for m in shared_models]
    sa_scores = [sa_sum[m]["total_score"] for m in shared_models]
    sa_norm = [(s - 1) / 6 for s in sa_scores]  # Normalize to 0-1

    analysis = {
        "n_models": len(shared_models),
        "models": shared_models,
    }

    # H1: IAT > 0
    print("\n  H1: IAT Bias > 0")
    ttest_result = one_sample_ttest(iat_scores, 0)
    analysis["h1_iat_positive"] = {
        "mean": ttest_result["mean"],
        "t": ttest_result["t_statistic"],
        "p": ttest_result["p_value"],
        "ci_95": (ttest_result["ci_lower"], ttest_result["ci_upper"]),
        "significant": ttest_result["p_value"] < 0.05,
    }
    print(f"    Mean IAT = {ttest_result['mean']:.3f}, t = {ttest_result['t_statistic']:.2f}, p = {ttest_result['p_value']:.4f}")

    # H2: Model differences (ANOVA)
    print("\n  H2: Model Differences in IAT")
    # Need raw trial data for ANOVA - use summaries for now
    analysis["h2_model_differences"] = {
        "note": "Requires raw trial data for proper ANOVA",
        "iat_range": [min(iat_scores), max(iat_scores)],
    }

    # H3: IAT-Behavior correlation
    print("\n  H3: IAT vs Behavior Correlation")
    r, p = pearson_correlation(iat_scores, fc_scores)
    analysis["h3_iat_behavior_corr"] = {
        "r": r,
        "p": p,
        "significant": p < 0.05,
    }
    print(f"    r = {r:.3f}, p = {p:.4f}")

    # H4: Self-Report vs Behavior correlation
    print("\n  H4: Self-Report vs Behavior Correlation")
    r, p = pearson_correlation(sa_norm, fc_scores)
    analysis["h4_self_behavior_corr"] = {
        "r": r,
        "p": p,
        "significant": p < 0.05,
    }
    print(f"    r = {r:.3f}, p = {p:.4f}")

    # H5: Calibration (Self > Behavior?)
    print("\n  H5: Calibration Analysis")
    calibration_diffs = [s - b for s, b in zip(sa_norm, fc_scores)]
    mean_diff = sum(calibration_diffs) / len(calibration_diffs)
    over_confident = sum(1 for d in calibration_diffs if d > 0.1)
    well_calibrated = sum(1 for d in calibration_diffs if abs(d) <= 0.1)
    under_confident = sum(1 for d in calibration_diffs if d < -0.1)

    analysis["h5_calibration"] = {
        "mean_difference": mean_diff,
        "over_confident": over_confident,
        "well_calibrated": well_calibrated,
        "under_confident": under_confident,
    }
    print(f"    Mean calibration diff: {mean_diff:.3f}")
    print(f"    Over-confident: {over_confident}, Well-calibrated: {well_calibrated}, Under: {under_confident}")

    # Save analysis
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    analysis_path = results_dir / f"full_study_analysis_{timestamp}.json"

    full_analysis = {
        "metadata": {
            "timestamp": timestamp,
            "n_models": len(shared_models),
        },
        "iat_summaries": iat_sum,
        "forced_choice_summaries": fc_sum,
        "self_assessment_summaries": sa_sum,
        "statistical_analysis": analysis,
    }

    with open(analysis_path, "w") as f:
        json.dump(full_analysis, f, indent=2, default=str)

    print(f"\n  Analysis saved: {analysis_path}")

    return analysis


def generate_paper_figures(results_dir: Path) -> None:
    """Generate publication-quality figures."""
    print_banner("GENERATING FIGURES")

    import subprocess
    figures_script = Path(__file__).parent / "paper" / "figures" / "generate_figures.py"

    if figures_script.exists():
        result = subprocess.run(
            [sys.executable, str(figures_script)],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"  Warning: {result.stderr}")
    else:
        print("  Figures script not found, skipping...")


def print_final_summary(iat_data: Dict, fc_data: Dict, sa_data: Dict, analysis: Dict) -> None:
    """Print final study summary."""
    print_banner("STUDY COMPLETE - FINAL SUMMARY")

    print(f"\n  Models tested: {analysis['n_models']}")

    print("\n  KEY RESULTS:")
    print(f"    H1 (IAT > 0): p = {analysis['h1_iat_positive']['p']:.4f} {'✓' if analysis['h1_iat_positive']['significant'] else '✗'}")
    print(f"    H3 (IAT-Behavior): r = {analysis['h3_iat_behavior_corr']['r']:.3f}, p = {analysis['h3_iat_behavior_corr']['p']:.4f}")
    print(f"    H4 (Self-Behavior): r = {analysis['h4_self_behavior_corr']['r']:.3f}, p = {analysis['h4_self_behavior_corr']['p']:.4f}")

    print("\n  CALIBRATION:")
    cal = analysis['h5_calibration']
    print(f"    Over-confident: {cal['over_confident']}")
    print(f"    Well-calibrated: {cal['well_calibrated']}")
    print(f"    Under-confident: {cal['under_confident']}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run complete LLM Altruism Study")
    parser.add_argument("--test", action="store_true", help="Test run with 2 models only")
    parser.add_argument("--models-file", type=str, help="File with model list (one per line)")
    parser.add_argument("--output-dir", type=str, default="results/full_study", help="Output directory")
    parser.add_argument("--iat-trials", type=int, default=30, help="IAT trials per model")
    parser.add_argument("--fc-repeats", type=int, default=3, help="Forced choice repeats per scenario")
    parser.add_argument("--sa-repeats", type=int, default=3, help="Self-assessment repeats")
    parser.add_argument("--skip-verify", action="store_true", help="Skip model verification")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    args = parser.parse_args()

    # Determine model list
    if args.test:
        models = TEST_MODELS
        print("TEST MODE: Using 2 models only")
    elif args.models_file:
        with open(args.models_file) as f:
            models = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        models = FULL_STUDY_MODELS

    # Setup output directory
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print_banner("LLM ALTRUISM STUDY - FULL PROTOCOL")
    print(f"  Models: {len(models)}")
    print(f"  IAT trials: {args.iat_trials}")
    print(f"  Forced Choice scenarios: 17 × {args.fc_repeats} = {17 * args.fc_repeats}")
    print(f"  Self-Assessment: 15 items × {args.sa_repeats} = {15 * args.sa_repeats}")
    print(f"  Output: {results_dir}")

    # Verify models
    if not args.skip_verify and not args.resume:
        models = verify_models(models)
        if len(models) < 10:
            print("\n  WARNING: Fewer than 10 valid models. Consider adding more.")

    # Save study config
    config = {
        "start_time": datetime.utcnow().isoformat(),
        "models": models,
        "iat_trials": args.iat_trials,
        "fc_repeats": args.fc_repeats,
        "sa_repeats": args.sa_repeats,
    }
    with open(results_dir / "study_config.json", "w") as f:
        json.dump(config, f, indent=2)

    start_time = time.time()

    # Run experiments (with resume support)
    iat_data = run_iat_experiment(models, results_dir, args.iat_trials, resume=args.resume)
    fc_data = run_forced_choice_experiment(models, results_dir, args.fc_repeats, resume=args.resume)
    sa_data = run_self_assessment_experiment(models, results_dir, args.sa_repeats, resume=args.resume)

    # Analysis
    analysis = run_statistical_analysis(results_dir, iat_data, fc_data, sa_data)

    # Figures
    generate_paper_figures(results_dir)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed/60:.1f} minutes")

    print_final_summary(iat_data, fc_data, sa_data, analysis)

    return 0


if __name__ == "__main__":
    sys.exit(main())
