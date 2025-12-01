"""Main experiment orchestrator for the LLM Implicit Altruism Test."""

import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.config import MODELS, EXPERIMENT_CONFIG, RESULTS_DIR
from src.iat_task import run_iat_experiment, summarize_iat_results, IATTrialResult
from src.decision_task import run_decision_experiment, summarize_decision_results, DecisionTrialResult
from src.stats import (
    one_sample_ttest,
    anova_between_models,
    correlate_iat_and_decision,
    generate_results_table,
)


class ExperimentRunner:
    """Orchestrates the full LLM-IA experiment."""

    def __init__(
        self,
        models: Optional[List[str]] = None,
        results_dir: Optional[str] = None,
    ):
        """
        Initialize the experiment runner.

        Args:
            models: List of model identifiers (defaults to config.MODELS)
            results_dir: Directory for output files (defaults to config.RESULTS_DIR)
        """
        self.models = models or MODELS
        self.results_dir = Path(results_dir or RESULTS_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed for reproducibility
        random.seed(EXPERIMENT_CONFIG["random_seed"])

        # Storage for results
        self.iat_results: Dict[str, List[IATTrialResult]] = {}
        self.decision_results: Dict[str, List[DecisionTrialResult]] = {}
        self.iat_summaries: Dict[str, Dict[str, Any]] = {}
        self.decision_summaries: Dict[str, Dict[str, Any]] = {}

        # Experiment metadata
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Incremental save file (continuously updated)
        self._checkpoint_path = self.results_dir / "checkpoint_latest.json"

    def run_iat_for_model(self, model_name: str) -> Dict[str, Any]:
        """Run IAT task for a single model."""
        print(f"\n{'='*60}")
        print(f"Running IAT task for: {model_name}")
        print(f"{'='*60}")

        n_trials = EXPERIMENT_CONFIG["iat_trials_per_model"]

        def progress(current, total):
            if current % 10 == 0 or current == total:
                print(f"  Progress: {current}/{total} trials")

        results = run_iat_experiment(
            model_name=model_name,
            n_trials=n_trials,
            progress_callback=progress,
        )

        self.iat_results[model_name] = results
        summary = summarize_iat_results(results)
        self.iat_summaries[model_name] = summary

        # Add t-test against 0
        biases = [r.altruism_bias for r in results if r.altruism_bias == r.altruism_bias]
        ttest = one_sample_ttest(biases, null_mean=0.0)
        summary.update({
            "ci_lower": ttest["ci_lower"],
            "ci_upper": ttest["ci_upper"],
            "ttest_t": ttest["t_statistic"],
            "ttest_p": ttest["p_value"],
        })

        print(f"\n  Results for {model_name}:")
        print(f"    Mean Altruism Bias: {summary['mean_altruism_bias']:.3f}")
        print(f"    95% CI: [{summary['ci_lower']:.3f}, {summary['ci_upper']:.3f}]")
        print(f"    t-test vs 0: t={ttest['t_statistic']:.2f}, p={ttest['p_value']:.4f}")
        print(f"    Parse rate: {summary['parse_rate']*100:.1f}%")

        # Save checkpoint after each model
        self._save_checkpoint()

        return summary

    def run_decision_for_model(self, model_name: str) -> Dict[str, Any]:
        """Run decision task for a single model."""
        print(f"\n{'='*60}")
        print(f"Running Decision task for: {model_name}")
        print(f"{'='*60}")

        n_repeats = EXPERIMENT_CONFIG["decision_repeats_per_scenario"]

        def progress(current, total):
            if current % 5 == 0 or current == total:
                print(f"  Progress: {current}/{total} scenarios")

        results = run_decision_experiment(
            model_name=model_name,
            n_repeats=n_repeats,
            progress_callback=progress,
        )

        self.decision_results[model_name] = results
        summary = summarize_decision_results(results)
        self.decision_summaries[model_name] = summary

        print(f"\n  Results for {model_name}:")
        print(f"    Decision Altruism Score: {summary['decision_altruism_score']:.3f}")
        print(f"    Parse rate: {summary['parse_rate']*100:.1f}%")

        # Save checkpoint after each model
        self._save_checkpoint()

        return summary

    def _save_checkpoint(self) -> None:
        """Save current progress to checkpoint file (called after each model)."""
        checkpoint = {
            "timestamp": datetime.utcnow().isoformat(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "models_completed_iat": list(self.iat_results.keys()),
            "models_completed_decision": list(self.decision_results.keys()),
            "iat_summaries": self.iat_summaries,
            "decision_summaries": self.decision_summaries,
            "iat_raw": {
                model: [r.to_dict() for r in results]
                for model, results in self.iat_results.items()
            },
            "decision_raw": {
                model: [r.to_dict() for r in results]
                for model, results in self.decision_results.items()
            },
        }

        with open(self._checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)

        print(f"  [Checkpoint saved: {len(self.iat_results)} IAT, {len(self.decision_results)} Decision models]")

    def run_full_experiment(self) -> Dict[str, Any]:
        """
        Run the complete experiment for all models.

        Returns:
            Dict with all results and analysis
        """
        self.start_time = datetime.utcnow()

        print("\n" + "="*70)
        print("LLM IMPLICIT ALTRUISM TEST (LLM-IA)")
        print("="*70)
        print(f"Models: {len(self.models)}")
        print(f"IAT trials per model: {EXPERIMENT_CONFIG['iat_trials_per_model']}")
        print(f"Decision repeats per scenario: {EXPERIMENT_CONFIG['decision_repeats_per_scenario']}")
        print("="*70)

        # Run IAT task for all models
        print("\n\n" + "#"*70)
        print("PHASE 1: IAT WORD ASSOCIATION TASK")
        print("#"*70)

        for model in self.models:
            try:
                self.run_iat_for_model(model)
            except Exception as e:
                print(f"\n  ERROR for {model}: {e}")
                self.iat_summaries[model] = {"error": str(e)}

        # Run Decision task for all models
        print("\n\n" + "#"*70)
        print("PHASE 2: DECISION ALTRUISM TASK")
        print("#"*70)

        for model in self.models:
            try:
                self.run_decision_for_model(model)
            except Exception as e:
                print(f"\n  ERROR for {model}: {e}")
                self.decision_summaries[model] = {"error": str(e)}

        self.end_time = datetime.utcnow()

        # Generate analysis
        analysis = self._generate_analysis()

        # Save results
        self._save_results(analysis)

        # Print summary
        self._print_summary(analysis)

        return analysis

    def _generate_analysis(self) -> Dict[str, Any]:
        """Generate statistical analysis of results."""
        analysis = {
            "metadata": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "models": self.models,
                "config": EXPERIMENT_CONFIG,
            },
            "iat_summaries": self.iat_summaries,
            "decision_summaries": self.decision_summaries,
        }

        # ANOVA for IAT scores across models
        iat_scores_by_model = {}
        for model, results in self.iat_results.items():
            biases = [r.altruism_bias for r in results if r.altruism_bias == r.altruism_bias]
            if biases:
                iat_scores_by_model[model] = biases

        if len(iat_scores_by_model) >= 2:
            analysis["iat_anova"] = anova_between_models(iat_scores_by_model)

        # Correlation between IAT and Decision scores
        if self.iat_summaries and self.decision_summaries:
            analysis["correlation"] = correlate_iat_and_decision(
                self.iat_summaries,
                self.decision_summaries,
            )

        # Results table
        analysis["results_table"] = generate_results_table(
            self.iat_summaries,
            self.decision_summaries,
        )

        return analysis

    def _save_results(self, analysis: Dict[str, Any]) -> None:
        """Save all results to files."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Save full analysis
        analysis_path = self.results_dir / f"analysis_{timestamp}.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\nAnalysis saved to: {analysis_path}")

        # Save raw IAT results
        iat_raw = {}
        for model, results in self.iat_results.items():
            iat_raw[model] = [r.to_dict() for r in results]

        iat_path = self.results_dir / f"iat_raw_{timestamp}.json"
        with open(iat_path, "w") as f:
            json.dump(iat_raw, f, indent=2, default=str)
        print(f"IAT raw results saved to: {iat_path}")

        # Save raw decision results
        dec_raw = {}
        for model, results in self.decision_results.items():
            dec_raw[model] = [r.to_dict() for r in results]

        dec_path = self.results_dir / f"decision_raw_{timestamp}.json"
        with open(dec_path, "w") as f:
            json.dump(dec_raw, f, indent=2, default=str)
        print(f"Decision raw results saved to: {dec_path}")

    def _print_summary(self, analysis: Dict[str, Any]) -> None:
        """Print a summary of results."""
        print("\n\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)

        # Results table
        print("\n" + "-"*70)
        print("RESULTS BY MODEL")
        print("-"*70)
        print(f"{'Model':<35} {'IAT Bias':>10} {'95% CI':>20} {'Decision':>10}")
        print("-"*70)

        for row in analysis.get("results_table", []):
            model = row["model"]
            if len(model) > 33:
                model = model[:30] + "..."

            iat = row.get("iat_mean", float("nan"))
            ci_l = row.get("iat_ci_lower", float("nan"))
            ci_u = row.get("iat_ci_upper", float("nan"))
            dec = row.get("decision_mean", float("nan"))

            iat_str = f"{iat:.3f}" if iat == iat else "N/A"
            ci_str = f"[{ci_l:.3f}, {ci_u:.3f}]" if ci_l == ci_l else "N/A"
            dec_str = f"{dec:.3f}" if dec == dec else "N/A"

            print(f"{model:<35} {iat_str:>10} {ci_str:>20} {dec_str:>10}")

        # ANOVA
        if "iat_anova" in analysis:
            anova = analysis["iat_anova"]
            print("\n" + "-"*70)
            print("ANOVA: IAT Altruism Bias Across Models")
            print("-"*70)
            print(f"  F({anova['df_between']}, {anova['df_within']}) = {anova['f_statistic']:.2f}")
            print(f"  p = {anova['p_value']:.4f}")

        # Correlation
        if "correlation" in analysis:
            corr = analysis["correlation"]
            print("\n" + "-"*70)
            print("CORRELATION: IAT Bias vs Decision Score")
            print("-"*70)
            print(f"  N models: {corr['n_models']}")
            print(f"  Pearson r = {corr['r']:.3f}")
            print(f"  p = {corr['p_value']:.4f}")

        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE")
        print("="*70)


def run_experiment(
    models: Optional[List[str]] = None,
    results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run the full experiment.

    Args:
        models: Optional list of models (defaults to config.MODELS)
        results_dir: Optional results directory (defaults to config.RESULTS_DIR)

    Returns:
        Analysis results dict
    """
    runner = ExperimentRunner(models=models, results_dir=results_dir)
    return runner.run_full_experiment()
