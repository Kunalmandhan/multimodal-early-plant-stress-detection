"""
Early Stress Detection Analysis.
Evaluates model performance across chronological experimental periods:
- P1: Days 1 - 14 (Baseline growth / healthy establishment)
- P2: Days 18 - 32 (Early stress onset / vegetative transition)
- P3: Days 36 - 50 (Established drought / canopy expansion)
- P4: Days 54 - 71 (Severe desiccation / pre-harvest)

Investigates whether multimodal fusion detects stress earlier than unimodal baselines.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import paths
from src.training.evaluate import compute_metrics


def analyze_early_detection(
    df_predictions: pd.DataFrame,
    variant_name: str,
) -> pd.DataFrame:
    """Computes detection performance segmented by experimental period (P1 to P4)."""
    records = []
    periods = ["P1 (Days 1-14)", "P2 (Days 18-32)", "P3 (Days 36-50)", "P4 (Days 54-71)"]

    for period in sorted(df_predictions["period"].unique()):
        sub = df_predictions[df_predictions["period"] == period]
        y_true = sub["y_true"].values
        y_pred = sub["y_pred"].values
        y_prob = sub["y_prob"].values

        n_samples = len(sub)
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))

        # Metrics
        acc = float(np.mean(y_true == y_pred)) if n_samples > 0 else 0.0
        
        # Stressed recall (Sensitivity / Detection Rate)
        if n_pos > 0:
            rec_stress = float(np.sum((y_true == 1) & (y_pred == 1)) / n_pos)
        else:
            rec_stress = np.nan  # No positives in healthy period

        # Healthy specificity
        if n_neg > 0:
            spec_healthy = float(np.sum((y_true == 0) & (y_pred == 0)) / n_neg)
        else:
            spec_healthy = np.nan

        # False positive rate in healthy period
        fpr = 1.0 - spec_healthy if not np.isnan(spec_healthy) else 0.0

        records.append({
            "model_variant": variant_name,
            "period": period,
            "samples": n_samples,
            "drought_positives": n_pos,
            "healthy_controls": n_neg,
            "accuracy": acc,
            "stress_detection_rate": rec_stress,
            "healthy_specificity": spec_healthy,
            "mean_stress_probability": float(np.mean(y_prob)),
            "mean_pos_probability": float(np.mean(y_prob[y_true == 1])) if n_pos > 0 else 0.0,
            "mean_neg_probability": float(np.mean(y_prob[y_true == 0])) if n_neg > 0 else 0.0,
        })

    return pd.DataFrame(records)


def plot_early_detection_timeline(
    df_all_periods: pd.DataFrame,
    output_path: Path,
):
    """Plots stress detection rate and mean predicted probability across periods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Palette
    palette = sns.color_palette("tab10")

    # 1. Detection Rate (Recall on Stressed Boxes)
    ax1 = axes[0]
    sns.barplot(
        data=df_all_periods.dropna(subset=["stress_detection_rate"]),
        x="period",
        y="stress_detection_rate",
        hue="model_variant",
        ax=ax1,
        palette=palette,
    )
    ax1.set_title("Stress Detection Rate (Recall) by Period", fontsize=12, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (Recall)")
    ax1.set_xlabel("Experimental Period")
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(title="Model Variant", loc="upper left")

    # 2. Predicted Stress Probability on Stressed Plants
    ax2 = axes[1]
    sns.lineplot(
        data=df_all_periods,
        x="period",
        y="mean_stress_probability",
        hue="model_variant",
        marker="o",
        linewidth=2.5,
        ax=ax2,
        palette=palette,
    )
    ax2.set_title("Mean Predicted Stress Probability Timeline", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Mean P(Stress)")
    ax2.set_xlabel("Experimental Period")
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(title="Model Variant", loc="upper left")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved early detection timeline to {output_path}")


if __name__ == "__main__":
    print("Early detection module loaded.")
