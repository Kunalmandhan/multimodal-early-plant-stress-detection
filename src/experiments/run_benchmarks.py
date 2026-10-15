"""
Benchmark Runner & Academic Experiment Pipeline.
Conducts 5-Fold Cross-Validation across unimodal baselines and multimodal architectures.
Computes:
- Cross-validation performance (Mean ± Std)
- Out-of-fold overall metrics
- Statistical comparison across fusion architectures
- Early stress detection evaluation across chronological periods (P1 to P4)
- Publication-quality figures (ROC, PR, Confusion Matrices, Timeline)
- Formatted summary tables in CSV, Markdown, and LaTeX
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    auc,
)
import torch

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import paths, exp_config
from src.data.dataset import create_dataloaders
from src.data.feature_cache import build_or_load_feature_cache
from src.training.train import set_seed, train_single_fold, build_model
from src.training.evaluate import compute_metrics
from src.training.early_detection import analyze_early_detection, plot_early_detection_timeline
from src.interpretability.gradcam import generate_gradcam_visualizations

# Set modern scientific aesthetic for figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 300


MODEL_VARIANTS = [
    ("rgb_only", "RGB Canopy Only (Unimodal)"),
    ("env_only", "Environmental Telemetry (Unimodal)"),
    ("diag_only", "Diagnostic Soil/SPAD (Unimodal)"),
    ("multimodal_concat", "Concat Fusion (Multimodal)"),
    ("multimodal_gated", "Gated Fusion (Proposed Multimodal)"),
    ("multimodal_cross_attn", "Cross-Attention (Multimodal)"),
]


def run_benchmark_for_variant(
    variant: str,
    variant_display: str,
    df_all: pd.DataFrame,
    splits: dict,
    cached_features: Optional[Dict[str, torch.Tensor]] = None,
    num_epochs: int = 15,
    batch_size: int = 8,
    patience: int = 5,
    device: str = "cpu",
) -> Tuple[List[dict], pd.DataFrame]:
    """Runs 5-fold cross validation for a single model variant."""
    print(f"\n========================================================", flush=True)
    print(f"BENCHMARKING: {variant_display} ({variant})", flush=True)
    print(f"========================================================", flush=True)

    fold_metrics = []
    all_predictions = []

    for fold_name in sorted(splits.keys()):
        print(f">>> Running {fold_name} for {variant}...", end=" ", flush=True)
        res = train_single_fold(
            fold_name=fold_name,
            fold_dict=splits[fold_name],
            df_all=df_all,
            variant=variant,
            cached_features=cached_features,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=exp_config.learning_rate,
            weight_decay=exp_config.weight_decay,
            patience=patience,
            device=device,
            verbose=False,
        )

        test_m = res["test_metrics"]
        test_m["fold"] = fold_name
        test_m["variant"] = variant
        fold_metrics.append(test_m)

        for p in res["test_predictions"]:
            p["variant"] = variant
            p["fold"] = fold_name
            all_predictions.append(p)

        print(
            f"    {fold_name} Done | BalAcc: {test_m['balanced_accuracy']:.4f} | "
            f"F1: {test_m['f1']:.4f} | ROC-AUC: {test_m.get('roc_auc', 0.5):.4f}"
        )

    df_preds = pd.DataFrame(all_predictions)
    return fold_metrics, df_preds


def to_markdown_safe(df: pd.DataFrame) -> str:
    """Formats DataFrame as markdown table with graceful fallback if tabulate is unavailable."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = []
        for _, r in df.iterrows():
            rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join([header, sep] + rows)


def generate_benchmark_tables(
    benchmark_summary: List[dict],
    tables_dir: Path,
):
    """Generates comparison tables in CSV, Markdown, and LaTeX formats."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    df_summary = pd.DataFrame(benchmark_summary)

    # Reorder columns logically
    col_order = [
        "Model Architecture",
        "Modality Type",
        "Accuracy (Mean±SD)",
        "Balanced Acc (Mean±SD)",
        "F1-Score (Mean±SD)",
        "ROC-AUC (Mean±SD)",
        "PR-AUC (Mean±SD)",
        "Precision (Mean±SD)",
        "Recall (Mean±SD)",
    ]
    available_cols = [c for c in col_order if c in df_summary.columns]
    df_summary = df_summary[available_cols]

    # 1. Save CSV
    csv_path = tables_dir / "benchmark_results.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"Saved benchmark CSV to {csv_path}")

    # 2. Save Markdown
    md_path = tables_dir / "benchmark_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Multimodal Plant Stress Detection — Model Benchmark Results\n\n")
        f.write("Evaluation across 5-Fold Stratified GroupKFold (grouped by physical Box ID, $N=27$ sugar beet boxes).\n")
        f.write("All test sets represent unseen plant boxes.\n\n")
        f.write(to_markdown_safe(df_summary))
        f.write("\n\n*Target: Dynamic Physiological Drought Stress (Desiccation threshold $W_{box} < 12.0$ kg).*\n")
    print(f"Saved benchmark Markdown to {md_path}")

    # 3. Save LaTeX
    tex_path = tables_dir / "benchmark_results.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Benchmark Results Table for Academic Paper\n")
        f.write(df_summary.to_latex(index=False, escape=False))
    print(f"Saved benchmark LaTeX to {tex_path}")


def plot_combined_roc_and_pr_curves(
    predictions_dict: Dict[str, pd.DataFrame],
    figures_dir: Path,
):
    """Generates publication-quality ROC and Precision-Recall Curves."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    ax_roc = axes[0]
    ax_pr = axes[1]

    colors = {
        "rgb_only": "#2b5c8f",
        "env_only": "#e26d5c",
        "diag_only": "#38b000",
        "multimodal_concat": "#f77f00",
        "multimodal_gated": "#7209b7",
        "multimodal_cross_attn": "#0077b6",
    }

    display_names = dict(MODEL_VARIANTS)

    for variant, df_preds in predictions_dict.items():
        y_true = df_preds["y_true"].values
        y_prob = df_preds["y_prob"].values
        color = colors.get(variant, "#333333")
        label_base = display_names.get(variant, variant)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_val = auc(fpr, tpr)
        ax_roc.plot(
            fpr, tpr,
            label=f"{label_base} (AUC = {roc_auc_val:.3f})",
            color=color,
            linewidth=2.0,
        )

        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc_val = auc(recall, precision)
        ax_pr.plot(
            recall, precision,
            label=f"{label_base} (AUC = {pr_auc_val:.3f})",
            color=color,
            linewidth=2.0,
        )

    # Baseline chance diagonal
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.500)")
    ax_roc.set_title("(a) Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)")
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax_roc.grid(True, linestyle="--", alpha=0.5)

    # PR curve baseline
    pos_rate = np.mean(list(predictions_dict.values())[0]["y_true"])
    ax_pr.axhline(y=pos_rate, color="k", linestyle="--", alpha=0.5, label=f"Baseline Prevalence ({pos_rate:.1%})")
    ax_pr.set_title("(b) Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax_pr.set_xlabel("Recall (Detection Rate)")
    ax_pr.set_ylabel("Precision (Positive Predictive Value)")
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.02])
    ax_pr.legend(loc="lower left", fontsize=8.5, frameon=True)
    ax_pr.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    roc_pr_path = figures_dir / "fig5_roc_pr_curves_comparison.png"
    fig.savefig(roc_pr_path, dpi=300)
    plt.close(fig)
    print(f"Saved ROC and PR curves to {roc_pr_path}")


def plot_confusion_matrices(
    predictions_dict: Dict[str, pd.DataFrame],
    figures_dir: Path,
):
    """Plots normalized confusion matrices for all evaluated architectures."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)
    axes_flat = axes.flatten()

    display_names = dict(MODEL_VARIANTS)

    for idx, (variant, df_preds) in enumerate(predictions_dict.items()):
        ax = axes_flat[idx]
        y_true = df_preds["y_true"].values
        y_pred = df_preds["y_pred"].values

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        annot = np.array([
            [f"{cm[0, 0]}\n({cm_norm[0, 0]:.1%})", f"{cm[0, 1]}\n({cm_norm[0, 1]:.1%})"],
            [f"{cm[1, 0]}\n({cm_norm[1, 0]:.1%})", f"{cm[1, 1]}\n({cm_norm[1, 1]:.1%})"],
        ])

        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Healthy", "Drought"],
            yticklabels=["Healthy", "Drought"],
            vmin=0.0,
            vmax=1.0,
        )

        title = display_names.get(variant, variant)
        acc = np.mean(y_true == y_pred)
        bal_acc = 0.5 * (cm_norm[0, 0] + cm_norm[1, 1])
        ax.set_title(f"{title}\nAcc: {acc:.1%} | BalAcc: {bal_acc:.1%}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()
    cm_path = figures_dir / "fig7_confusion_matrices.png"
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrices to {cm_path}")


def main():
    parser = argparse.ArgumentParser(description="Multimodal Plant Stress Benchmark Pipeline")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per fold")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    set_seed(exp_config.seed)
    device = exp_config.device
    print(f"Executing benchmarks on: {device}", flush=True)

    # 1. Precompute / Load visual features (cached in RAM for blazing fast training)
    print("\n--- Initializing Pretrained Visual Feature Cache ---", flush=True)
    cached_features = build_or_load_feature_cache()

    # 2. Load dataset & splits
    df_all = pd.read_csv(paths.aligned_dataset_csv)
    with open(paths.data_split_manifest_json, "r") as f:
        splits = json.load(f)["folds"]

    benchmark_summary = []
    predictions_dict = {}
    period_evaluations = []

    for variant_code, variant_name in MODEL_VARIANTS:
        fold_metrics, df_preds = run_benchmark_for_variant(
            variant=variant_code,
            variant_display=variant_name,
            df_all=df_all,
            splits=splits,
            cached_features=cached_features,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            device=device,
        )

        # Save out-of-fold predictions
        pred_csv = paths.outputs_dir / f"test_predictions_{variant_code}.csv"
        df_preds.to_csv(pred_csv, index=False)
        predictions_dict[variant_code] = df_preds

        # Compute mean +- std across folds
        df_folds = pd.DataFrame(fold_metrics)
        acc_m, acc_s = df_folds["accuracy"].mean(), df_folds["accuracy"].std()
        bal_m, bal_s = df_folds["balanced_accuracy"].mean(), df_folds["balanced_accuracy"].std()
        f1_m, f1_s = df_folds["f1"].mean(), df_folds["f1"].std()
        roc_m, roc_s = df_folds["roc_auc"].mean(), df_folds["roc_auc"].std()
        pr_m, pr_s = df_folds["pr_auc"].mean(), df_folds["pr_auc"].std()
        prec_m, prec_s = df_folds["precision"].mean(), df_folds["precision"].std()
        rec_m, rec_s = df_folds["recall"].mean(), df_folds["recall"].std()

        modality_type = "Unimodal" if "unimodal" in variant_name.lower() else "Multimodal"

        benchmark_summary.append({
            "Model Architecture": variant_name,
            "Modality Type": modality_type,
            "Accuracy (Mean±SD)": f"{acc_m:.3f} ± {acc_s:.3f}",
            "Balanced Acc (Mean±SD)": f"{bal_m:.3f} ± {bal_s:.3f}",
            "F1-Score (Mean±SD)": f"{f1_m:.3f} ± {f1_s:.3f}",
            "ROC-AUC (Mean±SD)": f"{roc_m:.3f} ± {roc_s:.3f}",
            "PR-AUC (Mean±SD)": f"{pr_m:.3f} ± {pr_s:.3f}",
            "Precision (Mean±SD)": f"{prec_m:.3f} ± {prec_s:.3f}",
            "Recall (Mean±SD)": f"{rec_m:.3f} ± {rec_s:.3f}",
        })

        # Early stress detection analysis across periods
        df_p = analyze_early_detection(df_preds, variant_name)
        period_evaluations.append(df_p)

    # 1. Output summary tables
    generate_benchmark_tables(benchmark_summary, paths.tables_dir)

    # 2. Output combined period early detection table
    df_all_periods = pd.concat(period_evaluations, ignore_index=True)
    early_csv = paths.tables_dir / "early_detection_comparison.csv"
    early_md = paths.tables_dir / "early_detection_comparison.md"
    df_all_periods.to_csv(early_csv, index=False)
    with open(early_md, "w", encoding="utf-8") as f:
        f.write("# Early Stress Detection Timeline Evaluation\n\n")
        f.write("Comparison of stress detection sensitivity across chronological developmental periods:\n")
        f.write("- **P1 (Days 1-14)**: Baseline vegetative establishment\n")
        f.write("- **P2 (Days 18-32)**: Early drought onset / asymptomatic transition\n")
        f.write("- **P3 (Days 36-50)**: Established drought stress\n")
        f.write("- **P4 (Days 54-71)**: Severe moisture deficit & pre-harvest\n\n")
        f.write(to_markdown_safe(df_all_periods))
        f.write("\n")
    print(f"Saved early detection comparison to {early_md}")

    # 3. Generate Visual Figures
    plot_combined_roc_and_pr_curves(predictions_dict, paths.figures_dir)
    plot_confusion_matrices(predictions_dict, paths.figures_dir)
    plot_early_detection_timeline(df_all_periods, paths.figures_dir / "fig8_early_detection_timeline.png")

    # 4. Generate Grad-CAM Visualizations using best multimodal gated checkpoint
    ckpt_path = paths.checkpoints_dir / "multimodal_gated_fold_1_best.pt"
    if ckpt_path.exists():
        print("\n>>> Generating Grad-CAM Canopy Visual Saliency Maps...")
        best_model = build_model("multimodal_gated", num_classes=2).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        best_model.load_state_dict(ckpt["model_state_dict"])
        generate_gradcam_visualizations(
            best_model,
            df_all,
            paths.figures_dir / "fig10_gradcam_visual_explanations.png",
        )

    print("\n========================================================")
    print("ALL BENCHMARKS, EVALUATIONS, AND FIGURES COMPLETED!")
    print("========================================================")


if __name__ == "__main__":
    main()
