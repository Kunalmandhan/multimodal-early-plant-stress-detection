"""
Feature Attribution and Modality Importance Module.
Computes:
1. Permutation feature importance for environmental and diagnostic tabular inputs.
2. Ranking of individual atmospheric variables and in-situ phenotyping metrics.
"""

import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import paths, exp_config
from src.data.dataset import ENV_FEATURE_COLS, DIAG_FEATURE_COLS, create_dataloaders
from src.data.feature_cache import build_or_load_feature_cache
from src.models.multimodal_fusion import MultimodalPlantStressModel
from src.training.evaluate import compute_metrics
from src.training.train import build_model


def compute_permutation_importance(
    model: MultimodalPlantStressModel,
    loader: torch.utils.data.DataLoader,
    feature_names: List[str],
    modality: str = "env",  # 'env' or 'diag'
    device: str = "cpu",
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Computes permutation feature importance by measuring decrease in ROC-AUC upon shuffling."""
    model.eval()

    # 1. Baseline performance
    all_y_true = []
    all_y_prob = []
    with torch.no_grad():
        for batch in loader:
            has_cached = batch.get("has_cached_image", [False])[0] if "has_cached_image" in batch else False
            img = batch["image"].to(device) if (model.use_image and not has_cached and batch["image"].numel() > 0) else None
            img_feats = batch["image_features"].to(device) if (model.use_image and has_cached) else None
            env = batch["env_features"].to(device) if model.use_env else None
            diag = batch["diag_features"].to(device) if model.use_diag else None
            targets = batch["target_drought"].to(device)

            logits, _ = model(image=img, env_features=env, diag_features=diag, image_features=img_feats)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_y_true.extend(targets.cpu().numpy())
            all_y_prob.extend(probs)

    base_metrics = compute_metrics(np.array(all_y_true), (np.array(all_y_prob) >= 0.5).astype(int), np.array(all_y_prob))
    base_auc = base_metrics.get("roc_auc", 0.5)

    importance_results = []

    for f_idx, f_name in enumerate(feature_names):
        drop_scores = []
        for _ in range(n_repeats):
            shuffled_probs = []
            with torch.no_grad():
                for batch in loader:
                    has_cached = batch.get("has_cached_image", [False])[0] if "has_cached_image" in batch else False
                    img = batch["image"].to(device) if (model.use_image and not has_cached and batch["image"].numel() > 0) else None
                    img_feats = batch["image_features"].to(device) if (model.use_image and has_cached) else None
                    env = batch["env_features"].to(device) if model.use_env else None
                    diag = batch["diag_features"].to(device) if model.use_diag else None

                    # Shuffle the specific column in this batch
                    if modality == "env" and env is not None:
                        env_shuffled = env.clone()
                        perm = torch.randperm(env.size(0))
                        env_shuffled[:, f_idx] = env[perm, f_idx]
                        logits, _ = model(image=img, env_features=env_shuffled, diag_features=diag, image_features=img_feats)
                    elif modality == "diag" and diag is not None:
                        diag_shuffled = diag.clone()
                        perm = torch.randperm(diag.size(0))
                        diag_shuffled[:, f_idx] = diag[perm, f_idx]
                        logits, _ = model(image=img, env_features=env, diag_features=diag_shuffled, image_features=img_feats)
                    else:
                        logits, _ = model(image=img, env_features=env, diag_features=diag, image_features=img_feats)

                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    shuffled_probs.extend(probs)

            shuff_metrics = compute_metrics(np.array(all_y_true), (np.array(shuffled_probs) >= 0.5).astype(int), np.array(shuffled_probs))
            shuff_auc = shuff_metrics.get("roc_auc", 0.5)
            # Importance is AUC drop
            drop_scores.append(max(0.0, base_auc - shuff_auc))

        importance_results.append({
            "feature": f_name,
            "modality": modality.upper(),
            "mean_auc_drop": float(np.mean(drop_scores)),
            "std_auc_drop": float(np.std(drop_scores)),
        })

    return pd.DataFrame(importance_results)


def plot_feature_importance(
    df_importance: pd.DataFrame,
    output_path: Path,
):
    """Plots permutation importance ranking for environmental and diagnostic signals."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    # Sort by importance
    df_sorted = df_importance.sort_values("mean_auc_drop", ascending=True)

    colors = ["#2b5c8f" if m == "DIAG" else "#e26d5c" for m in df_sorted["modality"]]
    
    y_pos = np.arange(len(df_sorted))
    ax.barh(y_pos, df_sorted["mean_auc_drop"], xerr=df_sorted["std_auc_drop"], color=colors, capsize=3, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["feature"], fontsize=9)
    ax.set_title("Permutation Feature Importance (Decrease in Test ROC-AUC)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Test ROC-AUC Reduction upon Feature Permutation (5 Repeats)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2b5c8f", label="Diagnostic Tabular (Weight / SPAD)"),
        Patch(facecolor="#e26d5c", label="Atmospheric Telemetry (Temp / RH / VPD)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved feature importance figure to {output_path}")


def main():
    device = exp_config.device
    print(f"Running Feature Importance Analysis on {device}...")

    df_all = pd.read_csv(paths.aligned_dataset_csv)
    with open(paths.data_split_manifest_json, "r") as f:
        splits = json.load(f)["folds"]

    cached_features = build_or_load_feature_cache()

    # Load best multimodal gated checkpoint from fold 1
    ckpt_path = paths.checkpoints_dir / "multimodal_gated_fold_1_best.pt"
    if not ckpt_path.exists():
        print("Checkpoint not found!")
        return

    model = build_model("multimodal_gated", num_classes=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Evaluate on fold 1 test set
    _, _, test_loader, _ = create_dataloaders(
        df_all, splits["fold_1"], cached_features=cached_features, batch_size=8
    )

    print("Computing permutation importance for Environmental Telemetry features...")
    df_env = compute_permutation_importance(model, test_loader, ENV_FEATURE_COLS, modality="env", device=device)

    print("Computing permutation importance for Diagnostic features...")
    df_diag = compute_permutation_importance(model, test_loader, DIAG_FEATURE_COLS, modality="diag", device=device)

    df_combined = pd.concat([df_diag, df_env], ignore_index=True)
    csv_path = paths.tables_dir / "feature_importance.csv"
    df_combined.to_csv(csv_path, index=False)
    print(f"Saved feature importance table to {csv_path}")

    fig_path = paths.figures_dir / "fig9_feature_importance.png"
    plot_feature_importance(df_combined, fig_path)


if __name__ == "__main__":
    main()
