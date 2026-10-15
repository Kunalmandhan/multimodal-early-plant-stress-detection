"""
Training Engine for Multimodal Plant Stress Detection.
Supports unimodal baselines and multimodal architectures across cross-validation folds.
Implements:
- Class-weighted CrossEntropyLoss for imbalanced stress classes
- Early stopping with patience on validation balanced accuracy / macro-F1
- Checkpoint persistence and reproducible random seeding
- Test evaluation with comprehensive metric tracking
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import paths, exp_config
from src.data.dataset import create_dataloaders
from src.models.multimodal_fusion import MultimodalPlantStressModel
from src.training.evaluate import compute_metrics


def set_seed(seed: int = 42):
    """Sets deterministic random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(
    variant: str,
    num_classes: int = 2,
    fine_tune_backbone: bool = True,
    dropout: float = 0.3,
) -> MultimodalPlantStressModel:
    """Instantiates model architecture according to experiment variant."""
    variant_lower = variant.lower()

    if variant_lower == "rgb_only":
        return MultimodalPlantStressModel(
            num_classes=num_classes,
            use_image=True,
            use_env=False,
            use_diag=False,
            fine_tune_backbone=fine_tune_backbone,
            dropout=dropout,
        )
    elif variant_lower == "env_only":
        return MultimodalPlantStressModel(
            num_classes=num_classes,
            use_image=False,
            use_env=True,
            use_diag=False,
            dropout=dropout,
        )
    elif variant_lower == "diag_only":
        return MultimodalPlantStressModel(
            num_classes=num_classes,
            use_image=False,
            use_env=False,
            use_diag=True,
            dropout=dropout,
        )
    elif variant_lower == "multimodal_concat":
        return MultimodalPlantStressModel(
            num_classes=num_classes,
            fusion_type="concat",
            use_image=True,
            use_env=True,
            use_diag=True,
            fine_tune_backbone=fine_tune_backbone,
            dropout=dropout,
        )
    elif variant_lower == "multimodal_gated":
        return MultimodalPlantStressModel(
            num_classes=num_classes,
            fusion_type="gated",
            use_image=True,
            use_env=True,
            use_diag=True,
            fine_tune_backbone=fine_tune_backbone,
            dropout=dropout,
        )
    elif variant_lower == "multimodal_cross_attn":
        return MultimodalPlantStressModel(
            num_classes=num_classes,
            fusion_type="cross_attention",
            use_image=True,
            use_env=True,
            use_diag=True,
            fine_tune_backbone=fine_tune_backbone,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model variant: {variant}")


def calculate_class_weights(df_train: pd.DataFrame, target_col: str = "target_drought_stress") -> torch.Tensor:
    """Computes inverse class frequencies to counteract class imbalance."""
    counts = df_train[target_col].value_counts().sort_index()
    total = len(df_train)
    num_classes = len(counts)
    weights = [total / (num_classes * counts[c]) for c in sorted(counts.keys())]
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
    is_train: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    """Runs a single epoch of training or evaluation."""
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    metadata_list = []

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            has_cached = batch.get("has_cached_image", [False])[0] if "has_cached_image" in batch else False
            images = batch["image"].to(device) if (model.use_image and not has_cached and batch["image"].numel() > 0) else None
            img_feats = batch["image_features"].to(device) if (model.use_image and has_cached) else None
            env_feat = batch["env_features"].to(device) if model.use_env else None
            diag_feat = batch["diag_features"].to(device) if model.use_diag else None
            targets = batch["target_drought"].to(device)

            if is_train:
                optimizer.zero_grad()

            logits, _ = model(image=images, env_features=env_feat, diag_features=diag_feat, image_features=img_feats)
            loss = criterion(logits, targets)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

            total_loss += loss.item() * len(targets)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_y_true.extend(targets.cpu().numpy())
            all_y_pred.extend(preds)
            all_y_prob.extend(probs)

            # Record batch metadata for fine-grained per-period evaluation
            for i in range(len(targets)):
                metadata_list.append({
                    "sample_id": batch["sample_id"][i],
                    "box_id": int(batch["box_id"][i]),
                    "date_str": batch["date_str"][i],
                    "period": batch["period"][i],
                    "y_true": int(targets[i].cpu().item()),
                    "y_pred": int(preds[i]),
                    "y_prob": float(probs[i][1]) if probs.shape[1] > 1 else float(probs[i][0]),
                })

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob), metadata_list


def train_single_fold(
    fold_name: str,
    fold_dict: dict,
    df_all: pd.DataFrame,
    variant: str = "multimodal_gated",
    cached_features: Optional[Dict[str, torch.Tensor]] = None,
    num_epochs: int = 25,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 7,
    device: str = "cpu",
    checkpoint_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict:
    """Trains a model on a single cross-validation fold with early stopping."""
    if checkpoint_dir is None:
        checkpoint_dir = paths.checkpoints_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoint_dir / f"{variant}_{fold_name}_best.pt"

    # Create dataloaders with fit-on-train feature scaling
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        df_all, fold_dict, cached_features=cached_features, batch_size=batch_size, image_size=exp_config.image_size
    )

    # Class weights from training set
    train_ids = set(fold_dict["train_sample_ids"])
    df_train = df_all[df_all["sample_id"].isin(train_ids)]
    class_weights = calculate_class_weights(df_train).to(device)

    model = build_model(variant=variant, num_classes=2, dropout=exp_config.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_score = -float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_true, train_pred, train_prob, _ = run_epoch(
            model, train_loader, criterion, optimizer, device=device, is_train=True
        )
        val_loss, val_true, val_pred, val_prob, _ = run_epoch(
            model, val_loader, criterion, device=device, is_train=False
        )
        scheduler.step()

        val_metrics = compute_metrics(val_true, val_pred, val_prob, num_classes=2)
        # We track balanced accuracy for model checkpointing on imbalanced datasets
        val_score = val_metrics["balanced_accuracy"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_roc_auc": val_metrics.get("roc_auc", 0.5),
            "val_pr_auc": val_metrics.get("pr_auc", 0.0),
        }
        history.append(epoch_record)

        if verbose:
            print(
                f"[{fold_name} | {variant}] Epoch {epoch:02d}/{num_epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val BalAcc: {val_metrics['balanced_accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f}"
            )

        # Early stopping check
        if val_score > best_val_score + 1e-4:
            best_val_score = val_score
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "variant": variant,
                "fold": fold_name,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch})")
                break

    # Load best model for testing
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_true, test_pred, test_prob, test_metadata = run_epoch(
        model, test_loader, criterion, device=device, is_train=False
    )
    test_metrics = compute_metrics(test_true, test_pred, test_prob, num_classes=2)
    test_metrics["test_loss"] = test_loss
    test_metrics["best_epoch"] = best_epoch

    if verbose:
        print(
            f"[{fold_name} | {variant}] TEST RESULTS: "
            f"Acc: {test_metrics['accuracy']:.4f} | "
            f"BalAcc: {test_metrics['balanced_accuracy']:.4f} | "
            f"F1: {test_metrics['f1']:.4f} | "
            f"ROC-AUC: {test_metrics.get('roc_auc', 0.5):.4f} | "
            f"PR-AUC: {test_metrics.get('pr_auc', 0.0):.4f}"
        )

    return {
        "fold": fold_name,
        "variant": variant,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "test_predictions": test_metadata,
        "history": history,
        "checkpoint_path": str(ckpt_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Multimodal Plant Stress Detection")
    parser.add_argument("--variant", type=str, default="multimodal_gated",
                        choices=["rgb_only", "env_only", "diag_only",
                                 "multimodal_concat", "multimodal_gated", "multimodal_cross_attn"])
    parser.add_argument("--fold", type=str, default="fold_1", help="Fold to train or 'all'")
    parser.add_argument("--epochs", type=int, default=exp_config.num_epochs)
    parser.add_argument("--batch_size", type=int, default=exp_config.batch_size)
    parser.add_argument("--lr", type=float, default=exp_config.learning_rate)
    parser.add_argument("--smoke_test", action="store_true", help="Quick 1-epoch test")
    args = parser.parse_args()

    set_seed(exp_config.seed)
    device = exp_config.device
    print(f"Using device: {device}")

    # Load data and splits
    df_all = pd.read_csv(paths.aligned_dataset_csv)
    with open(paths.data_split_manifest_json, "r") as f:
        splits = json.load(f)["folds"]

    epochs = 1 if args.smoke_test else args.epochs
    folds_to_run = [args.fold] if args.fold != "all" else list(splits.keys())

    for f_name in folds_to_run:
        print(f"\n--- Starting {f_name} | {args.variant} (epochs={epochs}) ---")
        res = train_single_fold(
            fold_name=f_name,
            fold_dict=splits[f_name],
            df_all=df_all,
            variant=args.variant,
            num_epochs=epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=3 if args.smoke_test else exp_config.early_stopping_patience,
            device=device,
            verbose=True,
        )
        print(f"Finished {f_name}: test balanced acc = {res['test_metrics']['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
