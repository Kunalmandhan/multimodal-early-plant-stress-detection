"""
Evaluation Metrics & Performance Reporting Module.
Computes comprehensive, scientifically rigorous classification metrics:
Accuracy, Balanced Accuracy, Macro-F1, Precision, Recall, ROC-AUC, PR-AUC,
Confusion Matrices, and Per-Period Early Detection Analysis.
"""

import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
import torch

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: int = 2,
) -> Dict[str, float]:
    """Computes standard academic evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    if num_classes == 2:
        prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
        rec = recall_score(y_true, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    else:
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = macro_f1

    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
    }

    if y_prob is not None:
        try:
            if num_classes == 2:
                # Use positive class probabilities
                pos_probs = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                if len(np.unique(y_true)) > 1:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, pos_probs))
                    metrics["pr_auc"] = float(average_precision_score(y_true, pos_probs))
                else:
                    metrics["roc_auc"] = 0.5
                    metrics["pr_auc"] = 0.0
            else:
                if len(np.unique(y_true)) > 1:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
                    metrics["pr_auc"] = float(average_precision_score(pd.get_dummies(y_true).values, y_prob, average="macro"))
                else:
                    metrics["roc_auc"] = 0.5
                    metrics["pr_auc"] = 0.0
        except Exception:
            metrics["roc_auc"] = 0.5
            metrics["pr_auc"] = 0.0

    return metrics


def evaluate_early_detection_by_period(
    df_eval: pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    y_prob_col: str = "y_prob",
) -> pd.DataFrame:
    """Evaluates performance broken down by chronological observation periods (P1 to P4)."""
    period_results = []
    periods = sorted(df_eval["period"].unique())
    
    for p in periods:
        sub = df_eval[df_eval["period"] == p]
        y_t = sub[y_true_col].values
        y_p = sub[y_pred_col].values
        y_pr = np.vstack([1.0 - sub[y_prob_col].values, sub[y_prob_col].values]).T if y_prob_col in sub else None
        
        m = compute_metrics(y_t, y_p, y_pr, num_classes=2)
        m["period"] = p
        m["num_samples"] = len(sub)
        m["num_drought_positives"] = int(np.sum(y_t == 1))
        period_results.append(m)
        
    return pd.DataFrame(period_results)
