"""
Leakage-Free Grouped Data Splitting Module.
Implements Phase 4:
- Grouped cross-validation on experimental units (Box ID)
- Zero box overlap assertion between Train, Validation, and Test partitions
- Treatment stratification
- Saving of data_splits_manifest.json
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import paths, treatments_meta, exp_config


def generate_grouped_splits(n_splits: int = 5, seed: int = 42) -> dict:
    print("=" * 80)
    print("GENERATING LEAKAGE-FREE GROUPED CROSS-VALIDATION SPLITS")
    print("=" * 80)
    
    if not paths.aligned_dataset_csv.exists():
        raise FileNotFoundError(f"Aligned dataset manifest not found at {paths.aligned_dataset_csv}. Run alignment.py first.")
        
    df = pd.read_csv(paths.aligned_dataset_csv)
    
    # 27 unique sugarbeet boxes
    boxes = sorted(df["box_id"].unique())
    print(f"Total experimental units to split: {len(boxes)} boxes.")
    
    # Extract box-level treatment label for stratification
    treatments = treatments_meta.treatments
    box_records = []
    for b in boxes:
        t = treatments[b]
        # High-level stratification label: Water treatment + Nitrogen
        strat_label = f"{t['water']}_{t['n']}"
        box_records.append({
            "box_id": b,
            "desc": t["desc"],
            "water": t["water"],
            "strat_label": strat_label
        })
    df_boxes = pd.DataFrame(box_records)
    
    # Use StratifiedGroupKFold at the box level
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # We split the 27 boxes
    splits = {}
    
    # Use dummy X and strat_label as y, box_id as groups
    box_splits = list(sgkf.split(df_boxes, df_boxes["strat_label"], groups=df_boxes["box_id"]))
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(box_splits):
        test_boxes = df_boxes.iloc[test_idx]["box_id"].tolist()
        train_val_boxes = df_boxes.iloc[train_val_idx]["box_id"].tolist()
        
        # Partition train_val_boxes into Train and Val (hold out ~3-4 boxes for validation)
        # Using reproducible sub-split
        rng = np.random.RandomState(seed + fold_idx)
        shuffled_train_val = rng.permutation(train_val_boxes)
        n_val = max(3, len(shuffled_train_val) // 5)
        val_boxes = sorted(shuffled_train_val[:n_val].tolist())
        train_boxes = sorted(shuffled_train_val[n_val:].tolist())
        test_boxes = sorted(test_boxes)
        
        # ----------------------------------------------------------------------
        # MANDATORY LEAKAGE DEFENSE ASSERTIONS
        # ----------------------------------------------------------------------
        set_train = set(train_boxes)
        set_val = set(val_boxes)
        set_test = set(test_boxes)
        
        assert len(set_train & set_test) == 0, f"LEAKAGE DETECTED: Train and Test share boxes: {set_train & set_test}"
        assert len(set_train & set_val) == 0, f"LEAKAGE DETECTED: Train and Val share boxes: {set_train & set_val}"
        assert len(set_val & set_test) == 0, f"LEAKAGE DETECTED: Val and Test share boxes: {set_val & set_test}"
        assert len(set_train | set_val | set_test) == len(boxes), "Not all boxes accounted for!"
        
        # Get sample IDs
        train_samples = df[df["box_id"].isin(train_boxes)]["sample_id"].tolist()
        val_samples = df[df["box_id"].isin(val_boxes)]["sample_id"].tolist()
        test_samples = df[df["box_id"].isin(test_boxes)]["sample_id"].tolist()
        
        # Drought distribution in test set
        test_drought_count = df[df["box_id"].isin(test_boxes)]["target_drought_stress"].sum()
        test_total = len(test_samples)
        
        print(f"Fold {fold_idx + 1}/{n_splits}:")
        print(f"  Train boxes ({len(train_boxes)}): {train_boxes} -> {len(train_samples)} samples")
        print(f"  Val boxes   ({len(val_boxes)}): {val_boxes} -> {len(val_samples)} samples")
        print(f"  Test boxes  ({len(test_boxes)}): {test_boxes} -> {len(test_samples)} samples ({test_drought_count}/{test_total} drought instances)")
        
        splits[f"fold_{fold_idx + 1}"] = {
            "train_boxes": train_boxes,
            "val_boxes": val_boxes,
            "test_boxes": test_boxes,
            "train_sample_ids": train_samples,
            "val_sample_ids": val_samples,
            "test_sample_ids": test_samples,
            "n_train": len(train_samples),
            "n_val": len(val_samples),
            "n_test": len(test_samples),
        }

    manifest = {
        "dataset_name": "ETH Zurich Eschikon Plant Stress Phenotyping Dataset (2018)",
        "split_strategy": f"Grouped Stratified {n_splits}-Fold Cross-Validation by Box ID",
        "random_seed": seed,
        "total_boxes": len(boxes),
        "total_samples": len(df),
        "leakage_verification": "PASSED (Strict zero-box overlap between all partitions)",
        "folds": splits
    }
    
    with open(paths.data_split_manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nSaved split manifest to: {paths.data_split_manifest_json}")
    return manifest


if __name__ == "__main__":
    generate_grouped_splits(n_splits=exp_config.num_folds, seed=exp_config.seed)
