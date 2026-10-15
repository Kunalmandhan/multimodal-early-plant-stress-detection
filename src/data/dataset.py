"""
PyTorch Dataset and DataLoader Module for Multimodal Plant Stress Detection.
Handles image transformations, fit-on-train feature scaling, missing SPAD imputation,
and multimodal tensor batching.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import paths, exp_config

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Feature columns for environmental telemetry
ENV_FEATURE_COLS = [
    "env_temp_mean_48h",
    "env_temp_std_48h",
    "env_temp_min_48h",
    "env_temp_max_48h",
    "env_rh_mean_48h",
    "env_rh_std_48h",
    "env_rh_min_48h",
    "env_rh_max_48h",
    "env_vpd_mean_48h",
    "env_vpd_max_48h",
]

# Diagnostic tabular columns
DIAG_FEATURE_COLS = [
    "relative_weight_ratio",
    "box_weight_kg",
    "spad_value",
    "spad_observed",
]


class FeatureScaler:
    """Computes mean and std strictly on training data and transforms validation/test sets."""

    def __init__(self):
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.spad_default: float = 35.0  # Median SPAD baseline

    def fit(self, df_train: pd.DataFrame):
        for col in ENV_FEATURE_COLS:
            self.means[col] = float(df_train[col].mean())
            self.stds[col] = float(df_train[col].std() if df_train[col].std() > 1e-6 else 1.0)
            
        for col in ["relative_weight_ratio", "box_weight_kg"]:
            self.means[col] = float(df_train[col].mean())
            self.stds[col] = float(df_train[col].std() if df_train[col].std() > 1e-6 else 1.0)
            
        # For SPAD, fit mean on observed values only
        valid_spad = df_train[df_train["spad_observed"] == 1]["spad_value"]
        if len(valid_spad) > 0:
            self.spad_default = float(valid_spad.mean())
            self.means["spad_value"] = self.spad_default
            self.stds["spad_value"] = float(valid_spad.std() if valid_spad.std() > 1e-6 else 1.0)
        else:
            self.means["spad_value"] = 35.0
            self.stds["spad_value"] = 5.0

    def transform_env(self, row: pd.Series) -> np.ndarray:
        scaled = []
        for col in ENV_FEATURE_COLS:
            val = float(row[col])
            scaled_val = (val - self.means[col]) / self.stds[col]
            scaled.append(scaled_val)
        return np.array(scaled, dtype=np.float32)

    def transform_diag(self, row: pd.Series) -> np.ndarray:
        # Moisture ratio
        ratio = (float(row["relative_weight_ratio"]) - self.means["relative_weight_ratio"]) / self.stds["relative_weight_ratio"]
        # Absolute weight
        weight = (float(row["box_weight_kg"]) - self.means["box_weight_kg"]) / self.stds["box_weight_kg"]
        
        # SPAD with explicit missingness mask indicator
        spad_obs = float(row["spad_observed"])
        if spad_obs == 1.0 and not pd.isna(row["spad_value"]):
            spad_scaled = (float(row["spad_value"]) - self.means["spad_value"]) / self.stds["spad_value"]
        else:
            # Impute with fitted training mean
            spad_scaled = 0.0
            
        return np.array([ratio, weight, spad_scaled, spad_obs], dtype=np.float32)


def get_image_transforms(is_train: bool = False, image_size: int = 224) -> transforms.Compose:
    """Builds torchvision transformations.

    Data augmentation (flips) is applied strictly during training.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class MultimodalPlantDataset(Dataset):
    """PyTorch Dataset loading synchronized visual images, environmental telemetry,

    and in-situ diagnostic phenotyping measurements.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        scaler: FeatureScaler,
        cached_features: Optional[Dict[str, torch.Tensor]] = None,
        is_train: bool = False,
        image_size: int = 224,
        image_modality: str = "rgb",  # 'rgb', 'ir1', 'ir2', 'multispec'
    ):
        self.df = df.reset_index(drop=True)
        self.scaler = scaler
        self.cached_features = cached_features
        self.is_train = is_train
        self.transform = get_image_transforms(is_train=is_train, image_size=image_size)
        self.image_modality = image_modality
        self.project_root = paths.project_root

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        
        # 1. Load Image or Cached Feature
        if self.cached_features is not None and sample_id in self.cached_features:
            img_tensor = torch.empty(0)  # placeholder
            img_feat = self.cached_features[sample_id]
        else:
            img_feat = torch.empty(0)
            if self.image_modality == "rgb":
                img_rel_path = row["rgb_path"]
                img_path = self.project_root / img_rel_path
                with Image.open(img_path) as im:
                    img_rgb = im.convert("RGB")
                    img_tensor = self.transform(img_rgb)
            else:
                path_col = f"{self.image_modality}_path"
                img_rel_path = row[path_col]
                img_path = self.project_root / img_rel_path
                with Image.open(img_path) as im:
                    im_gray = im.convert("L")
                    im_3ch = Image.merge("RGB", (im_gray, im_gray, im_gray))
                    img_tensor = self.transform(im_3ch)

        # 2. Environmental Telemetry Features (10-dim vector)
        env_vec = self.scaler.transform_env(row)
        env_tensor = torch.tensor(env_vec, dtype=torch.float32)

        # 3. Diagnostic & SPAD Features (4-dim vector: rel_moisture, weight, spad, spad_mask)
        diag_vec = self.scaler.transform_diag(row)
        diag_tensor = torch.tensor(diag_vec, dtype=torch.float32)

        # 4. Targets
        target_drought = torch.tensor(row["target_drought_stress"], dtype=torch.long)
        target_multiclass = torch.tensor(row["target_multiclass"], dtype=torch.long)

        return {
            "sample_id": sample_id,
            "box_id": int(row["box_id"]),
            "date_str": row["date_str"],
            "period": row["period"],
            "image": img_tensor,
            "image_features": img_feat,
            "has_cached_image": (self.cached_features is not None and sample_id in self.cached_features),
            "env_features": env_tensor,
            "diag_features": diag_tensor,
            "target_drought": target_drought,
            "target_multiclass": target_multiclass,
        }


def create_dataloaders(
    df: pd.DataFrame,
    fold_dict: dict,
    cached_features: Optional[Dict[str, torch.Tensor]] = None,
    batch_size: int = 8,
    image_size: int = 224,
    image_modality: str = "rgb",
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, FeatureScaler]:
    """Builds train, validation, and test dataloaders strictly fitting feature scalers on train set only."""
    train_ids = set(fold_dict["train_sample_ids"])
    val_ids = set(fold_dict["val_sample_ids"])
    test_ids = set(fold_dict["test_sample_ids"])

    df_train = df[df["sample_id"].isin(train_ids)].copy()
    df_val = df[df["sample_id"].isin(val_ids)].copy()
    df_test = df[df["sample_id"].isin(test_ids)].copy()

    # Fit scaler strictly on training split
    scaler = FeatureScaler()
    scaler.fit(df_train)

    train_ds = MultimodalPlantDataset(df_train, scaler, cached_features=cached_features, is_train=True, image_size=image_size, image_modality=image_modality)
    val_ds = MultimodalPlantDataset(df_val, scaler, cached_features=cached_features, is_train=False, image_size=image_size, image_modality=image_modality)
    test_ds = MultimodalPlantDataset(df_test, scaler, cached_features=cached_features, is_train=False, image_size=image_size, image_modality=image_modality)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, scaler


if __name__ == "__main__":
    # Test dataset instantiation
    df_all = pd.read_csv(paths.aligned_dataset_csv)
    with open(paths.data_split_manifest_json, "r") as f:
        splits = json.load(f)["folds"]
    
    t_l, v_l, te_l, sc = create_dataloaders(df_all, splits["fold_1"], batch_size=4)
    batch = next(iter(t_l))
    print(f"Batch test successful:")
    print(f"  Image tensor shape: {batch['image'].shape}")
    print(f"  Env tensor shape:   {batch['env_features'].shape}")
    print(f"  Diag tensor shape:  {batch['diag_features'].shape}")
    print(f"  Drought target:     {batch['target_drought']}")
