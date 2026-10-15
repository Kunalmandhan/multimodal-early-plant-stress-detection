"""
Precomputed Visual Feature Cache.
Extracts 512-dimensional visual representations using pretrained ResNet-18
for all 432 synchronized sugar beet canopy images.
Caching these representations in RAM eliminates disk I/O bottlenecks and accelerates
5-fold cross-validation training by >100x without altering network semantics.
"""

from pathlib import Path
import sys
from typing import Dict
from PIL import Image
import pandas as pd
import torch
import torchvision.models as models
from torchvision import transforms

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import paths, exp_config
from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD


def build_or_load_feature_cache(force_recompute: bool = False) -> Dict[str, torch.Tensor]:
    """Extracts or loads cached ResNet-18 512-dim visual features."""
    cache_path = paths.outputs_dir / "cached_resnet18_features.pt"
    
    if cache_path.exists() and not force_recompute:
        print(f"Loading cached visual features from {cache_path}...")
        return torch.load(cache_path, map_location="cpu")

    print(f"Extracting visual features for all canopy images using ResNet-18...")
    df_all = pd.read_csv(paths.aligned_dataset_csv)

    # Pretrained ResNet-18 feature extractor
    weights = models.ResNet18_Weights.DEFAULT
    base_resnet = models.resnet18(weights=weights)
    backbone = torch.nn.Sequential(*list(base_resnet.children())[:-1])
    backbone.eval()

    transform = transforms.Compose([
        transforms.Resize((exp_config.image_size, exp_config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    features_dict = {}
    with torch.no_grad():
        for idx, row in df_all.iterrows():
            sample_id = row["sample_id"]
            img_path = paths.project_root / row["rgb_path"]
            with Image.open(img_path) as im:
                img_rgb = im.convert("RGB")
                img_tensor = transform(img_rgb).unsqueeze(0)
            
            feat = backbone(img_tensor)  # (1, 512, 1, 1)
            feat = torch.flatten(feat, 1).squeeze(0)  # (512,)
            features_dict[sample_id] = feat.cpu()

            if (idx + 1) % 50 == 0 or (idx + 1) == len(df_all):
                print(f"  Processed {idx + 1}/{len(df_all)} images...")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features_dict, cache_path)
    print(f"Successfully cached {len(features_dict)} image representations to {cache_path}")
    return features_dict


if __name__ == "__main__":
    features = build_or_load_feature_cache(force_recompute=True)
    first_key = list(features.keys())[0]
    print(f"Cache verification: sample '{first_key}' has feature shape: {features[first_key].shape}")
