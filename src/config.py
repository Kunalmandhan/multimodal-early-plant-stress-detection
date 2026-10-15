"""
Configuration module for Multimodal Learning for Early Plant Stress Detection.
Centralizes all paths, hyperparameters, random seeds, and experimental settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import torch


@dataclass
class PathConfig:
    # Root paths
    project_root: Path = Path(__file__).resolve().parent.parent
    dataset_root: Path = project_root / "plant_stress_dataset"
    downloads_dir: Path = dataset_root / "downloads"
    extracted_dir: Path = dataset_root / "extracted"
    images_dir: Path = extracted_dir / "images"
    calibration_dir: Path = extracted_dir / "calibration"
    
    # Specific data files
    readme_pdf: Path = downloads_dir / "README.pdf"
    directory_structure: Path = downloads_dir / "directory-structure.txt"
    box_weights_csv: Path = downloads_dir / "box-weights-kg.csv"
    greenhouse_logs_csv: Path = downloads_dir / "greenhouse-temperature-humidity-logs.csv"
    spad_csv: Path = downloads_dir / "spad-measurements.csv"
    yield_csv: Path = downloads_dir / "yield-weight-grams.csv"
    overview_jpg: Path = downloads_dir / "dataset-overview.jpg"
    
    # Output paths
    outputs_dir: Path = project_root / "outputs"
    figures_dir: Path = outputs_dir / "figures"
    tables_dir: Path = outputs_dir / "tables"
    checkpoints_dir: Path = project_root / "checkpoints"
    
    # Manifest files
    aligned_dataset_csv: Path = outputs_dir / "aligned_multimodal_dataset.csv"
    data_split_manifest_json: Path = outputs_dir / "data_splits_manifest.json"


@dataclass
class TreatmentMetadata:
    """Documented treatments from official ETH Zurich dataset README.pdf"""
    # Box mapping: 1-31
    # Box 31 is CALTag / Color Checker (non-plant calibration target)
    # Boxes 16, 17, 18 are Weed Only (no sugar beet crop planted)
    # Plant boxes are 1-15 and 19-30 (total 27 sugar beet boxes)
    treatments: Dict[int, dict] = field(default_factory=lambda: {
        1:  {"desc": "Control", "soil": 2, "water": "Sufficient", "n": "High", "weed": "None", "has_sugarbeet": True},
        2:  {"desc": "Control", "soil": 2, "water": "Sufficient", "n": "High", "weed": "None", "has_sugarbeet": True},
        3:  {"desc": "Control", "soil": 2, "water": "Sufficient", "n": "High", "weed": "None", "has_sugarbeet": True},
        4:  {"desc": "Med N", "soil": 2, "water": "Sufficient", "n": "Medium", "weed": "None", "has_sugarbeet": True},
        5:  {"desc": "Med N", "soil": 2, "water": "Sufficient", "n": "Medium", "weed": "None", "has_sugarbeet": True},
        6:  {"desc": "Med N", "soil": 2, "water": "Sufficient", "n": "Medium", "weed": "None", "has_sugarbeet": True},
        7:  {"desc": "Med weeds", "soil": 2, "water": "Sufficient", "n": "High", "weed": "Medium", "has_sugarbeet": True},
        8:  {"desc": "Med weeds", "soil": 2, "water": "Sufficient", "n": "High", "weed": "Medium", "has_sugarbeet": True},
        9:  {"desc": "Med weeds", "soil": 2, "water": "Sufficient", "n": "High", "weed": "Medium", "has_sugarbeet": True},
        10: {"desc": "High weeds", "soil": 2, "water": "Sufficient", "n": "High", "weed": "High", "has_sugarbeet": True},
        11: {"desc": "High weeds", "soil": 2, "water": "Sufficient", "n": "High", "weed": "High", "has_sugarbeet": True},
        12: {"desc": "High weeds", "soil": 2, "water": "Sufficient", "n": "High", "weed": "High", "has_sugarbeet": True},
        13: {"desc": "Dry", "soil": 2, "water": "Low", "n": "Medium", "weed": "None", "has_sugarbeet": True},
        14: {"desc": "Dry", "soil": 2, "water": "Low", "n": "Medium", "weed": "None", "has_sugarbeet": True},
        15: {"desc": "Dry", "soil": 2, "water": "Low", "n": "Medium", "weed": "None", "has_sugarbeet": True},
        16: {"desc": "Weed Only Dicot", "soil": 2, "water": "Sufficient", "n": "High", "weed": "High", "has_sugarbeet": False},
        17: {"desc": "Weed Only Monocot", "soil": 2, "water": "Sufficient", "n": "High", "weed": "High", "has_sugarbeet": False},
        18: {"desc": "Weed Only Mixed", "soil": 2, "water": "Sufficient", "n": "High", "weed": "High", "has_sugarbeet": False},
        19: {"desc": "Drying - Med N - High Weed", "soil": 2, "water": "Low", "n": "Medium", "weed": "High", "has_sugarbeet": True},
        20: {"desc": "Drying - Med N - High Weed", "soil": 2, "water": "Low", "n": "Medium", "weed": "High", "has_sugarbeet": True},
        21: {"desc": "Drying - Med N - High Weed", "soil": 2, "water": "Low", "n": "Medium", "weed": "High", "has_sugarbeet": True},
        22: {"desc": "Low N", "soil": 1, "water": "Sufficient", "n": "Low", "weed": "None", "has_sugarbeet": True},
        23: {"desc": "Low N", "soil": 1, "water": "Sufficient", "n": "Low", "weed": "None", "has_sugarbeet": True},
        24: {"desc": "Low N", "soil": 1, "water": "Sufficient", "n": "Low", "weed": "None", "has_sugarbeet": True},
        25: {"desc": "Low N - Med Weed", "soil": 1, "water": "Sufficient", "n": "Low", "weed": "Medium", "has_sugarbeet": True},
        26: {"desc": "Low N - Med Weed", "soil": 1, "water": "Sufficient", "n": "Low", "weed": "Medium", "has_sugarbeet": True},
        27: {"desc": "Low N - Med Weed", "soil": 1, "water": "Sufficient", "n": "Low", "weed": "Medium", "has_sugarbeet": True},
        28: {"desc": "Drying - Low N", "soil": 1, "water": "Low", "n": "Low", "weed": "None", "has_sugarbeet": True},
        29: {"desc": "Drying - Low N", "soil": 1, "water": "Low", "n": "Low", "weed": "None", "has_sugarbeet": True},
        30: {"desc": "Drying - Low N", "soil": 1, "water": "Low", "n": "Low", "weed": "None", "has_sugarbeet": True},
        31: {"desc": "CALTag and Color Checker", "soil": 1, "water": "NA", "n": "NA", "weed": "NA", "has_sugarbeet": False},
    })


@dataclass
class ExperimentConfig:
    # Random seed for reproducibility
    seed: int = 42
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Image parameters
    image_size: int = 224
    image_modalities: List[str] = field(default_factory=lambda: ["rgb"])  # Can expand to ["rgb", "ir1", "multispec"]
    
    # Model architecture parameters
    image_backbone: str = "resnet18"
    image_embedding_dim: int = 128
    env_embedding_dim: int = 32
    spad_embedding_dim: int = 16
    fusion_hidden_dim: int = 64
    dropout: float = 0.3
    
    # Training settings
    batch_size: int = 8
    num_epochs: int = 25
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 7
    num_folds: int = 5
    
    # Environmental sensor aggregation window (hours prior to image observation)
    env_lookback_hours: int = 48


# Default configuration instance
paths = PathConfig()
treatments_meta = TreatmentMetadata()
exp_config = ExperimentConfig()
