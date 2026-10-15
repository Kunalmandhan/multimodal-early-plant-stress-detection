"""
Multimodal Deep Learning Architecture & Fusion Mechanisms.
Implements:
1. Concat Fusion: Feature Concatenation + MLP head
2. Gated Fusion: Modality-specific gating mechanisms
3. Cross-Attention Fusion: Visual tokens attended with environmental/diagnostic context
Supports strict modality ablation (RGB-only, Env-only, Diag-only, Multimodal).
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.models.image_encoder import ImageEncoder
from src.models.sensor_encoder import EnvironmentalEncoder
from src.models.spad_encoder import DiagnosticEncoder


class GatedMultimodalUnit(nn.Module):
    """Learns a sigmoid gating weight to dynamically scale modality contributions."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.proj(x))
        g = self.gate(x)
        return g * h


class MultimodalPlantStressModel(nn.Module):
    """End-to-End Multimodal Deep Learning System for Early Plant Stress Detection.

    Integrates visual canopy representations, atmospheric environmental context,
    and in-situ diagnostic phenotyping metrics.
    """

    def __init__(
        self,
        num_classes: int = 2,
        fusion_type: str = "gated",  # 'concat', 'gated', 'cross_attention'
        use_image: bool = True,
        use_env: bool = True,
        use_diag: bool = True,
        image_embedding_dim: int = 128,
        env_embedding_dim: int = 32,
        diag_embedding_dim: int = 16,
        fusion_dim: int = 64,
        dropout: float = 0.3,
        fine_tune_backbone: bool = True,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.use_image = use_image
        self.use_env = use_env
        self.use_diag = use_diag
        self.num_classes = num_classes

        assert any([use_image, use_env, use_diag]), "At least one modality must be enabled!"

        # 1. Modality Encoders
        if self.use_image:
            self.image_encoder = ImageEncoder(
                embedding_dim=image_embedding_dim,
                pretrained=True,
                fine_tune=fine_tune_backbone,
                dropout=dropout
            )
        else:
            self.image_encoder = None

        if self.use_env:
            self.env_encoder = EnvironmentalEncoder(
                in_features=10,
                embedding_dim=env_embedding_dim,
                hidden_dim=64,
                dropout=dropout
            )
        else:
            self.env_encoder = None

        if self.use_diag:
            self.diag_encoder = DiagnosticEncoder(
                in_features=4,
                embedding_dim=diag_embedding_dim,
                hidden_dim=32,
                dropout=dropout
            )
        else:
            self.diag_encoder = None

        # 2. Fusion Mechanisms
        total_embed_dim = 0
        if self.use_image:
            total_embed_dim += image_embedding_dim
        if self.use_env:
            total_embed_dim += env_embedding_dim
        if self.use_diag:
            total_embed_dim += diag_embedding_dim

        if self.fusion_type == "gated":
            self.gated_units = nn.ModuleDict()
            if self.use_image:
                self.gated_units["image"] = GatedMultimodalUnit(image_embedding_dim, fusion_dim)
            if self.use_env:
                self.gated_units["env"] = GatedMultimodalUnit(env_embedding_dim, fusion_dim)
            if self.use_diag:
                self.gated_units["diag"] = GatedMultimodalUnit(diag_embedding_dim, fusion_dim)
                
            num_active = sum([self.use_image, self.use_env, self.use_diag])
            classifier_in_dim = fusion_dim * num_active

        elif self.fusion_type == "cross_attention":
            # Multihead attention: visual tokens attend to structured tokens
            self.common_proj = nn.ModuleDict()
            if self.use_image:
                self.common_proj["image"] = nn.Linear(image_embedding_dim, fusion_dim)
            if self.use_env:
                self.common_proj["env"] = nn.Linear(env_embedding_dim, fusion_dim)
            if self.use_diag:
                self.common_proj["diag"] = nn.Linear(diag_embedding_dim, fusion_dim)

            self.cross_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=2, batch_first=True)
            self.norm = nn.LayerNorm(fusion_dim)
            classifier_in_dim = fusion_dim

        else:  # 'concat'
            classifier_in_dim = total_embed_dim

        # 3. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes)
        )

    def extract_modality_embeddings(
        self,
        image: Optional[torch.Tensor] = None,
        env_features: Optional[torch.Tensor] = None,
        diag_features: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        embeddings = {}
        if self.use_image and (image is not None or image_features is not None):
            embeddings["image"] = self.image_encoder(x=image, image_features=image_features)
        if self.use_env and env_features is not None:
            embeddings["env"] = self.env_encoder(env_features)
        if self.use_diag and diag_features is not None:
            embeddings["diag"] = self.diag_encoder(diag_features)
        return embeddings

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        env_features: Optional[torch.Tensor] = None,
        diag_features: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Returns:
            logits: (B, num_classes)
            modality_embeddings: Dict of individual modality representations
        """
        mod_embs = self.extract_modality_embeddings(image, env_features, diag_features, image_features=image_features)

        if self.fusion_type == "gated":
            fused_parts = []
            if "image" in mod_embs:
                fused_parts.append(self.gated_units["image"](mod_embs["image"]))
            if "env" in mod_embs:
                fused_parts.append(self.gated_units["env"](mod_embs["env"]))
            if "diag" in mod_embs:
                fused_parts.append(self.gated_units["diag"](mod_embs["diag"]))
            fused = torch.cat(fused_parts, dim=1)

        elif self.fusion_type == "cross_attention":
            tokens = []
            if "image" in mod_embs:
                tokens.append(self.common_proj["image"](mod_embs["image"]).unsqueeze(1))
            if "env" in mod_embs:
                tokens.append(self.common_proj["env"](mod_embs["env"]).unsqueeze(1))
            if "diag" in mod_embs:
                tokens.append(self.common_proj["diag"](mod_embs["diag"]).unsqueeze(1))

            seq = torch.cat(tokens, dim=1)  # (B, num_modalities, fusion_dim)
            attn_out, _ = self.cross_attn(seq, seq, seq)
            fused = self.norm(seq + attn_out).mean(dim=1)  # Average pooling across tokens

        else:  # 'concat'
            parts = []
            if "image" in mod_embs:
                parts.append(mod_embs["image"])
            if "env" in mod_embs:
                parts.append(mod_embs["env"])
            if "diag" in mod_embs:
                parts.append(mod_embs["diag"])
            fused = torch.cat(parts, dim=1)

        logits = self.classifier(fused)
        return logits, mod_embs


if __name__ == "__main__":
    B = 4
    img = torch.randn(B, 3, 224, 224)
    env = torch.randn(B, 10)
    diag = torch.randn(B, 4)

    for f_type in ["concat", "gated", "cross_attention"]:
        model = MultimodalPlantStressModel(fusion_type=f_type, num_classes=2)
        logits, embs = model(img, env, diag)
        print(f"Fusion '{f_type}' test: logits={logits.shape}")
        assert logits.shape == (B, 2)

    # Test RGB-only ablation
    rgb_only = MultimodalPlantStressModel(use_image=True, use_env=False, use_diag=False)
    l_rgb, _ = rgb_only(image=img)
    print("RGB-only test: logits=", l_rgb.shape)
    assert l_rgb.shape == (B, 2)

    # Test Env-only ablation
    env_only = MultimodalPlantStressModel(use_image=False, use_env=True, use_diag=False)
    l_env, _ = env_only(env_features=env)
    print("Env-only test: logits=", l_env.shape)
    assert l_env.shape == (B, 2)
