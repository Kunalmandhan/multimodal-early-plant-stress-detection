"""
Image Encoder Module using Pretrained ResNet-18 Backbone.
Extracts visual feature embeddings from plant canopy RGB images.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    """Convolutional image encoder leveraging a pretrained ResNet-18 backbone

    to project high-resolution canopy imagery into a compact embedding space.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = True,
        fine_tune: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Load torchvision ResNet-18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet18(weights=weights)
        
        # Extract feature backbone (everything up to global average pooling)
        self.backbone = nn.Sequential(*list(base_resnet.children())[:-1])
        in_features = base_resnet.fc.in_features  # 512 for ResNet-18
        
        # Projection head to latent embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        # Freeze early convolutional layers if not full fine-tuning
        if not fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            # Freeze only early layers (conv1, bn1, layer1) to preserve low-level edge features
            for name, child in self.backbone.named_children():
                if name in ["0", "1", "2", "3", "4"]:
                    for param in child.parameters():
                        param.requires_grad = False

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if image_features is not None:
            features = image_features
        elif x is not None:
            features = self.backbone(x)  # (B, 512, 1, 1)
            features = torch.flatten(features, 1)  # (B, 512)
        else:
            raise ValueError("Either x or image_features must be provided to ImageEncoder")

        embedding = self.projection(features)  # (B, embedding_dim)
        return embedding


if __name__ == "__main__":
    dummy_img = torch.randn(2, 3, 224, 224)
    encoder = ImageEncoder(embedding_dim=128)
    out = encoder(dummy_img)
    print("ImageEncoder forward test:")
    print("  Input shape: ", dummy_img.shape)
    print("  Output shape:", out.shape)
    assert out.shape == (2, 128)
