"""
Environmental Telemetry Sensor Encoder.
Encodes continuous greenhouse atmospheric variables (temperature, humidity, VPD).
"""

import torch
import torch.nn as nn


class EnvironmentalEncoder(nn.Module):
    """Multi-Layer Perceptron (MLP) for continuous environmental sensor telemetry.

    Transforms 10-dimensional causal climate aggregates (mean, std, min, max, VPD)
    into a latent environmental context embedding.
    """

    def __init__(
        self,
        in_features: int = 10,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 10)
        return self.net(x)  # (B, embedding_dim)


if __name__ == "__main__":
    dummy_env = torch.randn(4, 10)
    enc = EnvironmentalEncoder(embedding_dim=32)
    out = enc(dummy_env)
    print("EnvironmentalEncoder forward test:")
    print("  Input shape: ", dummy_env.shape)
    print("  Output shape:", out.shape)
    assert out.shape == (4, 32)
