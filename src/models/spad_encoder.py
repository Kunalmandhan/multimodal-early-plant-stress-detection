"""
Diagnostic and In-Situ Measurement Encoder.
Encodes soil moisture metrics (relative box weight) and SPAD chlorophyll readings.
"""

import torch
import torch.nn as nn


class DiagnosticEncoder(nn.Module):
    """MLP encoding in-situ plant and soil measurements:

    [relative_weight_ratio, box_weight_kg, spad_value, spad_observed_mask].
    """

    def __init__(
        self,
        in_features: int = 4,
        embedding_dim: int = 16,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4)
        return self.net(x)  # (B, embedding_dim)


if __name__ == "__main__":
    dummy_diag = torch.randn(4, 4)
    enc = DiagnosticEncoder(embedding_dim=16)
    out = enc(dummy_diag)
    print("DiagnosticEncoder forward test:")
    print("  Input shape: ", dummy_diag.shape)
    print("  Output shape:", out.shape)
    assert out.shape == (4, 16)
