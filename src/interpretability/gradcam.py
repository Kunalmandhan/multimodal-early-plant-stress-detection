"""
Interpretability and Visual Explainability Module.
Implements Grad-CAM (Gradient-weighted Class Activation Mapping) for the ResNet-18 visual encoder
to visualize the regions of the sugar beet plant canopy the multimodal model attends to
when detecting drought stress vs healthy control.
"""

from pathlib import Path
import sys
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import paths, exp_config
from src.data.dataset import get_image_transforms, IMAGENET_MEAN, IMAGENET_STD
from src.models.multimodal_fusion import MultimodalPlantStressModel


class GradCAM:
    """Computes Grad-CAM saliency maps for the final convolutional layer of ResNet-18."""

    def __init__(self, model: MultimodalPlantStressModel, target_layer_idx: int = 7):
        self.model = model
        self.model.eval()
        
        # Access the conv layer in ResNet-18 backbone (layer 7 corresponds to layer4 in standard ResNet)
        # children: 0:conv1, 1:bn1, 2:relu, 3:maxpool, 4:layer1, 5:layer2, 6:layer3, 7:layer4
        self.target_layer = list(self.model.image_encoder.backbone.children())[target_layer_idx]
        
        self.gradients = None
        self.activations = None
        
        self.hook_forward = self.target_layer.register_forward_hook(self._save_activation)
        self.hook_backward = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        env_features: Optional[torch.Tensor] = None,
        diag_features: Optional[torch.Tensor] = None,
        target_class: int = 1,
    ) -> np.ndarray:
        """Generates 2D Grad-CAM heatmap normalized to [0, 1]."""
        self.model.zero_grad()
        
        logits, _ = self.model(image_tensor, env_features, diag_features)
        target_score = logits[:, target_class].sum()
        target_score.backward(retain_graph=True)

        # Global average pooling of gradients
        # gradients shape: (B, C, H, W)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight activation channels
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(self.activations, dim=1).squeeze().cpu()
        heatmap = F.relu(heatmap)
        
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
            
        return heatmap.numpy()

    def remove_hooks(self):
        self.hook_forward.remove()
        self.hook_backward.remove()


def overlay_cam_on_image(
    img_pil: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlays Grad-CAM heatmap onto an RGB PIL image."""
    img_resized = img_pil.resize((exp_config.image_size, exp_config.image_size))
    img_np = np.array(img_resized) / 255.0

    # Resize heatmap to match image size
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(
        (exp_config.image_size, exp_config.image_size), resample=Image.BILINEAR
    )
    heatmap_np = np.array(heatmap_img) / 255.0

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored_heatmap = cmap(heatmap_np)[:, :, :3]

    overlay = alpha * colored_heatmap + (1 - alpha) * img_np
    overlay = np.clip(overlay, 0, 1)
    return overlay


def generate_gradcam_visualizations(
    model: MultimodalPlantStressModel,
    df_aligned: pd.DataFrame,
    output_path: Path,
):
    """Generates a multi-panel visual explanation comparison for healthy vs drought sugar beet plants."""
    grad_cam = GradCAM(model)
    transform = get_image_transforms(is_train=False, image_size=exp_config.image_size)

    # Select representative samples:
    # 1. Healthy control sample (Box 1 or 2, P3/P4)
    # 2. Early onset drought sample (Box 13 or 14, P2)
    # 3. Severe drought sample (Box 13 or 14, P4)
    # 4. Low Nitrogen + Drought sample (Box 28 or 29, P4)
    sample_queries = [
        {"desc": "Healthy Control (Box 1, P3)", "filter": (df_aligned["box_id"] == 1) & (df_aligned["period"] == "P3 (Days 36-50)")},
        {"desc": "Early Drought Onset (Box 13, P2)", "filter": (df_aligned["box_id"] == 13) & (df_aligned["period"] == "P2 (Days 18-32)")},
        {"desc": "Severe Drought Desiccation (Box 13, P4)", "filter": (df_aligned["box_id"] == 13) & (df_aligned["period"] == "P4 (Days 54-71)")},
        {"desc": "Combined Nitrogen+Drought (Box 28, P4)", "filter": (df_aligned["box_id"] == 28) & (df_aligned["period"] == "P4 (Days 54-71)")},
    ]

    selected_rows = []
    for q in sample_queries:
        matched = df_aligned[q["filter"]]
        if len(matched) > 0:
            selected_rows.append((q["desc"], matched.iloc[0]))
        else:
            # Fallback to any positive / negative sample
            if "Healthy" in q["desc"]:
                selected_rows.append((q["desc"], df_aligned[df_aligned["target_drought_stress"] == 0].iloc[0]))
            else:
                selected_rows.append((q["desc"], df_aligned[df_aligned["target_drought_stress"] == 1].iloc[0]))

    fig, axes = plt.subplots(len(selected_rows), 3, figsize=(11, 3.2 * len(selected_rows)), dpi=300)

    for idx, (title, row) in enumerate(selected_rows):
        img_path = paths.project_root / row["rgb_path"]
        pil_img = Image.open(img_path).convert("RGB")
        img_tensor = transform(pil_img).unsqueeze(0)

        # Mock / default sensor features for inference
        env_tensor = torch.zeros(1, 10)
        diag_tensor = torch.zeros(1, 4)

        heatmap = grad_cam.generate_heatmap(img_tensor, env_tensor, diag_tensor, target_class=1)
        overlay = overlay_cam_on_image(pil_img, heatmap)

        # 1. Original RGB Canopy
        ax_orig = axes[idx, 0]
        ax_orig.imshow(pil_img.resize((exp_config.image_size, exp_config.image_size)))
        ax_orig.set_title(f"{title}\nOriginal RGB Canopy", fontsize=9, fontweight="bold")
        ax_orig.axis("off")

        # 2. Grad-CAM Saliency Heatmap
        ax_map = axes[idx, 1]
        im_heat = ax_map.imshow(heatmap, cmap="jet")
        ax_map.set_title("Grad-CAM Activation Map\n(Target: Drought Stress)", fontsize=9, fontweight="bold")
        ax_map.axis("off")
        plt.colorbar(im_heat, ax=ax_map, fraction=0.046, pad=0.04)

        # 3. Canopy Saliency Overlay
        ax_over = axes[idx, 2]
        ax_over.imshow(overlay)
        status_str = "DROUGHT" if row["target_drought_stress"] == 1 else "HEALTHY"
        ax_over.set_title(f"Canopy Attention Overlay\n[Ground Truth: {status_str}]", fontsize=9, fontweight="bold")
        ax_over.axis("off")

    grad_cam.remove_hooks()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Grad-CAM visual saliency figure to {output_path}")


if __name__ == "__main__":
    print("Grad-CAM module loaded.")
