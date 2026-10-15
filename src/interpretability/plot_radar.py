"""
Generates publication-quality radar / spider chart comparing unimodal vs multimodal architectures.
"""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import paths

def plot_radar_chart():
    csv_path = paths.tables_dir / "benchmark_results.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)

    # Categories to plot
    metrics = ["Accuracy", "Balanced Acc", "F1-Score", "ROC-AUC", "Precision", "Recall"]
    
    # Extract mean values
    model_data = {}
    for _, row in df.iterrows():
        m_name = row["Model Architecture"]
        vals = []
        for m in metrics:
            col_name = f"{m} (Mean±SD)"
            val_str = str(row[col_name]).split("±")[0].strip()
            vals.append(float(val_str))
        model_data[m_name] = vals

    # Number of variables
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Complete loop

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True), dpi=300)

    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], metrics, color="grey", size=10, fontweight="bold")

    # Set y-limits
    ax.set_rlabel_position(0)
    plt.yticks([0.4, 0.6, 0.8, 1.0], ["0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0.25, 1.02)

    colors = {
        "RGB Canopy Only (Unimodal)": "#2b5c8f",
        "Environmental Telemetry (Unimodal)": "#e26d5c",
        "Diagnostic Soil/SPAD (Unimodal)": "#38b000",
        "Concat Fusion (Multimodal)": "#f77f00",
        "Gated Fusion (Proposed Multimodal)": "#7209b7",
        "Cross-Attention (Multimodal)": "#0077b6",
    }

    # Plot each model
    for model_name, values in model_data.items():
        v = values + values[:1]
        color = colors.get(model_name, "#333333")
        is_highlight = "Proposed" in model_name or "Cross" in model_name
        lw = 2.4 if is_highlight else 1.4
        ls = "-" if "Multimodal" in model_name else "--"
        ax.plot(angles, v, linewidth=lw, linestyle=ls, label=model_name, color=color)
        if is_highlight:
            ax.fill(angles, v, color=color, alpha=0.1)

    plt.title("Multimodal vs Unimodal Cross-Validation Performance Radar", size=13, fontweight="bold", y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8.5, frameon=True)
    plt.tight_layout()

    out_path = paths.figures_dir / "fig6_multimodal_performance_radar.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved radar chart to {out_path}")

if __name__ == "__main__":
    plot_radar_chart()
