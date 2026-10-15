# Multimodal Learning for Early Plant Stress Detection

An academic-grade, reproducible research implementation investigating whether multimodal deep learning (canopy visual imagery + atmospheric environmental telemetry + in-situ diagnostic soil/chlorophyll phenotyping) improves early plant stress detection compared with individual unimodal baselines.

Based on the official **ETH Zurich Eschikon Plant Stress Phenotyping Dataset (2018)**.

---

## 🔬 Key Research Findings

1. **Multimodal Fusion vs Unimodal Visual Baseline:**
   - **RGB Canopy Visual Baseline:** Accuracy $93.3\% \pm 2.3\%$, Balanced Acc $92.1\% \pm 4.5\%$, **Macro-F1 $80.2\% \pm 7.0\%$**, ROC-AUC $0.974 \pm 0.014$, PR-AUC $0.866 \pm 0.109$.
   - **Gated Multimodal Fusion (Proposed):** Accuracy $97.4\% \pm 1.6\%$, Balanced Acc $96.4\% \pm 2.8\%$, **Macro-F1 $92.1\% \pm 3.7\%$**, **ROC-AUC $0.995 \pm 0.005$**, **PR-AUC $0.981 \pm 0.018$**.
   - **Cross-Attention Multimodal Fusion:** Accuracy **$97.7\% \pm 1.5\%$**, Balanced Acc **$96.8\% \pm 2.4\%$**, **Macro-F1 $93.4\% \pm 1.5\%$**, ROC-AUC $0.994 \pm 0.007$, PR-AUC $0.979 \pm 0.017$.
   - **Improvement:** **+12.0 to +13.2 percentage point gain in Macro-F1**, with standard deviation reduced from $\pm 7.0\%$ to $\pm 1.5\%$.

2. **Early Stress Detection Timeline:**
   - **P1 (Germination/Baseline, Days 1–14):** Multimodal models achieve $100\%$ specificity and suppress false stress probability on healthy seedlings to **$0.032$** (vs $0.180$ for visual-only).
   - **P2 (Early Vegetative Drought Onset, Days 18–32):** Multimodal models detect transitioning drought boxes while maintaining zero false alarms on the 132 healthy control boxes ($100\%$ specificity).
   - **P3 (Canopy Expansion & Established Drought, Days 36–50):** Gated Multimodal achieves **$97.6\%$ stress recall and $94.7\%$ specificity** (vs $90.2\%$ recall and $83.0\%$ specificity for RGB visual-only).
   - **P4 (Severe Desiccation, Days 54–71):** Cross-Attention achieves **$100\%$ accuracy, $100\%$ recall, and $100\%$ specificity** on unseen boxes.

3. **Interpretability:**
   - **Grad-CAM Canopy Saliency:** Confirms network visual attention is grounded directly in wilting leaf margins, drooping petioles, and necrotic tips, ignoring background greenhouse fixtures.
   - **Feature Importance:** Permutation attribution identifies relative soil moisture ratio ($\Delta\text{AUC} = 0.0161$), 48h maximum relative humidity ($\Delta\text{AUC} = 0.0012$), and 48h mean relative humidity ($\Delta\text{AUC} = 0.0007$) as key non-visual drivers.

---

## 📊 Summary Benchmark Table

*5-Fold Stratified GroupKFold cross-validation grouped strictly by physical Box ID ($N=27$ sugar beet boxes). Target: Dynamic Physiological Drought Desiccation ($W_{box} < 12.0\text{ kg}$).*

| Model Architecture | Modality Type | Accuracy (Mean±SD) | Balanced Acc (Mean±SD) | F1-Score (Mean±SD) | ROC-AUC (Mean±SD) | PR-AUC (Mean±SD) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **RGB Canopy Only** | Unimodal Visual | $0.933 \pm 0.023$ | $0.921 \pm 0.045$ | $0.802 \pm 0.070$ | $0.974 \pm 0.014$ | $0.866 \pm 0.109$ |
| **Environmental Telemetry** | Unimodal Sensor | $0.716 \pm 0.081$ | $0.655 \pm 0.113$ | $0.293 \pm 0.169$ | $0.768 \pm 0.044$ | $0.321 \pm 0.144$ |
| **Diagnostic Soil/SPAD** | Unimodal Tabular | $0.913 \pm 0.048$ | $0.883 \pm 0.138$ | $0.716 \pm 0.200$ | $0.983 \pm 0.028$ | $0.888 \pm 0.204$ |
| **Concat Fusion** | Multimodal | $0.972 \pm 0.014$ | $0.963 \pm 0.027$ | $0.915 \pm 0.029$ | $0.995 \pm 0.005$ | $0.979 \pm 0.015$ |
| **Gated Fusion (Proposed)** | Multimodal | **$0.974 \pm 0.016$** | **$0.964 \pm 0.028$** | **$0.921 \pm 0.037$** | **$0.995 \pm 0.005$** | **$0.981 \pm 0.018$** |
| **Cross-Attention Fusion** | Multimodal | **$0.977 \pm 0.015$** | **$0.968 \pm 0.024$** | **$0.934 \pm 0.015$** | $0.994 \pm 0.007$ | $0.979 \pm 0.017$ |

---

## 📁 Repository & File Organization

```
f:\research paper\
├── plant_stress_dataset/            # Official dataset files & extracted imagery
│   ├── downloads/                   # CSVs, PDF documentation, calibration.zip
│   ├── extracted/
│   │   ├── calibration/             # Camera calibration intrinsics & extrinsics
│   │   └── images/                  # 2,194 plant canopy images across 17 folders
│   └── manifest.json                # Cryptographic checksums and download statuses
├── src/
│   ├── config.py                    # Global paths, metadata, and hyperparameters
│   ├── data/
│   │   ├── audit.py                 # Dataset audit & exploratory statistical checks
│   │   ├── alignment.py             # Spatio-temporal multimodal alignment (432 samples)
│   │   ├── splits.py                # 5-Fold Stratified GroupKFold generator by Box ID
│   │   ├── dataset.py               # PyTorch MultimodalPlantDataset & dataloaders
│   │   └── feature_cache.py         # Visual feature extractor & fast RAM caching
│   ├── models/
│   │   ├── image_encoder.py         # Pretrained ResNet-18 visual encoder (128-d)
│   │   ├── sensor_encoder.py        # Environmental telemetry MLP (32-d)
│   │   ├── spad_encoder.py          # Diagnostic soil/SPAD MLP (16-d)
│   │   └── multimodal_fusion.py     # Concat, Gated Fusion, Cross-Attention architectures
│   ├── training/
│   │   ├── evaluate.py              # Metric calculation (BalAcc, F1, ROC-AUC, PR-AUC)
│   │   ├── train.py                 # Single-fold training loop with early stopping
│   │   └── early_detection.py       # Longitudinal period evaluation (P1 to P4)
│   ├── interpretability/
│   │   ├── gradcam.py               # Grad-CAM visual canopy heatmaps
│   │   ├── feature_attribution.py   # Permutation feature importance
│   │   └── plot_radar.py            # Multimodal performance radar visualization
│   └── experiments/
│       └── run_benchmarks.py        # 5-fold cross-validation master benchmark runner
├── checkpoints/                     # Serialized best model weights (.pt)
├── outputs/
│   ├── aligned_multimodal_dataset.csv # Leakage-free aligned multimodal tabular dataset
│   ├── data_splits_manifest.json    # Strict 5-fold Box-grouped splits manifest
│   ├── dataset_audit_report.md      # Comprehensive data audit report
│   ├── label_definition_report.md   # Dynamic drought target formulation & defense
│   ├── research_results_report.md   # Full academic paper results & discussion report
│   ├── tables/
│   │   ├── benchmark_results.csv    # Cross-validation results table
│   │   ├── benchmark_results.md     # Formatted markdown table
│   │   ├── benchmark_results.tex    # LaTeX publication table
│   │   ├── early_detection_comparison.csv
│   │   └── feature_importance.csv   # Permutation importance rankings
│   └── figures/
│       ├── fig1_environmental_timeseries.png
│       ├── fig2_box_weights_drought_progression.png
│       ├── fig3_spad_nitrogen_progression.png
│       ├── fig4_yield_biomass_by_treatment.png
│       ├── fig5_roc_pr_curves_comparison.png
│       ├── fig6_multimodal_performance_radar.png
│       ├── fig7_confusion_matrices.png
│       ├── fig8_early_detection_timeline.png
│       ├── fig9_feature_importance.png
│       └── fig10_gradcam_visual_explanations.png
└── README.md
```

---

## 🚀 How to Reproduce All Results

### 1. Requirements & Setup
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn pillow tabulate
```

### 2. Run Data Alignment & Leakage-Free Splitting
```bash
python src/data/alignment.py
python src/data/splits.py
```

### 3. Run Precomputed Visual Feature Caching
```bash
python src/data/feature_cache.py
```

### 4. Execute Full 5-Fold Benchmark Suite
```bash
python src/experiments/run_benchmarks.py --epochs 15 --patience 5
```
This trains all 6 architectures across all 5 folds, generates summary tables (`benchmark_results.csv`, `benchmark_results.tex`), ROC/PR curves, confusion matrices, early detection timelines, and Grad-CAM saliency maps.

### 5. Run Interpretability & Feature Attribution
```bash
python src/interpretability/feature_attribution.py
python src/interpretability/plot_radar.py
```

---

## 📄 Academic Research Paper Report
The complete research report with statistical analysis, experimental design, and limitations is located at:
[`outputs/research_results_report.md`](file:///f:/research%20paper/outputs/research_results_report.md)
