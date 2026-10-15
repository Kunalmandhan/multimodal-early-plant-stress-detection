# Label Definition and Leakage-Risk Report
## Research Project: Multimodal Learning for Early Plant Stress Detection

---

### 1. Research Problem Formulation

The objective of this research is to investigate whether fusing multimodal observations—specifically **aerial RGB canopy imagery**, **continuous greenhouse environmental telemetry** (ambient temperature and relative humidity), and **in-situ physiological diagnostics** (SPAD chlorophyll index and soil moisture box weights)—enables earlier and more reliable detection of plant stress compared to unimodal visual baselines.

---

### 2. Definition of Prediction Units and Observation Instances

- **Independent Experimental Unit:** **Experimental Box** ($N = 27$ independent sugarbeet boxes).
  - Controlled factor allocations in the ETH Zurich greenhouse are assigned at the box level.
  - Sugarbeets in the same box share soil volume, water input, nutrient availability, and weed pressure.
- **Multimodal Observation Instance:** A tuple $(b, t)$ representing box $b \in \{1, \dots, 15, 19, \dots, 30\}$ observed at measurement date $t \in \{t_1, \dots, t_{16}\}$.
  - Total valid sugarbeet observations: $27 \text{ boxes} \times 16 \text{ dates} = 432 \text{ multimodal instances}$.
  - Non-crop units (Boxes 16, 17, 18 for weed-only monoculture and Box 31 for calibration target) are explicitly excluded from sugarbeet stress detection.

---

### 3. Proposed Prediction Targets

To adhere to scientific rigor, we formulate two complementary prediction tasks:

#### Task A: Dynamic Drought Stress Detection (Primary Binary Benchmark)
- **Scientific Rationale:** Water input is one of the most critical stressors in agricultural phenotyping. In the ETH Eschikon experiment, water input was either "Sufficient" or "Low" (Drying).
- **Static Treatment vs. Dynamic Physiological State:**
  - *Static Treatment Label:* An indicator of whether the box was assigned to the drought protocol.
  - *Dynamic Physiological Ground Truth:* As verified in our empirical audit of `box-weights-kg.csv`, boxes in the "Low" water group did not experience moisture deficit in January (~15.7 kg baseline, identical to control). Real drought stress commenced in mid-February as the soil desiccated below 12.0 kg, reaching severe drought (<9.0 kg) in March.
  - **Dynamic Stress Label ($Y_{drought}(b, t)$):**
    $$Y_{drought}(b, t) = \begin{cases} 1 & \text{if assigned Low Water and } W_{box}(t) < 12.0\text{ kg (Moisture Deficit)} \\ 0 & \text{if Sufficient Water (Control / Well-watered)} \end{cases}$$
- **Why this avoids false early claims:** A model evaluated at Day 1 (Jan 18) cannot legitimately "detect drought" if the soil moisture is at field capacity. Defining the label dynamically tests whether the multimodal system detects the *onset* of physiological water deficit.

#### Task B: Multiclass Stress Factor Identification
- **Classes:**
  1. `Control`: Sufficient Water, High Nitrogen, No Weeds (Boxes 1, 2, 3)
  2. `Water Stress (Drought)`: Low Water, Medium Nitrogen, No Weeds (Boxes 13, 14, 15)
  3. `Nutrient Stress (Low N)`: Sufficient Water, Low Nitrogen, No Weeds (Boxes 22, 23, 24)
  4. `Biotic Stress (Weed Pressure)`: Sufficient Water, High Nitrogen, Med/High Weeds (Boxes 7, 8, 9, 10, 11, 12)
- **Role:** Evaluates whether multimodal representations can disambiguate between different abiotic and biotic stress manifestations.

---

### 4. Leakage-Risk Analysis & Defense Strategies

| Hazard Category | Specific Risk in Eschikon Dataset | Enforced Mitigation / Defense |
| :--- | :--- | :--- |
| **Box / Subject Leakage** | Images of the same box on different dates share identical camera perspectives, background soil patterns, and box borders. A random image split allows the CNN to memorize box identity. | **Strict Grouped Splitting by Box ID:** All 16 temporal observations of Box $i$ are assigned exclusively to either Train or Validation/Test. Zero box overlap across splits. |
| **Temporal Lookahead Leakage** | Using future sensor readings or harvest outcomes to predict stress at earlier dates. | **Causal Time Windowing:** Environmental telemetry is aggregated strictly over $[t - 48\text{h}, t]$ prior to the image acquisition timestamp. Harvest biomass (`yield-weight-grams.csv`) is strictly an outcome variable, never an input feature. |
| **Non-Crop Target Contamination** | Including Box 31 (CALTag) or Boxes 16–18 (weed-only) would allow the model to cheat by classifying non-beet images. | **Explicit Exclusion:** Boxes 16, 17, 18, and 31 are filtered out prior to training. |
| **Preprocessor Leakage** | Computing normalization scalers (mean, std) or imputation statistics across the whole dataset before splitting. | **Fit-on-Train Only:** All scalers, imputers, and normalizers are fit strictly on the training partition and applied blindly to validation and test partitions. |
| **Data Augmentation Leakage** | Applying augmentation to test/validation sets artificially distorts evaluation metrics. | **Train-Only Augmentation:** Random horizontal/vertical flips and color jitters are applied exclusively to training batches. |

---

### 5. Prediction Horizon and Early Detection Protocol

- **Chronological Evaluation Points:**
  - *Period 1 (Vegetative Establishment):* Jan 18 – Feb 01 (Days 1–14) — Pre-stress baseline.
  - *Period 2 (Early Stress Transition):* Feb 05 – Feb 19 (Days 18–32) — Initial soil drying & chlorophyll shifts.
  - *Period 3 (Established Stress):* Feb 23 – Mar 08 (Days 36–50) — Visible leaf wilting, severe moisture deficit.
  - *Period 4 (Late / Severe Stress):* Mar 12 – Mar 29 (Days 54–71) — Chronic stress and biomass divergence.
- **Evaluation Criteria:** Performance (Balanced Accuracy, Macro-F1, ROC-AUC) will be tracked across each observation period to determine the exact date where multimodal fusion surpasses unimodal baselines.
