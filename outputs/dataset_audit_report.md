# ETH Zurich Eschikon Plant Stress Phenotyping Dataset — Comprehensive Audit Report

## 1. Executive Summary

This audit report documents the empirical structure, modality availability, missingness characteristics, and experimental design of the official ETH Zurich Eschikon Plant Stress Phenotyping Dataset (2018).

- **Total Experimental Units (Boxes):** 31 boxes
  - **Sugarbeet Crop Treatment Boxes:** 27 boxes (Boxes 1–15, 19–30)
  - **Weed-Only Monoculture Boxes:** 3 boxes (Boxes 16, 17, 18 — no sugarbeet planted)
  - **Calibration / Color Target Box:** 1 box (Box 31 — CALTag & Macbeth color chart)
- **Total Image Files:** 2,108 images across 17 folders (16 observation dates + harvest root images)
- **Available Image Modalities:** 4 (RGB @ 1920x1080, IR1 stereo left @ 640x480, IR2 stereo right @ 640x480, Multispectral 5x5 snapshot mosaic @ 2048x1088)
- **Environmental Sensor Log Records:** 9,721 continuous 12-minute readings over 81 days (Jan 19 – Apr 10, 2018)
- **Box Weight Soil Moisture Log Records:** 18 dates x 31 boxes = 558 measurements (0% missing)
- **SPAD Chlorophyll Measurements:** 5 dates x 27 plant boxes = 135 measurements (0% missing for plant boxes; NaN for weed-only & caltag)
- **Harvest Biomass Outcomes:** 4 metrics x 27 plant boxes = 108 measurements (0% missing for plant boxes)

## 2. Experimental Design & Treatment Allocation

| Box ID | Treatment Description | Soil Type | Water Input | Nitrogen Input | Weed Pressure | Has Sugarbeet? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Control | 2 | Sufficient | High | None | Yes |
| 2 | Control | 2 | Sufficient | High | None | Yes |
| 3 | Control | 2 | Sufficient | High | None | Yes |
| 4 | Med N | 2 | Sufficient | Medium | None | Yes |
| 5 | Med N | 2 | Sufficient | Medium | None | Yes |
| 6 | Med N | 2 | Sufficient | Medium | None | Yes |
| 7 | Med weeds | 2 | Sufficient | High | Medium | Yes |
| 8 | Med weeds | 2 | Sufficient | High | Medium | Yes |
| 9 | Med weeds | 2 | Sufficient | High | Medium | Yes |
| 10 | High weeds | 2 | Sufficient | High | High | Yes |
| 11 | High weeds | 2 | Sufficient | High | High | Yes |
| 12 | High weeds | 2 | Sufficient | High | High | Yes |
| 13 | Dry | 2 | Low | Medium | None | Yes |
| 14 | Dry | 2 | Low | Medium | None | Yes |
| 15 | Dry | 2 | Low | Medium | None | Yes |
| 16 | Weed Only Dicot | 2 | Sufficient | High | High | No |
| 17 | Weed Only Monocot | 2 | Sufficient | High | High | No |
| 18 | Weed Only Mixed | 2 | Sufficient | High | High | No |
| 19 | Drying - Med N - High Weed | 2 | Low | Medium | High | Yes |
| 20 | Drying - Med N - High Weed | 2 | Low | Medium | High | Yes |
| 21 | Drying - Med N - High Weed | 2 | Low | Medium | High | Yes |
| 22 | Low N | 1 | Sufficient | Low | None | Yes |
| 23 | Low N | 1 | Sufficient | Low | None | Yes |
| 24 | Low N | 1 | Sufficient | Low | None | Yes |
| 25 | Low N - Med Weed | 1 | Sufficient | Low | Medium | Yes |
| 26 | Low N - Med Weed | 1 | Sufficient | Low | Medium | Yes |
| 27 | Low N - Med Weed | 1 | Sufficient | Low | Medium | Yes |
| 28 | Drying - Low N | 1 | Low | Low | None | Yes |
| 29 | Drying - Low N | 1 | Low | Low | None | Yes |
| 30 | Drying - Low N | 1 | Low | Low | None | Yes |
| 31 | CALTag and Color Checker | 1 | NA | NA | NA | No |

## 3. Modality Timeline & Cross-Modality Alignment

| Observation Date | In-Vivo Images (4 cameras) | Box Weights (Moisture) | Environmental Logs | SPAD Chlorophyll | Post-Harvest Roots |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 20180115 | — | ✓ | — | — | — |
| 20180118 | ✓ | — | — | — | — |
| 20180119 | — | ✓ | ✓ | — | — |
| 20180124 | — | ✓ | ✓ | — | — |
| 20180129 | — | ✓ | ✓ | — | — |
| 20180130 | ✓ | ✓ | ✓ | — | — |
| 20180201 | ✓ | ✓ | ✓ | — | — |
| 20180205 | ✓ | ✓ | ✓ | — | — |
| 20180208 | ✓ | ✓ | ✓ | — | — |
| 20180212 | ✓ | ✓ | ✓ | — | — |
| 20180215 | ✓ | ✓ | ✓ | — | — |
| 20180219 | ✓ | ✓ | ✓ | — | — |
| 20180223 | ✓ | ✓ | ✓ | — | — |
| 20180226 | ✓ | ✓ | ✓ | — | — |
| 20180302 | ✓ | ✓ | ✓ | — | — |
| 20180305 | ✓ | ✓ | ✓ | — | — |
| 20180308 | ✓ | ✓ | ✓ | — | — |
| 20180312 | ✓ | ✓ | ✓ | — | — |
| 20180315 | ✓ | ✓ | ✓ | — | — |
| 20180329 | ✓ | — | ✓ | ✓ | — |
| beet_roots (Harvest) | — | — | ✓ | — | ✓ |

## 4. Key Empirical Findings & Scientific Considerations

1. **Strict Experimental Unit Independence:** The true independent experimental unit is the **box** ($N=27$ plant boxes), NOT individual images. Multiple images over time or across camera modalities of the same box are repeated longitudinal measures of that single box. Any model evaluation that splits images randomly will suffer from severe **data leakage** and report falsely inflated generalization.
2. **Dynamic Drought Onset:** Analysis of `box-weights-kg.csv` demonstrates that drought stress is NOT present at day 1. From Jan 15 to Feb 1, box weights between 'Sufficient' and 'Low' water groups were identical (~15.7 kg). The true divergence begins in mid-February, dropping to 8.65 kg by March. This provides a genuine **early-stress vs. established-stress temporal progression**.
3. **Missing Modality Ground Truth:** Boxes 16, 17, 18 (weed only) and 31 (caltag) do not contain sugarbeets. SPAD and yield values are genuinely non-existent for them. The modeling pipeline must restrict sugarbeet health predictions to Boxes 1–15 and 19–30.
4. **Environmental Stationarity:** The greenhouse environmental sensor logs are room-level (ambient temperature and humidity) and are shared across all boxes at any given point in time. Therefore, ambient conditions alone cannot distinguish a drought box from a control box at a single static instant—they provide **temporal climate context** (e.g. heat load, vapor pressure deficit) that modulates how visual and soil symptoms manifest.
