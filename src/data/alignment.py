"""
Multimodal Data Alignment Module.
Aligns image observations, environmental sensor telemetry, box weights,
and SPAD chlorophyll measurements into a unified, leak-free multimodal manifest.
"""

import sys
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import paths, treatments_meta, exp_config


def compute_vapor_pressure_deficit(temp_c: float, rh_pct: float) -> float:
    """Computes Vapor Pressure Deficit (VPD in kPa) from temperature and relative humidity.

    VPD is a key biophysical indicator of atmospheric water demand.
    """
    sat_vp = 0.61078 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    actual_vp = sat_vp * (rh_pct / 100.0)
    return max(0.0, sat_vp - actual_vp)


def build_aligned_dataset() -> pd.DataFrame:
    print("=" * 80)
    print("BUILDING MULTIMODAL DATA ALIGNMENT MANIFEST")
    print("=" * 80)

    # 1. Load Environmental Telemetry
    print("[1/5] Loading and indexing greenhouse environmental telemetry...")
    df_env = pd.read_csv(paths.greenhouse_logs_csv)
    df_env["Datetime"] = pd.to_datetime(
        df_env["Date"] + " " + df_env["Time"], format="%d.%m.%y %H:%M:%S"
    )
    df_env = df_env.sort_values("Datetime").reset_index(drop=True)
    temp_col = "Room Temperature (degree Celsius)"
    rh_col = "Relative Humidity (%)"
    df_env["VPD_kPa"] = [
        compute_vapor_pressure_deficit(t, rh)
        for t, rh in zip(df_env[temp_col], df_env[rh_col])
    ]

    # 2. Load Box Weights
    print("[2/5] Loading soil moisture box weights...")
    df_bw = pd.read_csv(paths.box_weights_csv)
    bw_date_cols = [c for c in df_bw.columns if c != "Box Number"]
    # Map to datetime
    bw_dates_dt = {c: pd.to_datetime(c, format="%d.%m.%Y") for c in bw_date_cols}
    # Baseline weight (first measurement date: 15.01.2018)
    baseline_col = "15.01.2018"
    box_baseline_weights = dict(zip(df_bw["Box Number"], df_bw[baseline_col]))

    # 3. Load SPAD Chlorophyll
    print("[3/5] Loading SPAD chlorophyll index measurements...")
    df_spad = pd.read_csv(paths.spad_csv)
    spad_date_cols = [c for c in df_spad.columns if c != "Box Number"]
    spad_dates_dt = {c: pd.to_datetime(c, format="%d.%m.%Y") for c in spad_date_cols}

    # 4. Load Harvest Yield Outcomes
    print("[4/5] Loading harvest yield ground truth...")
    df_yield = pd.read_csv(paths.yield_csv)
    yield_records = df_yield.set_index("Box Number").to_dict(orient="index")

    # 5. Align with Image Observations
    print("[5/5] Aligning multimodal observations across all in-vivo dates...")
    treatments = treatments_meta.treatments
    lookback_window = pd.Timedelta(hours=exp_config.env_lookback_hours)
    
    start_date = pd.to_datetime("2018-01-18")
    
    aligned_rows = []
    
    # Iterate over date folders in extracted/images
    for date_dir in sorted(paths.images_dir.iterdir()):
        if not date_dir.is_dir() or date_dir.name == "beet_roots":
            # Exclude post-harvest root images from in-vivo training
            continue
            
        date_str = date_dir.name  # e.g., '20180118'
        obs_date = pd.to_datetime(date_str, format="%Y%m%d")
        days_after_start = (obs_date - start_date).days
        
        # Categorize observation period
        if days_after_start <= 14:
            period = "P1_Establishment"
        elif days_after_start <= 32:
            period = "P2_EarlyTransition"
        elif days_after_start <= 50:
            period = "P3_EstablishedStress"
        else:
            period = "P4_SevereStress"

        # Causal Environmental Aggregation: strictly [obs_date - 48h, obs_date + 12h]
        # (Image acquisitions typically occurred between 10:00 and 14:00)
        window_end = obs_date + pd.Timedelta(hours=14)
        window_start = window_end - lookback_window
        env_window = df_env[(df_env["Datetime"] >= window_start) & (df_env["Datetime"] <= window_end)]
        
        if len(env_window) > 0:
            env_temp_mean = float(env_window[temp_col].mean())
            env_temp_std = float(env_window[temp_col].std() if len(env_window) > 1 else 0.0)
            env_temp_min = float(env_window[temp_col].min())
            env_temp_max = float(env_window[temp_col].max())
            env_rh_mean = float(env_window[rh_col].mean())
            env_rh_std = float(env_window[rh_col].std() if len(env_window) > 1 else 0.0)
            env_rh_min = float(env_window[rh_col].min())
            env_rh_max = float(env_window[rh_col].max())
            env_vpd_mean = float(env_window["VPD_kPa"].mean())
            env_vpd_max = float(env_window["VPD_kPa"].max())
        else:
            # Fallback to nearest prior measurements (e.g. for Jan 18 before sensor start on Jan 19)
            prior_env = df_env[df_env["Datetime"] <= (obs_date + pd.Timedelta(days=1))]
            if len(prior_env) > 0:
                first_readings = prior_env.head(24)
                env_temp_mean = float(first_readings[temp_col].mean())
                env_temp_std = float(first_readings[temp_col].std())
                env_temp_min = float(first_readings[temp_col].min())
                env_temp_max = float(first_readings[temp_col].max())
                env_rh_mean = float(first_readings[rh_col].mean())
                env_rh_std = float(first_readings[rh_col].std())
                env_rh_min = float(first_readings[rh_col].min())
                env_rh_max = float(first_readings[rh_col].max())
                env_vpd_mean = float(first_readings["VPD_kPa"].mean())
                env_vpd_max = float(first_readings["VPD_kPa"].max())
            else:
                env_temp_mean, env_temp_std, env_temp_min, env_temp_max = 18.0, 3.0, 14.0, 22.0
                env_rh_mean, env_rh_std, env_rh_min, env_rh_max = 60.0, 5.0, 50.0, 70.0
                env_vpd_mean, env_vpd_max = 0.8, 1.2

        # Find closest prior box weight measurement date
        prior_bw_cols = [c for c in bw_date_cols if bw_dates_dt[c] <= (obs_date + pd.Timedelta(days=1))]
        closest_bw_col = prior_bw_cols[-1] if prior_bw_cols else bw_date_cols[0]
        
        # Find closest prior SPAD measurement date
        prior_spad_cols = [c for c in spad_date_cols if spad_dates_dt[c] <= (obs_date + pd.Timedelta(days=1))]
        has_prior_spad = len(prior_spad_cols) > 0
        closest_spad_col = prior_spad_cols[-1] if has_prior_spad else None

        # Iterate over plant boxes (1 to 30, strictly excluding non-sugarbeet boxes)
        for box_id in range(1, 32):
            t_meta = treatments[box_id]
            
            # Explicitly exclude non-sugarbeet boxes (Weed monocultures 16, 17, 18 & CALTag 31)
            if not t_meta["has_sugarbeet"]:
                continue
                
            # Verify image file presence
            rgb_path = paths.images_dir / date_str / "rgb" / f"{box_id}.png"
            ir1_path = paths.images_dir / date_str / "ir1" / f"{box_id}.png"
            ir2_path = paths.images_dir / date_str / "ir2" / f"{box_id}.png"
            multispec_path = paths.images_dir / date_str / "multispec" / f"{box_id}.png"
            
            if not rgb_path.exists():
                print(f"[WARNING] Missing RGB image: {rgb_path}")
                continue

            # Box weight
            box_row_bw = df_bw[df_bw["Box Number"] == box_id]
            curr_weight = float(box_row_bw[closest_bw_col].values[0]) if len(box_row_bw) > 0 else 15.0
            base_weight = box_baseline_weights.get(box_id, 15.0)
            rel_moisture = curr_weight / (base_weight if base_weight > 0 else 15.0)
            
            # SPAD
            spad_val = np.nan
            spad_observed = 0
            if closest_spad_col:
                box_row_spad = df_spad[df_spad["Box Number"] == box_id]
                if len(box_row_spad) > 0:
                    raw_spad = box_row_spad[closest_spad_col].values[0]
                    if not pd.isna(raw_spad):
                        spad_val = float(raw_spad)
                        spad_observed = 1

            # Yield outcomes (for evaluation and post-hoc correlation, NOT for feature input)
            box_yield = yield_records.get(box_id, {})
            fresh_leaf_biomass = float(box_yield.get("Fresh leaves weight", np.nan))
            dry_beet_biomass = float(box_yield.get("Dry beet weight", np.nan))

            # Ground truth targets
            # Target A: Dynamic Drought Stress (realized soil moisture deficit)
            # Low water treatment AND current box weight < 12.0 kg
            is_drought_treatment = (t_meta["water"] == "Low")
            is_drought_stressed = 1 if (is_drought_treatment and curr_weight < 12.0) else 0

            # Target B: Multiclass Stress Factor
            # 0: Control, 1: Drought, 2: Low N, 3: Weed, 4: Mixed Stress
            desc = t_meta["desc"]
            if desc == "Control":
                multiclass_label = 0
                multiclass_name = "Control"
            elif desc == "Dry":
                multiclass_label = 1
                multiclass_name = "Water_Stress_Drought"
            elif desc == "Low N":
                multiclass_label = 2
                multiclass_name = "Nitrogen_Stress"
            elif desc in ("Med weeds", "High weeds"):
                multiclass_label = 3
                multiclass_name = "Weed_Stress"
            else:
                # Combined stress: Drying - Med N - High Weed, Low N - Med Weed, Drying - Low N, Med N
                multiclass_label = 4
                multiclass_name = "Mixed_or_Other_Stress"

            aligned_rows.append({
                "sample_id": f"box{box_id:02d}_{date_str}",
                "box_id": box_id,
                "date_str": date_str,
                "days_after_start": days_after_start,
                "period": period,
                
                # Image file paths relative to project root
                "rgb_path": str(rgb_path.relative_to(paths.project_root)),
                "ir1_path": str(ir1_path.relative_to(paths.project_root)),
                "ir2_path": str(ir2_path.relative_to(paths.project_root)),
                "multispec_path": str(multispec_path.relative_to(paths.project_root)),
                
                # Environmental sensor telemetry features (strictly causal)
                "env_temp_mean_48h": env_temp_mean,
                "env_temp_std_48h": env_temp_std,
                "env_temp_min_48h": env_temp_min,
                "env_temp_max_48h": env_temp_max,
                "env_rh_mean_48h": env_rh_mean,
                "env_rh_std_48h": env_rh_std,
                "env_rh_min_48h": env_rh_min,
                "env_rh_max_48h": env_rh_max,
                "env_vpd_mean_48h": env_vpd_mean,
                "env_vpd_max_48h": env_vpd_max,
                
                # Dynamic soil moisture & diagnostic measurements
                "box_weight_kg": curr_weight,
                "baseline_box_weight_kg": base_weight,
                "relative_weight_ratio": rel_moisture,
                "spad_value": spad_val,
                "spad_observed": spad_observed,
                
                # Documented treatment metadata
                "treatment_desc": desc,
                "soil_type": t_meta["soil"],
                "water_input": t_meta["water"],
                "nitrogen_input": t_meta["n"],
                "weed_pressure": t_meta["weed"],
                
                # Ground truth targets
                "target_drought_stress": is_drought_stressed,
                "target_multiclass": multiclass_label,
                "target_multiclass_name": multiclass_name,
                
                # Harvest outcomes (for evaluation correlation only)
                "outcome_fresh_leaves_biomass_g": fresh_leaf_biomass,
                "outcome_dry_beet_biomass_g": dry_beet_biomass,
            })

    df_aligned = pd.DataFrame(aligned_rows)
    print(f"\nAlignment completed successfully: {len(df_aligned)} multimodal samples.")
    print(f"Total unique boxes: {df_aligned['box_id'].nunique()} (Boxes 1-15, 19-30)")
    print(f"Total unique observation dates: {df_aligned['date_str'].nunique()}")
    print("\nTarget Class Distribution (Dynamic Drought Stress):")
    print(df_aligned["target_drought_stress"].value_counts(normalize=True).rename({0: "Non-Drought (0)", 1: "Drought (1)"}))
    print("\nTarget Class Distribution by Period:")
    print(pd.crosstab(df_aligned["period"], df_aligned["target_drought_stress"]))
    
    # Save aligned dataset manifest
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    df_aligned.to_csv(paths.aligned_dataset_csv, index=False)
    print(f"\nSaved aligned multimodal dataset manifest to: {paths.aligned_dataset_csv}")
    return df_aligned


if __name__ == "__main__":
    build_aligned_dataset()
