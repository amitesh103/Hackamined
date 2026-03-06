"""
╔══════════════════════════════════════════════════════════════════════════╗
║   MASTER COMBINER — All 3 Plants → Single Unified Output                ║
║   HACKaMINeD 2026 · Solar Inverter Failure Prediction                   ║
║                                                                          ║
║   PREREQUISITES:                                                         ║
║     pip install pandas numpy scikit-learn                               ║
║                                                                          ║
║   RUN ORDER:                                                             ║
║     1. python final_patch.py          → inverter_features_READY.csv     ║
║     2. python plant1_anomaly.py       → plant1_anomaly_scores.csv       ║
║     3. python master_combiner.py      → unified_all_plants.csv  ✅      ║
║                                                                          ║
║   OUTPUT COLUMNS KEY:                                                    ║
║     risk_score    → 0-100 unified risk score for ALL inverters          ║
║     risk_class    → 0/1/2 for Plants 2&3, derived for Plant 1          ║
║     data_source   → "supervised" or "anomaly_detection"                 ║
║     plant_id      → Plant1 / Plant2 / Plant3                           ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def divider(title):
    print(f"\n{'='*65}\n  {title}\n{'='*65}")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD BOTH PIPELINE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
divider("Loading Pipeline Outputs")

# Plants 2 & 3 — supervised model features
print("  Loading Plants 2 & 3 (supervised)...")
df_sup = pd.read_csv("inverter_features_READY.csv", low_memory=False)
df_sup["data_source"] = "supervised"
print(f"    Shape: {df_sup.shape}")
print(f"    Inverters: {df_sup['inverter_id'].unique().tolist()}")

# Plant 1 — anomaly detection scores
print("\n  Loading Plant 1 (anomaly detection)...")
df_p1 = pd.read_csv("plant1_anomaly_scores.csv", low_memory=False)
df_p1["data_source"] = "anomaly_detection"
print(f"    Shape: {df_p1.shape}")
print(f"    Inverters: {df_p1['inverter_id'].unique().tolist()}")


# ══════════════════════════════════════════════════════════════════════════════
# NORMALISE SUPERVISED RISK TO 0-100 SCORE
# ══════════════════════════════════════════════════════════════════════════════
divider("Normalising Risk Scores to Unified 0-100 Scale")

"""
Plants 2 & 3 give us:
  risk_class 0 = No Risk       → risk_score 0-30
  risk_class 1 = Degradation   → risk_score 31-65
  risk_class 2 = Shutdown Risk → risk_score 66-100

We scale based on class AND the underlying signal strength
(major_alarm_count_7d and degradation signals).

Plant 1 already has anomaly_score 0-100.
"""

def supervised_to_risk_score(df):
    """Convert risk_class to 0-100 continuous risk score."""
    score = pd.Series(0.0, index=df.index)

    # Class 0: base score 0-30
    mask0 = df["risk_class"] == 0
    score[mask0] = 10.0

    # Class 1: degradation risk 31-65
    mask1 = df["risk_class"] == 1
    # Boost by alarm count if available
    alarm_boost = df.get("minor_alarm_count_7d", pd.Series(0, index=df.index))
    score[mask1] = 40 + (alarm_boost[mask1].clip(0, 20) / 20 * 25)

    # Class 2: shutdown risk 66-100
    mask2 = df["risk_class"] == 2
    major_alarm = df.get("major_alarm_count_7d", pd.Series(0, index=df.index))
    score[mask2] = 75 + (major_alarm[mask2].clip(0, 10) / 10 * 25)

    return score.clip(0, 100)

df_sup["risk_score"] = supervised_to_risk_score(df_sup)

# Plant 1: anomaly_score is already 0-100
df_p1["risk_score"] = df_p1["anomaly_score"]

# Map Plant 1 anomaly_severity → risk_class for unified schema
severity_to_class = {0: 0, 1: 0, 2: 1, 3: 2}
df_p1["risk_class"] = df_p1["anomaly_severity"].map(severity_to_class).fillna(0).astype(int)

# Plant 1 has no failure_label (no ground truth) — mark as -1
df_p1["failure_label"] = -1

print(f"  Plants 2&3 risk_score distribution:")
print(f"    mean={df_sup['risk_score'].mean():.1f}  "
      f"max={df_sup['risk_score'].max():.1f}")
print(f"\n  Plant 1 risk_score distribution:")
print(f"    mean={df_p1['risk_score'].mean():.1f}  "
      f"max={df_p1['risk_score'].max():.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# ALIGN SCHEMAS (keep common columns + key unique ones)
# ══════════════════════════════════════════════════════════════════════════════
divider("Aligning Schemas for Unified Output")

# Columns that exist in both
SHARED_COLS = [
    "inverter_id", "plant_id", "plant_encoded", "data_source",
    "timestamp",

    # Risk labels
    "risk_score", "risk_class", "failure_label",

    # Core power telemetry
    "ac_power", "inverter_temp", "is_daylight",

    # Time features
    "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_week",

    # Rolling power
    "ac_power_mean_7d", "ac_power_std_7d",
    "power_delta_7d", "temp_delta_7d",

    # Degradation signals (supervised) / anomaly info (Plant 1)
    "anomaly_score", "anomaly_severity", "is_anomaly",
]

# Add supervised-only columns (will be NaN for Plant 1)
SUPERVISED_ONLY = [
    "failure_label", "risk_class",
    "dc_voltage", "dc_current", "conversion_efficiency",
    "alarm_code", "is_major_fault",
    "minor_alarm_count_7d", "major_alarm_count_7d",
    "signal_a_throttled", "signal_b_alarm_cluster", "signal_c_power_drop",
    "alarm_acceleration",
]

# Add Plant1-only columns (will be NaN for Plants 2&3)
PLANT1_ONLY = [
    "anomaly_score", "anomaly_severity", "is_anomaly",
    "string_imbalance", "dead_string_count",
    "temp_vs_fleet", "power_vs_fleet_pct",
    "comms_lost",
]

def safe_select(df, cols):
    """Select columns that exist in dataframe, add NaN for missing ones."""
    result = df.copy()
    for col in cols:
        if col not in result.columns:
            result[col] = np.nan
    available = [c for c in cols if c in result.columns or c in cols]
    return result[[c for c in cols if c in result.columns or True]].reindex(columns=cols)

ALL_COLS = list(dict.fromkeys(SHARED_COLS + SUPERVISED_ONLY + PLANT1_ONLY))

df_sup_aligned = safe_select(df_sup, ALL_COLS)
df_p1_aligned  = safe_select(df_p1,  ALL_COLS)

# Handle timestamp column name difference
if "ts" in df_p1.columns and "timestamp" not in df_p1.columns:
    df_p1_aligned["timestamp"] = df_p1["ts"]

print(f"  Unified schema: {len(ALL_COLS)} columns")
print(f"  Plants 2&3 rows: {len(df_sup_aligned):,}")
print(f"  Plant 1 rows   : {len(df_p1_aligned):,}")


# ══════════════════════════════════════════════════════════════════════════════
# COMBINE & EXPORT
# ══════════════════════════════════════════════════════════════════════════════
divider("Combining & Exporting")

df_all = pd.concat([df_sup_aligned, df_p1_aligned], ignore_index=True)
df_all = df_all.sort_values(["plant_id", "inverter_id", "timestamp"]).reset_index(drop=True)

df_all.to_csv("unified_all_plants.csv", index=False)

print(f"\n  ✅ Unified output: unified_all_plants.csv")
print(f"  Shape: {df_all.shape}")

# Summary table
divider("Final Summary — All Plants & Inverters")

print(f"\n  {'Plant':<10} {'Inverter':<15} {'Rows':>8} "
      f"{'Method':<22} {'Avg Risk':>8} {'High Risk%':>10}")
print(f"  {'─'*10} {'─'*15} {'─'*8} {'─'*22} {'─'*8} {'─'*10}")

for (plant, inv), grp in df_all.groupby(["plant_id", "inverter_id"]):
    method  = grp["data_source"].iloc[0]
    avg_rs  = grp["risk_score"].mean()
    high_rs = (grp["risk_score"] >= 66).mean() * 100
    print(f"  {plant:<10} {inv:<15} {len(grp):>8,} "
          f"{method:<22} {avg_rs:>8.1f} {high_rs:>9.1f}%")

print(f"\n  COLUMN GUIDE FOR ALL TEAMS:")
print(f"  {'─'*60}")
print(f"""
  risk_score      → 0-100 unified risk (higher = more dangerous)
                    Works for ALL 6 inverters across all 3 plants

  risk_class      → 0=No Risk, 1=Degradation, 2=Shutdown Risk
                    Plants 2&3: from supervised model labels
                    Plant 1: derived from anomaly_severity

  failure_label   → 0/1 binary (Plants 2&3 only)
                    -1 for Plant 1 (no ground truth available)

  data_source     → "supervised" or "anomaly_detection"
                    Dashboard can show different confidence badges

  anomaly_score   → 0-100 IsolationForest score (Plant 1 only)
  anomaly_severity→ 0-3 severity tier (Plant 1 only)

  FOR ML TEAM:
    Train on:  inverter_features_READY.csv  (Plants 2&3, has labels)
    Infer on:  plant1_anomaly_scores.csv    (Plant 1, anomaly scores)
    Combined:  unified_all_plants.csv       (dashboard & GenAI)

  FOR DASHBOARD TEAM:
    Use: unified_all_plants.csv
    Key cols: inverter_id, plant_id, timestamp,
              risk_score, risk_class, data_source

  FOR GENAI TEAM:
    Use: plant1_top_anomalies.csv for Plant 1 narratives
    Use: risk_class==2 rows from inverter_features_READY.csv
         for Plants 2&3 narratives
""")

print(f"{'='*65}")
print(f"  COMPLETE — All 6 inverters across 3 plants processed")
print(f"  Feature engineering fully done ✅")
print(f"{'='*65}\n")
