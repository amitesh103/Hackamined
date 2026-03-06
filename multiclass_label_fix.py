"""
╔══════════════════════════════════════════════════════════════════════════╗
║   MULTI-CLASS LABEL FIX — Final Version                                 ║
║   Problem: op_state==4 is normal (50% of rows), not degradation         ║
║   Fix: Define degradation using ACTUAL performance metrics               ║
║        (conversion efficiency + power output drop vs rolling baseline)  ║
╚══════════════════════════════════════════════════════════════════════════╝

WHAT WE NOW KNOW ABOUT YOUR DATA:
───────────────────────────────────
  op_state==4 = 193,779 rows (50.7%)  → normal limited/regulated output
  op_state==0 = 175,507 rows (45.9%)  → nighttime off
  op_state==3 =   7,069 rows  (1.8%)  → full power generation
  alarm 534/556 =  48 rows  (0.01%)   → confirmed hardware faults

REVISED 3-CLASS STRATEGY:
───────────────────────────
  Class 0 — No Risk:
    Normal operation. No fault within 7 days.
    No sustained efficiency drop vs own historical baseline.

  Class 1 — Degradation Risk:
    Conversion efficiency drops >10% below inverter's own 7-day rolling mean
    during daylight hours, sustained for >4 hours.
    No confirmed shutdown fault within 7 days.
    This captures slow thermal drift, soiling, and partial faults.

  Class 2 — Shutdown Risk:
    alarm_code IN [534, 556] occurs within next 7 days.
    Highest urgency — schedule immediate inspection.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
FILES = {
    # Plant 3
    "INV-6E":   r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 3-20260305T111819Z-3-001\Plant 3\Copy of 54-10-EC-8C-14-6E.raws.csv",
    "INV-69":   r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 3-20260305T111819Z-3-001\Plant 3\Copy of 54-10-EC-8C-14-69.raws.csv",
    # Plant 2 — same column structure, fully compatible
    "INV-AC12": r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 2-20260305T111818Z-3-001\Plant 2\Copy of 80-1F-12-0F-AC-12.raws.csv",
    "INV-ACBB": r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 2-20260305T111818Z-3-001\Plant 2\Copy of 80-1F-12-0F-AC-BB.raws.csv",
}

ROWS_PER_HOUR          = 12
PREDICTION_WINDOW_DAYS = 7
LABEL_WINDOW           = PREDICTION_WINDOW_DAYS * 24 * ROWS_PER_HOUR
MAJOR_ALARM_CODES      = [534, 556]

# Degradation thresholds
EFFICIENCY_DROP_PCT    = 10    # % below own 7-day mean = degradation signal
EFFICIENCY_SUSTAIN_HRS = 4     # must persist 4+ hours to count
EFFICIENCY_SUSTAIN_ROWS= EFFICIENCY_SUSTAIN_HRS * ROWS_PER_HOUR  # 48 rows

def divider(title):
    print(f"\n{'='*65}\n  {title}\n{'='*65}")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD & PREP (abbreviated — same as before)
# ══════════════════════════════════════════════════════════════════════════════
divider("Load & Prep")

dfs = []
for inv_id, filepath in FILES.items():
    df = pd.read_csv(filepath, low_memory=False)
    df["inverter_id"] = inv_id
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df["ts"] = pd.to_datetime(df["timestampDate"], utc=True, errors="coerce").dt.tz_convert(None)  # strip UTC tz → naive datetime
df = df.dropna(subset=["ts"]).sort_values(["inverter_id", "ts"]).reset_index(drop=True)

# Rename core columns
df = df.rename(columns={
    "inverters[0].alarm_code":       "alarm_code",
    "inverters[0].op_state":         "op_state",
    "inverters[0].power":            "ac_power",
    "inverters[0].pv1_power":        "dc_power_raw",
    "inverters[0].pv1_voltage":      "dc_voltage",
    "inverters[0].temp":             "inverter_temp",
})

# Fix dtypes
df["ac_power"]    = pd.to_numeric(df["ac_power"],    errors="coerce").clip(lower=0)
df["dc_power_raw"]= pd.to_numeric(df["dc_power_raw"],errors="coerce").clip(lower=0)
df["alarm_code"]  = pd.to_numeric(df["alarm_code"],  errors="coerce")

print(f"  Loaded: {len(df):,} rows across {df['inverter_id'].nunique()} inverters")

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE CONVERSION EFFICIENCY (needed for degradation detection)
# ══════════════════════════════════════════════════════════════════════════════
divider("Compute Conversion Efficiency")

df["conversion_efficiency"] = np.where(
    df["dc_power_raw"] > 0.5,
    (df["ac_power"] / df["dc_power_raw"]) * 100,
    np.nan
)
df["conversion_efficiency"] = df["conversion_efficiency"].clip(0, 105)

# 14-day rolling mean efficiency per inverter — longer window avoids baseline contamination
# (if we used 7d, a degradation already in progress would skew the mean down,
#  making the efficiency_gap smaller and causing us to miss the anomaly)
EFF_BASELINE_WINDOW = 14 * 24 * ROWS_PER_HOUR  # 4032 rows
df["eff_rolling_mean_7d"] = df.groupby("inverter_id")["conversion_efficiency"].transform(
    lambda x: x.rolling(EFF_BASELINE_WINDOW, min_periods=EFF_BASELINE_WINDOW // 4).mean()
)

# Efficiency gap = how far below its own baseline is the inverter right now?
df["efficiency_gap"] = df["eff_rolling_mean_7d"] - df["conversion_efficiency"]
# Positive gap = current efficiency BELOW rolling mean (bad)
# Negative gap = current efficiency ABOVE rolling mean (good)

print(f"  Conversion efficiency stats (daylight only):")
mask_day = df["ac_power"] > 0.5
for inv_id, grp in df[mask_day].groupby("inverter_id"):
    eff = grp["conversion_efficiency"].dropna()
    print(f"    {inv_id}: mean={eff.mean():.1f}%  std={eff.std():.1f}%  "
          f"min={eff.min():.1f}%  max={eff.max():.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# BUILD DEGRADATION SIGNAL
# ══════════════════════════════════════════════════════════════════════════════
divider("Build Degradation Signal (Performance-Based)")

"""
LOGIC:
  A row is "degrading" if BOTH conditions are true:
    1. Efficiency is >10% below its own 7-day rolling mean (not just low, but dropping)
    2. It's daytime (ac_power > 0.5 kW) — efficiency is undefined at night

  A row is "sustained degradation" if degradation persists for 4+ consecutive hours.
  This filters out brief dips from clouds/transients.
"""

# Step 1: Flag each row where efficiency drop exceeds threshold
df["is_efficiency_drop"] = (
    (df["efficiency_gap"] > EFFICIENCY_DROP_PCT) &
    (df["ac_power"] > 0.5)  # daylight only
).astype(int)

# Step 2: Rolling sum to check persistence (4-hour window)
df["efficiency_drop_sustained"] = df.groupby("inverter_id")["is_efficiency_drop"].transform(
    lambda x: (x.rolling(EFFICIENCY_SUSTAIN_ROWS, min_periods=1).sum()
               >= EFFICIENCY_SUSTAIN_ROWS * 0.75).astype(int)
    # 75% of the 4-hour window must show the drop (allows for a few missing readings)
)

print(f"  Rows with sustained efficiency drop >10% for 4+ hours:")
for inv_id, grp in df.groupby("inverter_id"):
    n = grp["efficiency_drop_sustained"].sum()
    total = len(grp)
    print(f"    {inv_id}: {n:,} rows ({n/total*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# BUILD ALL THREE LABELS
# ══════════════════════════════════════════════════════════════════════════════
divider("Build 3-Class Label")

def forward_rolling_any(series, window):
    reversed_sum = series[::-1].rolling(window=window, min_periods=1).sum()[::-1]
    return (reversed_sum > 0).astype(int)

# Fault events
df["is_major_fault"] = df["alarm_code"].isin(MAJOR_ALARM_CODES).astype(int)

# Forward labels: "will X happen in the next 7 days?"
df["shutdown_risk_fwd"] = df.groupby("inverter_id")["is_major_fault"].transform(
    lambda x: forward_rolling_any(x, LABEL_WINDOW)
)
df["degradation_risk_fwd"] = df.groupby("inverter_id")["efficiency_drop_sustained"].transform(
    lambda x: forward_rolling_any(x, LABEL_WINDOW)
)

# 3-class label (higher class overrides lower)
df["risk_class"] = 0
df.loc[df["degradation_risk_fwd"] == 1, "risk_class"] = 1   # degradation
df.loc[df["shutdown_risk_fwd"]    == 1, "risk_class"] = 2   # shutdown (always wins)

# Binary label
df["failure_label"] = (df["shutdown_risk_fwd"] == 1).astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# PRINT FINAL DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
divider("Final Label Distribution")

labels = {0: "No Risk        ", 1: "Degradation Risk", 2: "Shutdown Risk  "}
total  = len(df)

print(f"\n  MULTI-CLASS (risk_class) — both inverters combined:")
print(f"  {'─'*55}")
vc3 = df["risk_class"].value_counts().sort_index()
for cls, cnt in vc3.items():
    pct = cnt / total * 100
    bar = "█" * int(pct / 2)
    print(f"  Class {cls} | {labels[cls]} | {cnt:7,} rows ({pct:5.1f}%)  {bar}")

print(f"\n  BINARY (failure_label) — shutdown risk only:")
print(f"  {'─'*55}")
vc2 = df["failure_label"].value_counts().sort_index()
for lbl, cnt in vc2.items():
    pct = cnt / total * 100
    bar = "█" * int(pct / 2)
    print(f"  Label {lbl} | {cnt:7,} rows ({pct:5.1f}%)  {bar}")

print(f"\n  Per-inverter breakdown:")
print(f"  {'─'*55}")
for inv_id, grp in df.groupby("inverter_id"):
    s0 = (grp["risk_class"] == 0).sum()
    s1 = (grp["risk_class"] == 1).sum()
    s2 = (grp["risk_class"] == 2).sum()
    g  = len(grp)
    print(f"\n  {inv_id}  ({g:,} total rows):")
    print(f"    Class 0 No Risk       : {s0:>7,}  ({s0/g*100:5.1f}%)")
    print(f"    Class 1 Degradation   : {s1:>7,}  ({s1/g*100:5.1f}%)")
    print(f"    Class 2 Shutdown Risk : {s2:>7,}  ({s2/g*100:5.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# PATCH INTO FEATURE CSV
# ══════════════════════════════════════════════════════════════════════════════
divider("Patching Final Labels into Feature CSV")

try:
    df_ml = pd.read_csv("inverter_features_REVISED.csv", parse_dates=["timestamp"])

    # Drop old label columns
    drop_cols = ["failure_label", "risk_class", "shutdown_risk",
                 "degradation_risk", "is_derated", "derated_persistent"]
    df_ml = df_ml.drop(columns=[c for c in drop_cols if c in df_ml.columns])

    # Build merge key
    revised = df[[
        "inverter_id", "ts",
        "failure_label", "risk_class",
        "shutdown_risk_fwd", "degradation_risk_fwd",
        "is_efficiency_drop", "efficiency_gap",
    ]].rename(columns={"ts": "timestamp"})

    df_ml = df_ml.merge(revised, on=["inverter_id", "timestamp"], how="left")

    # Fill any unmatched rows
    df_ml["failure_label"]         = df_ml["failure_label"].fillna(0).astype(int)
    df_ml["risk_class"]            = df_ml["risk_class"].fillna(0).astype(int)
    df_ml["efficiency_gap"]        = df_ml["efficiency_gap"].fillna(0)
    df_ml["is_efficiency_drop"]    = df_ml["is_efficiency_drop"].fillna(0).astype(int)

    df_ml.to_csv("inverter_features_FINAL.csv", index=False)

    print(f"\n  Feature matrix shape : {df_ml.shape}")
    print(f"  New feature added    : efficiency_gap (SHAP-important)")

    print(f"\n  ✅  FINAL label distribution in inverter_features_FINAL.csv:")
    vc = df_ml["risk_class"].value_counts().sort_index()
    for cls, cnt in vc.items():
        pct = cnt / len(df_ml) * 100
        print(f"    Class {cls}: {cnt:7,} rows ({pct:.1f}%)")

    print(f"\n  ✅ Saved → inverter_features_FINAL.csv")

except FileNotFoundError:
    print(f"  ⚠️  inverter_features_REVISED.csv not found.")
    print(f"  Run label_fix_patch.py first, then re-run this script.")

# ══════════════════════════════════════════════════════════════════════════════
# HANDOFF SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"""
{'='*65}
  FINAL HANDOFF TO ML TEAM
{'='*65}

  FILE TO USE: inverter_features_FINAL.csv

  BINARY CLASSIFICATION (minimum requirement):
    Target column  : failure_label
    Positive class : 1 (shutdown within 7 days)
    XGBoost param  : scale_pos_weight = n_negative / n_positive

  MULTI-CLASS CLASSIFICATION (bonus requirement):
    Target column  : risk_class
    Class 0        : No Risk
    Class 1        : Degradation Risk (efficiency drop >10% for 4hrs)
    Class 2        : Shutdown Risk (alarm 534/556 within 7 days)
    XGBoost param  : objective = 'multi:softprob', num_class = 3

  TOP FEATURES TO WATCH IN SHAP:
    1. efficiency_gap          ← new! efficiency vs own baseline
    2. conversion_efficiency_std_7d   ← instability signal
    3. major_alarm_count_7d    ← fault history
    4. temp_delta_7d           ← thermal drift
    5. voltage_delta_7d        ← DC voltage sag
    6. alarm_acceleration      ← alarm frequency trend

  EVALUATION METRICS:
    Binary  : F1-score, AUC-ROC, Precision, Recall
    Multi   : Macro F1, Confusion Matrix per class
    DO NOT use accuracy alone — dataset is imbalanced
{'='*65}
""")