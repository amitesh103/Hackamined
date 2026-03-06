"""
╔══════════════════════════════════════════════════════════════════════════╗
║   DEGRADATION CLASS FIX — Final Version                                 ║
║                                                                          ║
║   Problem: Efficiency is too stable (97.3% ± 1.4%) for a 10% drop     ║
║   to ever occur → Class 1 = 0 rows                                      ║
║                                                                          ║
║   Fix: Use what THIS dataset actually has:                              ║
║     Signal A: limit_percent < 90  (inverter being throttled)           ║
║     Signal B: minor alarms 8/10/12/39 clustering within 24 hours       ║
║     Signal C: power output > 2 std below own 7-day rolling mean        ║
║                                                                          ║
║   HOW TO RUN:                                                            ║
║     python degradation_fix_final.py                                     ║
║     → Produces: inverter_features_FINAL_v2.csv                         ║
╚══════════════════════════════════════════════════════════════════════════╝
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
MINOR_ALARM_CODES      = [8, 10, 12, 39]

def divider(title):
    print(f"\n{'='*65}\n  {title}\n{'='*65}")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD & PREP
# ══════════════════════════════════════════════════════════════════════════════
divider("Load & Prep")

dfs = []
for inv_id, filepath in FILES.items():
    df = pd.read_csv(filepath, low_memory=False)
    df["inverter_id"] = inv_id
    dfs.append(df)
    print(f"  {inv_id}: {len(df):,} rows")

df = pd.concat(dfs, ignore_index=True)
df["ts"] = pd.to_datetime(
    df["timestampDate"], utc=True, errors="coerce"
).dt.tz_convert(None)  # strip UTC tz → naive datetime
df = df.dropna(subset=["ts"]).sort_values(["inverter_id", "ts"]).reset_index(drop=True)

# Rename core columns
df = df.rename(columns={
    "inverters[0].alarm_code":      "alarm_code",
    "inverters[0].op_state":        "op_state",
    "inverters[0].power":           "ac_power",
    "inverters[0].pv1_power":       "dc_power_raw",
    "inverters[0].pv1_voltage":     "dc_voltage",
    "inverters[0].temp":            "inverter_temp",
    "inverters[0].limit_percent":   "limit_percent",
})

# Force numeric
for col in ["ac_power", "dc_power_raw", "alarm_code", "limit_percent", "dc_voltage"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["ac_power"]     = df["ac_power"].clip(lower=0)
df["dc_power_raw"] = df["dc_power_raw"].clip(lower=0)

# Daylight flag
df["is_daylight"] = (df["ac_power"] > 0.5).astype(int)

print(f"\n  Total rows: {len(df):,}")
print(f"  Daylight rows: {df['is_daylight'].sum():,} ({df['is_daylight'].mean()*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# UNDERSTAND THE DATA BEFORE LABELLING
# (print this so you understand what signals are available)
# ══════════════════════════════════════════════════════════════════════════════
divider("Data Reality Check")

print("\n  limit_percent distribution (per inverter):")
for inv_id, grp in df.groupby("inverter_id"):
    lp = grp["limit_percent"].dropna()
    vc = lp.value_counts().sort_index()
    print(f"\n  {inv_id}  (non-null: {len(lp):,}):")
    for val, cnt in vc.head(10).items():
        pct = cnt / len(lp) * 100
        print(f"    {val:>7.1f}%  →  {cnt:>7,} rows  ({pct:.1f}%)")

print("\n\n  Minor alarm distribution (codes 8/10/12/39):")
for inv_id, grp in df.groupby("inverter_id"):
    minor = grp[grp["alarm_code"].isin(MINOR_ALARM_CODES)]
    print(f"  {inv_id}: {len(minor):,} minor alarm rows")
    print(f"    {grp['alarm_code'].value_counts().sort_index().to_dict()}")

print("\n\n  ac_power stats during daylight:")
for inv_id, grp in df[(df["is_daylight"]==1)].groupby("inverter_id"):
    pw = grp["ac_power"]
    print(f"  {inv_id}: mean={pw.mean():.2f}kW  std={pw.std():.2f}kW  "
          f"min={pw.min():.2f}kW  max={pw.max():.2f}kW")


# ══════════════════════════════════════════════════════════════════════════════
# BUILD DEGRADATION SIGNAL — 3 independent signals, combined with OR
# ══════════════════════════════════════════════════════════════════════════════
divider("Build Degradation Signal (3 Signals Combined)")

"""
SIGNAL A — Throttled Output (limit_percent < 90 during daylight)
  When limit_percent drops below 90, the inverter is being restricted
  by firmware — could be thermal protection, grid compliance, or
  communication with plant controller before a fault.
  This is an early warning the inverter itself is "aware of" a problem.
"""
df["signal_a_throttled"] = (
    (df["limit_percent"] < 90) &
    (df["limit_percent"].notna()) &
    (df["is_daylight"] == 1)
).astype(int)

# ── ──────────────────────────────────────────────────────────────────────────

"""
SIGNAL B — Minor Alarm Clustering (3+ minor alarms within 24 hours)
  A single minor alarm is not meaningful.
  But 3+ minor alarms within 24 hours = escalating instability.
  This is the "alarm_acceleration" concept applied per-inverter.
"""
MINOR_CLUSTER_WINDOW = 24 * ROWS_PER_HOUR   # 24 hours in rows
MINOR_CLUSTER_COUNT  = 3                     # 3+ alarms in 24h = clustering

df["is_minor_alarm"] = df["alarm_code"].isin(MINOR_ALARM_CODES).astype(int)
df["minor_alarm_count_24h"] = df.groupby("inverter_id")["is_minor_alarm"].transform(
    lambda x: x.rolling(MINOR_CLUSTER_WINDOW, min_periods=1).sum()
)
df["signal_b_alarm_cluster"] = (
    df["minor_alarm_count_24h"] >= MINOR_CLUSTER_COUNT
).astype(int)

# ── ──────────────────────────────────────────────────────────────────────────

"""
SIGNAL C — Power Below 2-Sigma of Own Baseline (during daylight)
  Since efficiency is stable, we look at raw power output instead.
  Each inverter has seasonal variation, so we use its own 14-day
  rolling mean ± std as the "expected" range.
  Dropping below mean − 2×std during daylight = underperformance.
"""
POWER_WINDOW_14D = 14 * 24 * ROWS_PER_HOUR

df["power_rolling_mean_14d"] = df.groupby("inverter_id")["ac_power"].transform(
    lambda x: x.rolling(POWER_WINDOW_14D, min_periods=POWER_WINDOW_14D // 4).mean()
)
df["power_rolling_std_14d"] = df.groupby("inverter_id")["ac_power"].transform(
    lambda x: x.rolling(POWER_WINDOW_14D, min_periods=POWER_WINDOW_14D // 4).std()
)

df["signal_c_power_drop"] = (
    (df["ac_power"] < df["power_rolling_mean_14d"] - 2 * df["power_rolling_std_14d"]) &
    (df["is_daylight"] == 1) &
    (df["power_rolling_std_14d"] > 0.5)  # only when baseline is well-established
).astype(int)

# ── ──────────────────────────────────────────────────────────────────────────

# Combined: degrading if ANY signal fires
df["is_degrading"] = (
    (df["signal_a_throttled"] == 1) |
    (df["signal_b_alarm_cluster"] == 1) |
    (df["signal_c_power_drop"] == 1)
).astype(int)

print(f"\n  Signal counts (rows where each signal fires):")
print(f"  {'─'*50}")
for inv_id, grp in df.groupby("inverter_id"):
    sa = grp["signal_a_throttled"].sum()
    sb = grp["signal_b_alarm_cluster"].sum()
    sc = grp["signal_c_power_drop"].sum()
    combined = grp["is_degrading"].sum()
    total = len(grp)
    print(f"\n  {inv_id}:")
    print(f"    Signal A (throttled limit_percent<90) : {sa:>6,}  ({sa/total*100:.2f}%)")
    print(f"    Signal B (minor alarm cluster 3+/24h) : {sb:>6,}  ({sb/total*100:.2f}%)")
    print(f"    Signal C (power < mean-2σ daylight)   : {sc:>6,}  ({sc/total*100:.2f}%)")
    print(f"    COMBINED (any signal)                 : {combined:>6,}  ({combined/total*100:.2f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# BUILD FINAL 3-CLASS LABEL
# ══════════════════════════════════════════════════════════════════════════════
divider("Build Final 3-Class Label")

def forward_rolling_any(series, window):
    """1 if any positive event occurs in the NEXT `window` rows."""
    reversed_sum = series[::-1].rolling(window=window, min_periods=1).sum()[::-1]
    return (reversed_sum > 0).astype(int)

# Shutdown risk (Class 2) — alarm 534/556 within 7 days
df["is_major_fault"] = df["alarm_code"].isin(MAJOR_ALARM_CODES).astype(int)
df["shutdown_risk_fwd"] = df.groupby("inverter_id")["is_major_fault"].transform(
    lambda x: forward_rolling_any(x, LABEL_WINDOW)
)

# Degradation risk (Class 1) — any degradation signal within 7 days
df["degradation_risk_fwd"] = df.groupby("inverter_id")["is_degrading"].transform(
    lambda x: forward_rolling_any(x, LABEL_WINDOW)
)

# Final label: shutdown overrides degradation overrides no-risk
df["risk_class"] = 0
df.loc[df["degradation_risk_fwd"] == 1, "risk_class"] = 1
df.loc[df["shutdown_risk_fwd"]    == 1, "risk_class"] = 2  # always wins

# Binary
df["failure_label"] = (df["shutdown_risk_fwd"] == 1).astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# PRINT FINAL DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
divider("Final Label Distribution")

CLASS_LABELS = {0: "No Risk        ", 1: "Degradation Risk", 2: "Shutdown Risk  "}
total = len(df)

print(f"\n  MULTI-CLASS (risk_class):")
print(f"  {'─'*58}")
vc3 = df["risk_class"].value_counts().sort_index()
for cls, cnt in vc3.items():
    pct = cnt / total * 100
    bar = "█" * int(pct / 2)
    print(f"  Class {cls} | {CLASS_LABELS[cls]} | {cnt:7,} ({pct:5.1f}%)  {bar}")

print(f"\n  BINARY (failure_label):")
print(f"  {'─'*58}")
vc2 = df["failure_label"].value_counts().sort_index()
for lbl, cnt in vc2.items():
    pct = cnt / total * 100
    bar = "█" * int(pct / 2)
    print(f"  Label {lbl} | {cnt:7,} rows ({pct:5.1f}%)  {bar}")

print(f"\n  Per-inverter:")
print(f"  {'─'*58}")
for inv_id, grp in df.groupby("inverter_id"):
    g = len(grp)
    s0 = (grp["risk_class"]==0).sum()
    s1 = (grp["risk_class"]==1).sum()
    s2 = (grp["risk_class"]==2).sum()
    print(f"\n  {inv_id}  ({g:,} rows):")
    print(f"    Class 0 No Risk       : {s0:>7,}  ({s0/g*100:.1f}%)")
    print(f"    Class 1 Degradation   : {s1:>7,}  ({s1/g*100:.1f}%)")
    print(f"    Class 2 Shutdown Risk : {s2:>7,}  ({s2/g*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# PATCH INTO FINAL FEATURE CSV
# ══════════════════════════════════════════════════════════════════════════════
divider("Saving Final Feature CSV")

try:
    df_ml = pd.read_csv("inverter_features_FINAL.csv", parse_dates=["timestamp"])

    # Drop all old label columns
    drop_cols = ["failure_label", "risk_class", "shutdown_risk_fwd",
                 "degradation_risk_fwd", "is_efficiency_drop", "efficiency_gap",
                 "is_derated", "derated_persistent", "shutdown_risk", "degradation_risk"]
    df_ml = df_ml.drop(columns=[c for c in drop_cols if c in df_ml.columns])

    # New columns to merge
    new_cols = df[[
        "inverter_id", "ts",
        "failure_label", "risk_class",
        "signal_a_throttled", "signal_b_alarm_cluster", "signal_c_power_drop",
        "is_degrading", "minor_alarm_count_24h",
        "power_rolling_mean_14d", "power_rolling_std_14d",
    ]].rename(columns={"ts": "timestamp"})

    df_ml = df_ml.merge(new_cols, on=["inverter_id", "timestamp"], how="left")

    # Fillna for safety
    for col in ["failure_label", "risk_class", "signal_a_throttled",
                "signal_b_alarm_cluster", "signal_c_power_drop", "is_degrading"]:
        df_ml[col] = df_ml[col].fillna(0).astype(int)

    df_ml.to_csv("inverter_features_FINAL_v2.csv", index=False)

    print(f"\n  Feature matrix : {df_ml.shape}")
    print(f"  New features added : signal_a_throttled, signal_b_alarm_cluster,")
    print(f"                       signal_c_power_drop, minor_alarm_count_24h")

    print(f"\n  ✅ FINAL label distribution:")
    vc = df_ml["risk_class"].value_counts().sort_index()
    for cls, cnt in vc.items():
        pct = cnt / len(df_ml) * 100
        bar = "█" * int(pct / 2)
        print(f"    Class {cls}: {cnt:7,} ({pct:5.1f}%)  {bar}")

    print(f"\n  ✅ Saved → inverter_features_FINAL_v2.csv")
    print(f"  ✅ Hand THIS file to the ML team — this is the final version")

except FileNotFoundError:
    print(f"\n  ⚠️  inverter_features_FINAL.csv not found.")
    print(f"  Run scripts in this order:")
    print(f"    1. feature_engineering_aubergine.py")
    print(f"    2. label_fix_patch.py")
    print(f"    3. multiclass_label_fix.py")
    print(f"    4. degradation_fix_final.py  ← this script")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL HANDOFF
# ══════════════════════════════════════════════════════════════════════════════
print(f"""
{'='*65}
  COMPLETE FEATURE ENGINEERING HANDOFF
{'='*65}

  FILE: inverter_features_FINAL_v2.csv

  LABEL COLUMNS:
    failure_label  → binary (0/1) for minimum requirement
    risk_class     → 3-class (0/1/2) for bonus requirement

  DEGRADATION SIGNALS (also usable as features):
    signal_a_throttled        → limit_percent < 90 during daylight
    signal_b_alarm_cluster    → 3+ minor alarms in 24 hours
    signal_c_power_drop       → power < mean−2σ during daylight
    minor_alarm_count_24h     → rolling alarm count (continuous)

  ML TEAM PARAMETERS:
    Binary XGBoost:
      scale_pos_weight = n_class0 / n_class2
      eval_metric = ['auc', 'aucpr', 'logloss']

    Multi-class XGBoost:
      objective = 'multi:softprob'
      num_class = 3
      eval_metric = 'mlogloss'

  TOP SHAP FEATURES TO EXPECT:
    major_alarm_count_7d      → fault history
    signal_a_throttled        → inverter self-limiting
    minor_alarm_count_24h     → recent alarm burst
    temp_delta_7d             → thermal drift
    voltage_delta_7d          → DC voltage sag
    signal_c_power_drop       → power underperformance
    alarm_acceleration        → alarm trend

  YOUR WORK IS DONE — hand inverter_features_FINAL_v2.csv
  to the ML team along with the SHAP feature list above.
{'='*65}
""")
