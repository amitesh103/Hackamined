"""
╔══════════════════════════════════════════════════════════════════════════╗
║   FINAL PATCH — 3 Issues Fixed                                          ║
║                                                                          ║
║   Issue 1: INV-AC12 ac_power max = 1,106,247 kW (sensor corruption)    ║
║   Issue 2: Plant 2 fault codes 548/558/563 were missing from           ║
║            MAJOR_ALARM_CODES → 4,713 fault rows mislabelled as Class 0 ║
║   Issue 3: Row count mismatch (duplicate timestamps from merge)         ║
║                                                                          ║
║   HOW TO RUN:                                                            ║
║     python final_patch.py                                               ║
║     → Produces: inverter_features_READY.csv  (give this to ML team)    ║
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
    # Plant 2 — update these paths to your actual Plant 2 filenames
    "INV-AC12": r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 2-20260305T111818Z-3-001\Plant 2\Copy of 80-1F-12-0F-AC-12.raws.csv",
    "INV-ACBB": r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 2-20260305T111818Z-3-001\Plant 2\Copy of 80-1F-12-0F-AC-BB.raws.csv",
}

# PLANT ID mapping — so ML model knows which plant each inverter belongs to
PLANT_MAP = {
    "INV-6E":   "Plant3",
    "INV-69":   "Plant3",
    "INV-AC12": "Plant2",
    "INV-ACBB": "Plant2",
}

# ── FIXED: All major fault codes across BOTH plants ───────────────────────────
# Plant 3: 534, 556
# Plant 2: 548, 558, 563  ← these were missing before!
MAJOR_ALARM_CODES = [534, 548, 556, 558, 563]
MINOR_ALARM_CODES = [2, 4, 8, 10, 12, 39]   # also added 2 and 4 seen in Plant 2

# ── Physical limits per inverter model ───────────────────────────────────────
# Plant 3 inverters: PVSCL60E rated ~60kW → cap at 80kW (some headroom)
# Plant 2 inverters: unknown model, max seen legitimately ~110kW → cap at 150kW
POWER_CAP = {
    "INV-6E":   80.0,
    "INV-69":   80.0,
    "INV-AC12": 150.0,  # ← fixes the 1,106,247 kW corruption
    "INV-ACBB": 150.0,
}

ROWS_PER_HOUR          = 12
PREDICTION_WINDOW_DAYS = 7
LABEL_WINDOW           = PREDICTION_WINDOW_DAYS * 24 * ROWS_PER_HOUR

def divider(title):
    print(f"\n{'='*65}\n  {title}\n{'='*65}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — LOAD, CLEAN, LABEL (rebuilt cleanly, no patch-on-patch)
# ══════════════════════════════════════════════════════════════════════════════
divider("Stage 1: Load & Clean All 4 Inverters")

dfs = []
for inv_id, filepath in FILES.items():
    df = pd.read_csv(filepath, low_memory=False)
    df["inverter_id"] = inv_id
    df["plant_id"]    = PLANT_MAP[inv_id]

    # Parse timestamp
    df["ts"] = pd.to_datetime(
        df["timestampDate"], utc=True, errors="coerce"
    ).dt.tz_convert(None)
    df = df.dropna(subset=["ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    # Rename core columns
    rename_map = {
        "inverters[0].alarm_code":    "alarm_code",
        "inverters[0].op_state":      "op_state",
        "inverters[0].power":         "ac_power",
        "inverters[0].pv1_power":     "dc_power_raw",
        "inverters[0].pv1_voltage":   "dc_voltage",
        "inverters[0].pv1_current":   "dc_current",
        "inverters[0].temp":          "inverter_temp",
        "inverters[0].limit_percent": "limit_percent",
        "meters[0].v_r":              "grid_v_r",
        "meters[0].v_y":              "grid_v_y",
        "meters[0].v_b":              "grid_v_b",
        "meters[0].freq":             "grid_freq",
        "meters[0].pf":               "power_factor",
        "meters[0].meter_active_power": "grid_active_power",
        "inverters[0].kwh_today":     "kwh_today",
        "inverters[0].kwh_total":     "kwh_total",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Force numeric
    for col in ["ac_power", "dc_power_raw", "alarm_code", "limit_percent",
                "dc_voltage", "dc_current", "inverter_temp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── FIX 1: Clamp ac_power to physical maximum ────────────────────────────
    cap = POWER_CAP[inv_id]
    raw_max = df["ac_power"].max() if "ac_power" in df.columns else 0
    df["ac_power"] = df["ac_power"].clip(lower=0, upper=cap)
    if "dc_power_raw" in df.columns:
        df["dc_power_raw"] = df["dc_power_raw"].clip(lower=0, upper=cap * 1.1)

    corrupted = (df.get("ac_power", pd.Series()) > cap).sum()
    print(f"  {inv_id}: {len(df):,} rows | ac_power capped at {cap}kW "
          f"| raw max was {raw_max:.1f}kW")

    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df = df.sort_values(["inverter_id", "ts"]).reset_index(drop=True)
print(f"\n  Combined: {len(df):,} rows across {df['inverter_id'].nunique()} inverters")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — VERIFY ALARM CODES (show all alarm codes per inverter)
# ══════════════════════════════════════════════════════════════════════════════
divider("Stage 2: Alarm Code Verification")

print("\n  Alarm code distribution per inverter:")
for inv_id, grp in df.groupby("inverter_id"):
    if "alarm_code" not in grp.columns:
        print(f"  {inv_id}: NO alarm_code column")
        continue
    vc = grp["alarm_code"].value_counts().sort_index()
    major_count = grp["alarm_code"].isin(MAJOR_ALARM_CODES).sum()
    print(f"\n  {inv_id} — major faults (codes {MAJOR_ALARM_CODES}): {major_count:,} rows")
    for code, cnt in vc.items():
        marker = " ⚠️  MAJOR FAULT" if code in MAJOR_ALARM_CODES else \
                 " ⚡ minor"        if code in MINOR_ALARM_CODES else \
                 " 💤 standby"      if code == 100 else \
                 " ✅ normal"       if code == 0 else ""
        print(f"    code {int(code):>4}: {cnt:>7,} rows{marker}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
divider("Stage 3: Feature Engineering")

df["is_daylight"] = (df["ac_power"] > 0.5).astype(int)
df["hour"]        = df["ts"].dt.hour
df["month"]       = df["ts"].dt.month
df["day_of_week"] = df["ts"].dt.dayofweek
df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)

# Plant ID encoded (so model knows which plant)
df["plant_encoded"] = df["plant_id"].map({"Plant2": 2, "Plant3": 3})

# Conversion efficiency (only where DC data exists)
if "dc_power_raw" in df.columns:
    df["conversion_efficiency"] = np.where(
        df["dc_power_raw"] > 0.5,
        (df["ac_power"] / df["dc_power_raw"]).clip(0, 1.05) * 100,
        np.nan
    )

# Op state severity
OP_SEVERITY = {0: 0, 3: 0, 5: 1, 4: 2, 7: 3, 8: 5, -1: 0}
df["op_state_severity"] = df["op_state"].map(OP_SEVERITY).fillna(1) \
    if "op_state" in df.columns else 0

# Grid voltage imbalance
v_cols = [c for c in ["grid_v_r", "grid_v_y", "grid_v_b"] if c in df.columns]
if len(v_cols) == 3:
    for c in v_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["grid_v_mean"]      = df[v_cols].mean(axis=1)
    df["grid_v_imbalance"] = df[v_cols].std(axis=1) / df["grid_v_mean"].clip(lower=1) * 100

# Rolling features (per inverter — no cross-inverter bleeding)
ROLL_COLS = ["ac_power", "inverter_temp", "op_state_severity"]
if "dc_voltage" in df.columns:
    ROLL_COLS += ["dc_voltage"]
if "conversion_efficiency" in df.columns:
    ROLL_COLS += ["conversion_efficiency"]

WINDOWS = {"3d": 3*24*ROWS_PER_HOUR, "7d": 7*24*ROWS_PER_HOUR, "14d": 14*24*ROWS_PER_HOUR}

for col in ROLL_COLS:
    if col not in df.columns:
        continue
    grp = df.groupby("inverter_id")[col]
    for wname, wsize in WINDOWS.items():
        df[f"{col}_mean_{wname}"] = grp.transform(
            lambda x: x.rolling(wsize, min_periods=wsize//4).mean())
        df[f"{col}_std_{wname}"]  = grp.transform(
            lambda x: x.rolling(wsize, min_periods=wsize//4).std())

# Delta features
df["power_delta_7d"] = df["ac_power"] - df.get("ac_power_mean_7d", np.nan)
if "inverter_temp" in df.columns:
    df["temp_delta_7d"] = df["inverter_temp"] - df.get("inverter_temp_mean_7d", np.nan)
if "dc_voltage" in df.columns:
    df["voltage_delta_7d"] = df["dc_voltage"] - df.get("dc_voltage_mean_7d", np.nan)

# Lag features (1d, 7d)
for col in ["ac_power", "inverter_temp"]:
    if col not in df.columns:
        continue
    for lag_name, lag_rows in [("1d", 24*ROWS_PER_HOUR), ("7d", 7*24*ROWS_PER_HOUR)]:
        df[f"{col}_change_{lag_name}"] = df[col] - df.groupby("inverter_id")[col].transform(
            lambda x: x.shift(lag_rows))

print(f"  Features engineered ✅")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — LABEL ENGINEERING (with all fault codes)
# ══════════════════════════════════════════════════════════════════════════════
divider("Stage 4: Label Engineering (All Fault Codes)")

def forward_rolling_any(series, window):
    reversed_sum = series[::-1].rolling(window=window, min_periods=1).sum()[::-1]
    return (reversed_sum > 0).astype(int)

# Fault events — now includes Plant 2 codes 548/558/563
if "alarm_code" in df.columns:
    df["is_major_fault"] = df["alarm_code"].isin(MAJOR_ALARM_CODES).astype(int)
    df["is_minor_alarm"] = df["alarm_code"].isin(MINOR_ALARM_CODES).astype(int)
else:
    df["is_major_fault"] = 0
    df["is_minor_alarm"] = 0

# Alarm rolling counts
for wname, wsize in [("24h", 24*ROWS_PER_HOUR), ("3d", 3*24*ROWS_PER_HOUR),
                     ("7d", 7*24*ROWS_PER_HOUR)]:
    df[f"minor_alarm_count_{wname}"] = df.groupby("inverter_id")["is_minor_alarm"].transform(
        lambda x: x.rolling(wsize, min_periods=1).sum())
    df[f"major_alarm_count_{wname}"] = df.groupby("inverter_id")["is_major_fault"].transform(
        lambda x: x.rolling(wsize, min_periods=1).sum())

# Alarm acceleration
df["alarm_acceleration"] = np.where(
    df["minor_alarm_count_7d"] > 0,
    df["minor_alarm_count_3d"] / (df["minor_alarm_count_7d"] / 2.33 + 1e-6), 0)

# Degradation signals
df["signal_a_throttled"] = (
    df.get("limit_percent", pd.Series(100, index=df.index)).lt(90) &
    df.get("limit_percent", pd.Series(np.nan, index=df.index)).notna() &
    (df["is_daylight"] == 1)
).astype(int)

df["signal_b_alarm_cluster"] = (df["minor_alarm_count_24h"] >= 3).astype(int)

df["power_rolling_mean_14d"] = df.groupby("inverter_id")["ac_power"].transform(
    lambda x: x.rolling(14*24*ROWS_PER_HOUR, min_periods=14*24*ROWS_PER_HOUR//4).mean())
df["power_rolling_std_14d"] = df.groupby("inverter_id")["ac_power"].transform(
    lambda x: x.rolling(14*24*ROWS_PER_HOUR, min_periods=14*24*ROWS_PER_HOUR//4).std())
df["signal_c_power_drop"] = (
    (df["ac_power"] < df["power_rolling_mean_14d"] - 2*df["power_rolling_std_14d"]) &
    (df["is_daylight"] == 1) &
    (df["power_rolling_std_14d"] > 0.5)
).astype(int)

df["is_degrading"] = (
    (df["signal_a_throttled"] == 1) |
    (df["signal_b_alarm_cluster"] == 1) |
    (df["signal_c_power_drop"] == 1)
).astype(int)

# Forward labels
df["shutdown_risk_fwd"] = df.groupby("inverter_id")["is_major_fault"].transform(
    lambda x: forward_rolling_any(x, LABEL_WINDOW))
df["degradation_risk_fwd"] = df.groupby("inverter_id")["is_degrading"].transform(
    lambda x: forward_rolling_any(x, LABEL_WINDOW))

# Final labels
df["risk_class"]    = 0
df.loc[df["degradation_risk_fwd"] == 1, "risk_class"] = 1
df.loc[df["shutdown_risk_fwd"]    == 1, "risk_class"] = 2
df["failure_label"] = (df["shutdown_risk_fwd"] == 1).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — FIX 3: Deduplicate rows (fix the row count mismatch)
# ══════════════════════════════════════════════════════════════════════════════
divider("Stage 5: Deduplication")

before = len(df)
df = df.drop_duplicates(subset=["inverter_id", "ts"]).reset_index(drop=True)
after = len(df)
print(f"  Removed {before - after:,} duplicate rows ({before:,} → {after:,})")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — FINAL DISTRIBUTION & EXPORT
# ══════════════════════════════════════════════════════════════════════════════
divider("Stage 6: Final Label Distribution")

CLASS_LABELS = {0: "No Risk        ", 1: "Degradation Risk", 2: "Shutdown Risk  "}
total = len(df)

print(f"\n  MULTI-CLASS (risk_class) — all 4 inverters:")
print(f"  {'─'*60}")
for cls, cnt in df["risk_class"].value_counts().sort_index().items():
    pct = cnt / total * 100
    bar = "█" * int(pct / 2)
    print(f"  Class {cls} | {CLASS_LABELS[cls]} | {cnt:7,} ({pct:5.1f}%)  {bar}")

print(f"\n  Per-inverter breakdown:")
print(f"  {'─'*60}")
for inv_id, grp in df.groupby("inverter_id"):
    g  = len(grp)
    s0 = (grp["risk_class"]==0).sum()
    s1 = (grp["risk_class"]==1).sum()
    s2 = (grp["risk_class"]==2).sum()
    plant = PLANT_MAP[inv_id]
    major = grp["is_major_fault"].sum() if "is_major_fault" in grp else 0
    print(f"\n  {inv_id} ({plant}, {g:,} rows) — {major} actual fault events:")
    print(f"    Class 0 No Risk       : {s0:>7,}  ({s0/g*100:.1f}%)")
    print(f"    Class 1 Degradation   : {s1:>7,}  ({s1/g*100:.1f}%)")
    print(f"    Class 2 Shutdown Risk : {s2:>7,}  ({s2/g*100:.1f}%)")

# Drop rows from before 7-day rolling window is established
df_ml = df.dropna(subset=["ac_power_mean_7d"]).copy()

# Fill remaining NaN
numeric_cols = df_ml.select_dtypes(include=[np.number]).columns
df_ml[numeric_cols] = df_ml[numeric_cols].fillna(0)

# Remove metadata columns not needed by ML
drop_cols = ["ts", "_id", "mac", "createdAt", "timestampDate", "fromServer",
             "dataLoggerModelId", "__v", "alarm_DC_LOSS", "alarm_GRID_FAULT",
             "alarm_LV_FAULT", "alarm_OVT", "batteries", "grid_master",
             "sensors[0].ambient_temp"]
df_ml = df_ml.drop(columns=[c for c in drop_cols if c in df_ml.columns])

df_ml.to_csv("inverter_features_READY.csv", index=False)

divider("FINAL SUMMARY — Hand to ML Team")

n0 = (df_ml["failure_label"]==0).sum()
n1 = (df_ml["failure_label"]==1).sum()
spw = round(n0 / max(n1, 1))

print(f"""
  FILE: inverter_features_READY.csv
  Shape: {df_ml.shape}

  BINARY LABEL (failure_label):
    Class 0 (Safe)          : {n0:,} rows
    Class 1 (Shutdown Risk) : {n1:,} rows
    scale_pos_weight        : {spw}  ← use this in XGBoost

  MULTI-CLASS LABEL (risk_class):
    Class 0 / 1 / 2 shown above

  KEY FIXES IN THIS VERSION vs previous:
    ✅ Plant 2 fault codes 548/558/563 now included
    ✅ INV-AC12 power capped at 150kW (was 1,106,247 kW)
    ✅ Duplicate rows removed
    ✅ plant_encoded feature added (2=Plant2, 3=Plant3)
    ✅ All features rebuilt from raw data (no patch-on-patch)

  XGBoost parameters for ML team:
    Binary  → scale_pos_weight={spw}, eval_metric=['auc','aucpr']
    Multi   → objective='multi:softprob', num_class=3

  ✅ Feature engineering COMPLETE — ready for ML training
""")
