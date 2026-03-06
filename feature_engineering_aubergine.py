"""
╔══════════════════════════════════════════════════════════════════════════╗
║   FEATURE ENGINEERING PIPELINE — Adapted to Aubergine Dataset           ║
║   HACKaMINeD 2026 · Solar Inverter Failure Prediction                   ║
║                                                                          ║
║   FILES EXPECTED (update paths below):                                  ║
║     - Copy of 54-10-EC-8C-14-6E.raws.csv                               ║
║     - Copy of 54-10-EC-8C-14-69.raws.csv                               ║
║                                                                          ║
║   OUTPUT:                                                                ║
║     - inverter_features_final.csv  ← hand this to ML team               ║
║     - label_audit.csv              ← verify failure labels               ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# ✏️  CONFIGURE YOUR FILE PATHS HERE
# ══════════════════════════════════════════════════════════════════════════════

FILES = {
    "INV-6E": r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 3-20260305T111819Z-3-001\Plant 3\Copy of 54-10-EC-8C-14-6E.raws.csv",
    "INV-69": r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 3-20260305T111819Z-3-001\Plant 3\Copy of 54-10-EC-8C-14-69.raws.csv",
    "INV-AC12": r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 2-20260305T111818Z-3-001\Plant 2\Copy of 80-1F-12-0F-AC-12.raws.csv",

    "INV-ACBB": r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 2-20260305T111818Z-3-001\Plant 2\Copy of 80-1F-12-0F-AC-BB.raws.csv"
}

# Prediction window: label = 1 if a major fault happens within next N days
PREDICTION_WINDOW_DAYS = 7

# Rolling windows in hours (data is 5-min intervals = 12 rows/hour)
ROWS_PER_HOUR = 12
WINDOW_3D  = 3  * 24 * ROWS_PER_HOUR   # 1,080 rows
WINDOW_7D  = 7  * 24 * ROWS_PER_HOUR   # 2,016 rows
WINDOW_14D = 14 * 24 * ROWS_PER_HOUR   # 4,032 rows

# ══════════════════════════════════════════════════════════════════════════════
# ALARM CODE & OP STATE REFERENCE
# (decoded from your actual data — share this with your whole team)
# ══════════════════════════════════════════════════════════════════════════════
#
#  inverters[0].alarm_code:
#    0   → No alarm (normal)
#    8   → Minor warning (grid fluctuation)
#    10  → Minor warning
#    12  → Minor warning
#    39  → Minor warning
#    100 → Standby / Nighttime (NOT a fault)
#    534 → ⚠️  MAJOR FAULT — triggers failure label
#    556 → ⚠️  MAJOR FAULT — triggers failure label
#
#  inverters[0].op_state:
#    0  → Off / Nighttime
#    3  → Normal generation
#    4  → Derated output (limited)
#    5  → Starting up
#    7  → Recovering from fault
#    8  → ⚠️  FAULT / SHUTDOWN — triggers failure label
#
MAJOR_ALARM_CODES = [534, 556]
# op_state==8 is nighttime OFF/standby — NOT a fault. Use alarm codes only.
FAULT_OP_STATES   = []  # kept for reference but intentionally empty

def divider(title):
    print(f"\n{'='*65}\n  {title}\n{'='*65}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — LOAD & MERGE BOTH INVERTER FILES
# ══════════════════════════════════════════════════════════════════════════════
divider("STAGE 1: Load & Merge Both Inverter Files")

dfs = []
for inv_id, filepath in FILES.items():
    print(f"\n  Loading {inv_id} from: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)

    # Assign a clean inverter ID (instead of raw MAC address)
    df["inverter_id"] = inv_id

    print(f"    Rows: {len(df):,}  |  Columns: {len(df.columns)}")
    dfs.append(df)

df_raw = pd.concat(dfs, ignore_index=True)
print(f"\n  Combined shape: {df_raw.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — PARSE TIMESTAMPS & SORT
# ══════════════════════════════════════════════════════════════════════════════
divider("STAGE 2: Parse Timestamps & Sort")

# timestampDate is the cleanest column (ISO 8601 format, human readable)
# timestamp is Unix milliseconds — we keep both but use timestampDate as index
df_raw["ts"] = pd.to_datetime(df_raw["timestampDate"], utc=True, errors="coerce")
df_raw["ts"] = df_raw["ts"].dt.tz_convert(None)  # strip UTC tz → naive datetime

# Drop rows where timestamp is unparseable
before = len(df_raw)
df_raw = df_raw.dropna(subset=["ts"])
print(f"  Dropped {before - len(df_raw)} rows with unparseable timestamps")

# Sort by inverter first, then by time — CRITICAL for rolling calculations
df_raw = df_raw.sort_values(["inverter_id", "ts"]).reset_index(drop=True)

print(f"  Date range: {df_raw['ts'].min()} → {df_raw['ts'].max()}")
print(f"  Sampling interval check (should be ~5 min):")
for inv_id, grp in df_raw.groupby("inverter_id"):
    median_gap = grp["ts"].diff().dt.total_seconds().median()
    print(f"    {inv_id}: median gap = {median_gap:.0f} seconds")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — SELECT & RENAME CORE COLUMNS
# (mapped from your actual column names to clean readable names)
# ══════════════════════════════════════════════════════════════════════════════
divider("STAGE 3: Select & Rename Core Columns")

# Columns we keep and their clean names
COLUMN_MAP = {
    "inverter_id":                          "inverter_id",
    "ts":                                   "timestamp",

    # ── Inverter DC side ─────────────────────────────────────────
    "inverters[0].pv1_voltage":             "dc_voltage",       # V
    "inverters[0].pv1_current":             "dc_current",       # A
    "inverters[0].pv1_power":               "dc_power_raw",     # kW input

    # pv2 is all zeros (dead string) — intentionally excluded

    # ── Inverter AC output ────────────────────────────────────────
    "inverters[0].power":                   "ac_power",         # kW output

    # ── Energy counters ───────────────────────────────────────────
    "inverters[0].kwh_total":               "kwh_total",        # cumulative kWh
    "inverters[0].kwh_today":               "kwh_today",        # kWh today
    "inverters[0].kwh_midnight":            "kwh_midnight",     # kWh at last midnight

    # ── Inverter status ───────────────────────────────────────────
    "inverters[0].alarm_code":              "alarm_code",       # see ALARM CODE REFERENCE
    "inverters[0].op_state":                "op_state",         # see OP STATE REFERENCE
    "inverters[0].temp":                    "inverter_temp",    # °C (46% missing)
    "inverters[0].limit_percent":           "limit_percent",    # % of rated power allowed

    # ── Grid / Meter side ─────────────────────────────────────────
    "meters[0].v_r":                        "grid_v_r",         # Phase R voltage (V)
    "meters[0].v_y":                        "grid_v_y",         # Phase Y voltage (V)
    "meters[0].v_b":                        "grid_v_b",         # Phase B voltage (V)
    "meters[0].freq":                       "grid_freq",        # Hz
    "meters[0].pf":                         "power_factor",     # power factor
    "meters[0].meter_active_power":         "grid_active_power",# kW (negative = exporting)
    "meters[0].meter_kwh_total":            "meter_kwh_total",  # cumulative meter kWh

    # ── SMU String currents (individual panel strings) ────────────
    # These are the per-string currents — deviations between strings = panel issues
    "smu[0].string1":  "string1",  "smu[0].string2":  "string2",
    "smu[0].string3":  "string3",  "smu[0].string4":  "string4",
    "smu[0].string5":  "string5",  "smu[0].string6":  "string6",
    "smu[0].string7":  "string7",  "smu[0].string8":  "string8",
    "smu[0].string9":  "string9",  "smu[0].string10": "string10",
    "smu[0].string11": "string11", "smu[0].string12": "string12",
}

# Keep only columns that exist in the dataframe
valid_cols = {k: v for k, v in COLUMN_MAP.items() if k in df_raw.columns}
df = df_raw[list(valid_cols.keys())].rename(columns=valid_cols).copy()

print(f"  Selected {len(valid_cols)} columns → renamed to clean names")
print(f"  Shape: {df.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — DATA CLEANING (fix your specific data quality issues)
# ══════════════════════════════════════════════════════════════════════════════
divider("STAGE 4: Data Cleaning")

# 4a. Fix dc_voltage: file 1 has min=-1.08e8 (massive sensor corruption)
#     Physical range for this inverter model: 0 to ~840V
print("  Fixing dc_voltage sensor corruption...")
bad_voltage = (df["dc_voltage"] < 0) | (df["dc_voltage"] > 1000)
print(f"    Corrupted dc_voltage rows: {bad_voltage.sum():,}")
df["dc_voltage"] = df["dc_voltage"].where(~bad_voltage, np.nan)

# 4b. Clamp all physically impossible negatives
df["dc_current"]  = df["dc_current"].clip(lower=0)
df["dc_power_raw"]= df["dc_power_raw"].clip(lower=0)
df["ac_power"]    = df["ac_power"].clip(lower=0)
df["kwh_today"]   = df["kwh_today"].clip(lower=0)  # was going negative

# 4c. Fix SMU string outliers (string1/4/5 had values up to 2.1e9 in file 2)
#     Physical range for string current: 0 to ~15A
STRING_COLS = [c for c in df.columns if c.startswith("string")]
for col in STRING_COLS:
    if col in df.columns:
        outliers = (df[col] > 20) | (df[col] < 0)
        if outliers.sum() > 0:
            print(f"    Clamping {outliers.sum()} outliers in {col}")
        df[col] = df[col].where(~outliers, np.nan)

# 4d. Convert string columns to numeric (grid_v, freq, pf were stored as strings)
NUMERIC_FORCE = ["grid_v_r", "grid_v_y", "grid_v_b", "grid_freq", "power_factor",
                 "grid_active_power"]
for col in NUMERIC_FORCE:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 4e. Clamp grid voltages to physical range (220–260V for Indian grid)
for v_col in ["grid_v_r", "grid_v_y", "grid_v_b"]:
    if v_col in df.columns:
        df[v_col] = df[v_col].where(
            df[v_col].between(180, 280), np.nan
        )

# 4f. Grid frequency physical range (49–51 Hz)
if "grid_freq" in df.columns:
    df["grid_freq"] = df["grid_freq"].where(
        df["grid_freq"].between(48, 52), np.nan
    )

print(f"  Shape after cleaning: {df.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — ENGINEER THE FAILURE LABEL
# (no pre-existing label — we derive it from alarm_code + op_state)
# ══════════════════════════════════════════════════════════════════════════════
divider("STAGE 5: Engineer the Failure Label")

"""
STRATEGY:
  Step 1: Mark rows where a MAJOR FAULT actually occurs
          (alarm_code in [534, 556] OR op_state == 8)
  Step 2: For each row, look FORWARD 7 days
          If any fault occurs in that window → label = 1 (at risk)
          Otherwise → label = 0 (safe)

WHY THIS MATTERS:
  We want the model to predict BEFORE the failure, not detect it.
  So label = 1 should appear in the 7 days PRECEDING a fault,
  giving operators time to act.
"""

# Step 1: Mark actual fault events
# ONLY alarm_code 534/556 are confirmed hardware faults.
# op_state==8 is nighttime standby — do NOT include it.
df["is_major_fault"] = (
    df["alarm_code"].isin(MAJOR_ALARM_CODES)
).astype(int)

print(f"\n  Total major fault events found:")
for inv_id, grp in df.groupby("inverter_id"):
    n_faults = grp["is_major_fault"].sum()
    total    = len(grp)
    print(f"    {inv_id}: {n_faults:,} fault rows out of {total:,} ({n_faults/total*100:.2f}%)")

# Step 2: Forward-rolling label
# For each row: "will there be a fault in the next PREDICTION_WINDOW_DAYS days?"
LABEL_WINDOW = PREDICTION_WINDOW_DAYS * 24 * ROWS_PER_HOUR  # rows in 7 days

def make_forward_label(series):
    """
    For each position i, returns 1 if any fault occurs in the
    NEXT LABEL_WINDOW rows. Uses reverse rolling for efficiency.
    """
    # Reverse the series, apply rolling sum looking backward,
    # then reverse back → equivalent to forward-looking window
    reversed_sum = series[::-1].rolling(
        window=LABEL_WINDOW, min_periods=1
    ).sum()[::-1]
    return (reversed_sum > 0).astype(int)

df["failure_label"] = df.groupby("inverter_id")["is_major_fault"].transform(
    make_forward_label
)

print(f"\n  Failure label distribution (7-day forward window):")
vc = df["failure_label"].value_counts()
for lbl, cnt in vc.items():
    pct = cnt / len(df) * 100
    print(f"    Label {lbl}: {cnt:,} rows ({pct:.1f}%)")

# Save label audit for team review
label_audit = df[["inverter_id", "timestamp", "alarm_code",
                   "op_state", "is_major_fault", "failure_label"]].copy()
label_audit.to_csv("label_audit.csv", index=False)
print(f"\n  ✅ Label audit saved → label_audit.csv (review this to verify labels)")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — TIME & DAYLIGHT FEATURES
# ══════════════════════════════════════════════════════════════════════════════
divider("STAGE 6: Time & Daylight Features")

df["hour"]       = df["timestamp"].dt.hour
df["month"]      = df["timestamp"].dt.month
df["day_of_week"]= df["timestamp"].dt.dayofweek

# Cyclical encoding: model understands 23:00 and 00:00 are adjacent
df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)

# Daylight flag: inverter is actually producing power
# Using ac_power > 0.5kW as proxy (no irradiance sensor — ambient_temp is always 0)
df["is_daylight"] = (df["ac_power"] > 0.5).astype(int)

print(f"  Daylight rows: {df['is_daylight'].sum():,} / {len(df):,} "
      f"({df['is_daylight'].mean()*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — DOMAIN KPI FEATURES
# ══════════════════════════════════════════════════════════════════════════════
divider("STAGE 7: Domain-Specific KPI Features")

# 7a. Conversion Efficiency: AC out / DC in
#     Healthy = 95–98%. Dropping below 90% is a warning sign.
df["conversion_efficiency"] = np.where(
    df["dc_power_raw"] > 0.5,
    (df["ac_power"] / df["dc_power_raw"]) * 100,
    np.nan
)
df["conversion_efficiency"] = df["conversion_efficiency"].clip(0, 105)

# 7b. Inverter-to-Grid Power Ratio
#     How much of inverter output is reaching the meter?
#     Divergence suggests wiring or meter issues
df["inv_to_grid_ratio"] = np.where(
    df["ac_power"] > 0.5,
    abs(df["grid_active_power"]) / df["ac_power"],
    np.nan
)
df["inv_to_grid_ratio"] = df["inv_to_grid_ratio"].clip(0, 2)

# 7c. Grid Voltage Imbalance
#     Three-phase imbalance = grid quality issue (stresses inverter)
v_cols = ["grid_v_r", "grid_v_y", "grid_v_b"]
existing_v = [c for c in v_cols if c in df.columns]
if len(existing_v) == 3:
    df["grid_v_mean"] = df[existing_v].mean(axis=1)
    df["grid_v_std"]  = df[existing_v].std(axis=1)
    # Imbalance = std/mean × 100 (%) — >2% is concerning
    df["grid_v_imbalance"] = np.where(
        df["grid_v_mean"] > 0,
        (df["grid_v_std"] / df["grid_v_mean"]) * 100,
        np.nan
    )

# 7d. Daily kWh Production Rate
#     kwh_today keeps a running count during the day
#     Flat or dropping kwh_today during daylight = underperformance
df["kwh_today_rate"] = df.groupby(
    [df["inverter_id"], df["timestamp"].dt.date]
)["kwh_today"].transform(lambda x: x.diff().clip(lower=0))

# 7e. Operational State Severity Score
#     Encode op_state as a severity number for the model
OP_STATE_SEVERITY = {0: 0, 3: 0, 5: 1, 4: 2, 7: 3, 8: 5}
df["op_state_severity"] = df["op_state"].map(OP_STATE_SEVERITY).fillna(1)

# 7f. SMU String Imbalance
#     If one string drops while others are normal = partial shading or panel fault
string_cols = [c for c in df.columns if c.startswith("string")]
active_strings = [c for c in string_cols if df[c].max() > 0.5]
if len(active_strings) >= 2:
    df["string_mean"] = df[active_strings].mean(axis=1)
    df["string_std"]  = df[active_strings].std(axis=1)
    df["string_imbalance"] = np.where(
        df["string_mean"] > 0.5,
        df["string_std"] / df["string_mean"],
        np.nan
    )
    print(f"  Active string cols: {active_strings}")
    print(f"  String imbalance stats:")
    print(f"    mean={df['string_imbalance'].mean():.3f}  "
          f"max={df['string_imbalance'].max():.3f}")

# 7g. Inverter Temperature Imputation
#     46% of inverter_temp is missing — impute using ac_power proxy
#     Higher power generation = higher temp. Use daylight period mean as fallback.
df["inverter_temp_imputed"] = df.groupby("inverter_id")["inverter_temp"].transform(
    lambda x: x.fillna(x.median())
)
# Flag imputed rows so the ML model can optionally use this as a feature too
df["temp_was_imputed"] = df["inverter_temp"].isna().astype(int)

print(f"\n  Created KPI features: conversion_efficiency, inv_to_grid_ratio,")
print(f"  grid_v_imbalance, kwh_today_rate, op_state_severity, string_imbalance")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8 — ROLLING WINDOW FEATURES
# (5-min data: 1 hour = 12 rows, 1 day = 288 rows)
# ══════════════════════════════════════════════════════════════════════════════
divider("STAGE 8: Rolling Window Features (3d / 7d / 14d)")

ROLLING_TARGETS = {
    "dc_voltage":             ["mean", "std", "min"],
    "ac_power":               ["mean", "std", "min"],
    "conversion_efficiency":  ["mean", "std"],
    "inverter_temp_imputed":  ["mean", "std", "max"],
    "grid_v_imbalance":       ["mean", "max"],
    "op_state_severity":      ["mean", "max"],
    "string_imbalance":       ["mean", "max"],
}

WINDOWS = {
    "3d":  WINDOW_3D,
    "7d":  WINDOW_7D,
    "14d": WINDOW_14D,
}

roll_count = 0
for col, stats in ROLLING_TARGETS.items():
    if col not in df.columns:
        continue
    grp = df.groupby("inverter_id")[col]
    for win_name, win_size in WINDOWS.items():
        for stat in stats:
            feat = f"{col}_{stat}_{win_name}"
            if stat == "mean":
                df[feat] = grp.transform(
                    lambda x: x.rolling(win_size, min_periods=win_size//4).mean())
            elif stat == "std":
                df[feat] = grp.transform(
                    lambda x: x.rolling(win_size, min_periods=win_size//4).std())
            elif stat == "min":
                df[feat] = grp.transform(
                    lambda x: x.rolling(win_size, min_periods=win_size//4).min())
            elif stat == "max":
                df[feat] = grp.transform(
                    lambda x: x.rolling(win_size, min_periods=win_size//4).max())
            roll_count += 1

# Key delta features: current value vs rolling average
# These are typically the top SHAP features
df["temp_delta_7d"]       = df["inverter_temp_imputed"] - df["inverter_temp_imputed_mean_7d"]
df["voltage_delta_7d"]    = df["dc_voltage"]            - df["dc_voltage_mean_7d"]
df["efficiency_delta_7d"] = df["conversion_efficiency"] - df["conversion_efficiency_mean_7d"]
df["power_delta_7d"]      = df["ac_power"]              - df["ac_power_mean_7d"]

print(f"  Created {roll_count} rolling features + 4 delta features")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 9 — LAG FEATURES (backward time-shifted signals)
# ══════════════════════════════════════════════════════════════════════════════
divider("STAGE 9: Lag Features")

LAG_COLS = ["dc_voltage", "ac_power", "conversion_efficiency",
            "inverter_temp_imputed", "grid_v_imbalance"]

# Lags: 1 day back, 3 days back, 7 days back
LAG_SHIFTS = {
    "1d": 1  * 24 * ROWS_PER_HOUR,
    "3d": 3  * 24 * ROWS_PER_HOUR,
    "7d": 7  * 24 * ROWS_PER_HOUR,
}

lag_count = 0
for col in LAG_COLS:
    if col not in df.columns:
        continue
    for lag_name, shift in LAG_SHIFTS.items():
        # Raw lag value
        lag_feat    = f"{col}_lag_{lag_name}"
        df[lag_feat] = df.groupby("inverter_id")[col].transform(
            lambda x: x.shift(shift)
        )
        # Change = current − lagged (captures drift direction)
        change_feat = f"{col}_change_{lag_name}"
        df[change_feat] = df[col] - df[lag_feat]
        lag_count += 2

print(f"  Created {lag_count} lag + change features")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 10 — ALARM FEATURES
# ══════════════════════════════════════════════════════════════════════════════
divider("STAGE 10: Alarm-Based Features")

# Flag each alarm severity level separately
df["is_minor_alarm"] = df["alarm_code"].isin([8, 10, 12, 39]).astype(int)
df["is_major_alarm"] = df["alarm_code"].isin(MAJOR_ALARM_CODES).astype(int)
df["is_standby"]     = (df["alarm_code"] == 100).astype(int)

# Rolling alarm counts (how many alarms in the last 3/7 days?)
for win_name, win_size in [("3d", WINDOW_3D), ("7d", WINDOW_7D)]:
    df[f"minor_alarm_count_{win_name}"] = df.groupby("inverter_id")["is_minor_alarm"].transform(
        lambda x: x.rolling(win_size, min_periods=1).sum()
    )
    df[f"major_alarm_count_{win_name}"] = df.groupby("inverter_id")["is_major_alarm"].transform(
        lambda x: x.rolling(win_size, min_periods=1).sum()
    )

# Alarm acceleration: 3-day rate vs 7-day rate
# ratio > 1 means alarms are increasing in frequency (bad sign!)
df["alarm_acceleration"] = np.where(
    df["minor_alarm_count_7d"] > 0,
    df["minor_alarm_count_3d"] / (df["minor_alarm_count_7d"] / 2.33 + 1e-6),
    0
)  # normalize 7d to same 3-day-equivalent scale

# Hours since last alarm (recency)
def rows_since_last_event(series):
    result = np.full(len(series), np.nan)
    last_idx = -1
    for i, val in enumerate(series):
        if val == 1:
            last_idx = i
        if last_idx >= 0:
            result[i] = (i - last_idx) / ROWS_PER_HOUR  # convert rows → hours
    return result

df["hours_since_minor_alarm"] = df.groupby("inverter_id")["is_minor_alarm"].transform(
    rows_since_last_event
)

print(f"  Alarm features created: minor/major counts (3d/7d), acceleration, recency")
print(f"\n  Major alarm distribution:")
for inv_id, grp in df.groupby("inverter_id"):
    mc = grp["is_major_alarm"].sum()
    print(f"    {inv_id}: {mc:,} major alarm rows")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 11 — FINAL FEATURE SELECTION & EXPORT
# ══════════════════════════════════════════════════════════════════════════════
divider("STAGE 11: Final Feature Selection & Export")

FINAL_FEATURES = [
    # ── Raw telemetry (cleaned) ────────────────────────────────────
    "dc_voltage", "dc_current", "dc_power_raw",
    "ac_power", "inverter_temp_imputed", "temp_was_imputed",
    "limit_percent",

    # ── Grid readings ──────────────────────────────────────────────
    "grid_v_r", "grid_v_y", "grid_v_b", "grid_freq", "power_factor",
    "grid_active_power",

    # ── Time features ──────────────────────────────────────────────
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_week", "is_daylight",

    # ── Engineered KPIs ────────────────────────────────────────────
    "conversion_efficiency", "inv_to_grid_ratio",
    "grid_v_imbalance", "grid_v_std",
    "kwh_today_rate", "op_state_severity",
    "string_imbalance", "string_std",

    # ── Delta features (current vs rolling average) ────────────────
    "temp_delta_7d", "voltage_delta_7d",
    "efficiency_delta_7d", "power_delta_7d",

    # ── Rolling means ──────────────────────────────────────────────
    "dc_voltage_mean_3d", "dc_voltage_mean_7d",
    "ac_power_mean_3d", "ac_power_mean_7d",
    "conversion_efficiency_mean_7d",
    "inverter_temp_imputed_mean_3d", "inverter_temp_imputed_mean_7d",
    "op_state_severity_mean_7d",

    # ── Rolling std (instability signals) ─────────────────────────
    "dc_voltage_std_7d",
    "ac_power_std_7d",
    "conversion_efficiency_std_7d",
    "inverter_temp_imputed_std_7d",

    # ── Lag + change features ──────────────────────────────────────
    "dc_voltage_change_1d", "dc_voltage_change_7d",
    "ac_power_change_1d", "ac_power_change_7d",
    "conversion_efficiency_change_7d",
    "inverter_temp_imputed_change_7d",

    # ── Alarm features ─────────────────────────────────────────────
    "is_minor_alarm", "is_major_alarm",
    "minor_alarm_count_3d", "minor_alarm_count_7d",
    "major_alarm_count_3d", "major_alarm_count_7d",
    "alarm_acceleration", "hours_since_minor_alarm",
]

# Keep only features that exist
FINAL_FEATURES = [f for f in FINAL_FEATURES if f in df.columns]

# Build final ML dataframe
df_ml = df[["timestamp", "inverter_id", "failure_label", "is_major_fault"]
           + FINAL_FEATURES].copy()

# Drop rows where most rolling features are still NaN (first 7 days of data)
df_ml = df_ml.dropna(subset=["dc_voltage_mean_7d"])

# Fill remaining NaN with per-inverter median
for col in FINAL_FEATURES:
    if df_ml[col].isnull().any():
        df_ml[col] = df_ml.groupby("inverter_id")[col].transform(
            lambda x: x.fillna(x.median())
        )
    # Final fallback: fill with 0 if still NaN
    df_ml[col] = df_ml[col].fillna(0)

# Export
df_ml.to_csv("inverter_features_final.csv", index=False)

# ── Final Summary ─────────────────────────────────────────────────────────────
print(f"\n  Final feature matrix    : {df_ml.shape}")
print(f"  Total ML features       : {len(FINAL_FEATURES)}")

print(f"\n  Label distribution:")
for inv_id, grp in df_ml.groupby("inverter_id"):
    pos = grp["failure_label"].sum()
    neg = (grp["failure_label"] == 0).sum()
    print(f"    {inv_id}: {pos:,} at-risk | {neg:,} safe | "
          f"imbalance ratio = 1:{neg//max(pos,1)}")

print(f"\n  Feature breakdown:")
cats = {
    "Raw telemetry":    [f for f in FINAL_FEATURES if any(x in f for x in ["dc_", "ac_power", "temp", "limit", "grid_v_r", "grid_v_y", "grid_v_b", "grid_freq", "power_factor", "grid_active"])],
    "Time features":    [f for f in FINAL_FEATURES if any(x in f for x in ["sin", "cos", "day_of", "is_day"])],
    "KPI engineered":   [f for f in FINAL_FEATURES if any(x in f for x in ["efficiency", "ratio", "imbalance", "rate", "severity", "string_"])],
    "Rolling stats":    [f for f in FINAL_FEATURES if any(x in f for x in ["_mean_", "_std_", "_min_", "_max_"])],
    "Delta features":   [f for f in FINAL_FEATURES if "_delta_" in f],
    "Lag/change":       [f for f in FINAL_FEATURES if "_lag_" in f or "_change_" in f],
    "Alarm features":   [f for f in FINAL_FEATURES if "alarm" in f or "is_major" in f or "is_minor" in f],
}
total = 0
for cat, feats in cats.items():
    print(f"    [{len(feats):2d}] {cat}")
    total += len(feats)
print(f"\n  ✅ TOTAL: {total} features")
print(f"  ✅ Output → inverter_features_final.csv")
print(f"  ✅ Labels → label_audit.csv")
print(f"\n{'='*65}")
print(f"  HANDOFF NOTE FOR ML TEAM:")
print(f"  Top features to expect in SHAP (domain knowledge):")
print(f"    1. efficiency_delta_7d    — efficiency drop trend")
print(f"    2. temp_delta_7d          — thermal anomaly")
print(f"    3. major_alarm_count_7d   — fault history")
print(f"    4. voltage_delta_7d       — DC voltage sag")
print(f"    5. op_state_severity_mean — operational stress")
print(f"    6. alarm_acceleration     — alarm frequency trend")
print(f"    7. string_imbalance       — panel-level anomaly")
print(f"{'='*65}\n")