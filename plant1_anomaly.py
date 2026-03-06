"""
╔══════════════════════════════════════════════════════════════════════════╗
║   PLANT 1 — Feature Engineering + Anomaly Detection                     ║
║   HACKaMINeD 2026 · Solar Inverter Failure Prediction                   ║
║                                                                          ║
║   WHY ANOMALY DETECTION FOR PLANT 1:                                    ║
║     - No alarm_code column → no ground truth fault labels               ║
║     - op_state only has values {0, -1} → no fault state transitions     ║
║     - IsolationForest finds inverters behaving "differently" from        ║
║       normal without needing historical fault labels                    ║
║                                                                          ║
║   PREREQUISITES:                                                         ║
║     pip install pandas numpy scikit-learn                               ║
║                                                                          ║
║   HOW TO RUN:                                                            ║
║     1. Update PLANT1_FILES below with your actual file paths            ║
║     2. python plant1_anomaly.py                                         ║
║     3. Output: plant1_anomaly_scores.csv                                ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# ✏️  CONFIGURE — update these paths
# ══════════════════════════════════════════════════════════════════════════════

PLANT1_FILES = [
    r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 1-20260305T111817Z-3-001\Plant 1\Copy of ICR2-LT1-Celestical-10000.73.raws.csv",   # ← update
    r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 1-20260305T111817Z-3-001\Plant 1\Copy of ICR2-LT2-Celestical-10000.73.raws.csv",   # ← update (if 2 files)
]

# Number of inverters in Plant 1 (from column headers: inverters[0] to [11])
N_INVERTERS = 12

# Rolling windows (5-min data = 12 rows/hour)
ROWS_PER_HOUR = 12
WINDOWS = {
    "3d":  3  * 24 * ROWS_PER_HOUR,
    "7d":  7  * 24 * ROWS_PER_HOUR,
    "14d": 14 * 24 * ROWS_PER_HOUR,
}

# IsolationForest config
CONTAMINATION = 0.05   # assume ~5% of readings are anomalous
RANDOM_STATE  = 42

def divider(title):
    print(f"\n{'='*65}\n  {title}\n{'='*65}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — LOAD PLANT 1 (wide format)
# ══════════════════════════════════════════════════════════════════════════════
divider("Stage 1: Load Plant 1 Wide-Format Data")

dfs = []
for filepath in PLANT1_FILES:
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"  Loaded: {filepath.split(chr(92))[-1]}")
        print(f"    Rows: {len(df):,}  |  Columns: {len(df.columns)}")
        dfs.append(df)
    except FileNotFoundError:
        print(f"  ⚠️  File not found: {filepath}")

if not dfs:
    print("  ❌ No files loaded. Update PLANT1_FILES paths.")
    exit()

df_wide = pd.concat(dfs, ignore_index=True)

# Parse timestamp
df_wide["ts"] = pd.to_datetime(
    df_wide.get("timestampDate", df_wide.get("timestamp")),
    utc=True, errors="coerce"
).dt.tz_localize(None)
df_wide = df_wide.dropna(subset=["ts"])
df_wide = df_wide.sort_values("ts").reset_index(drop=True)

print(f"\n  Combined shape: {df_wide.shape}")
print(f"  Date range: {df_wide['ts'].min()} → {df_wide['ts'].max()}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — MELT WIDE → LONG (one row per inverter per timestamp)
# ══════════════════════════════════════════════════════════════════════════════
divider("Stage 2: Reshape Wide → Long Format")

"""
WHY THIS STEP:
  Plant 1 stores all 12 inverters in one row:
    timestamp | inv[0].power | inv[1].power | ... | inv[11].power

  We need one row per inverter per timestamp:
    timestamp | inverter_id | power | temp | ...

  This is called "melting" or "unpivoting".
"""

long_records = []

for inv_idx in range(N_INVERTERS):
    prefix = f"inverters[{inv_idx}]"
    inv_id = f"P1-INV-{inv_idx:02d}"

    # Extract all columns for this inverter
    inv_cols = {
        f"{prefix}.power":      "ac_power",
        f"{prefix}.pv1_power":  "pv1_power",
        f"{prefix}.temp":       "inverter_temp",
        f"{prefix}.freq":       "grid_freq",
        f"{prefix}.kwh_today":  "kwh_today",
        f"{prefix}.kwh_total":  "kwh_total",
        f"{prefix}.op_state":   "op_state",
        f"{prefix}.v_ab":       "v_ab",
        f"{prefix}.v_bc":       "v_bc",
        f"{prefix}.v_ca":       "v_ca",
    }

    # String currents (up to 24)
    string_cols = {}
    for s in range(1, 25):
        col = f"smu[{inv_idx}].string{s}"
        if col in df_wide.columns:
            string_cols[col] = f"string{s}"

    # Build per-inverter dataframe
    available = {k: v for k, v in inv_cols.items() if k in df_wide.columns}
    available.update({k: v for k, v in string_cols.items() if k in df_wide.columns})

    inv_df = df_wide[["ts"] + list(available.keys())].copy()
    inv_df = inv_df.rename(columns=available)
    inv_df["inverter_id"] = inv_id
    inv_df["plant_id"]    = "Plant1"
    inv_df["plant_encoded"] = 1

    long_records.append(inv_df)
    print(f"  Extracted {inv_id}: {len(inv_df):,} rows × {len(inv_df.columns)} cols")

df = pd.concat(long_records, ignore_index=True)
df = df.sort_values(["inverter_id", "ts"]).reset_index(drop=True)

print(f"\n  Long format shape: {df.shape}")
print(f"  Inverters: {df['inverter_id'].nunique()} "
      f"({df['inverter_id'].unique().tolist()})")

# Also add ambient temp (one per timestamp, shared across all inverters)
if "sensors[0].ambient_temp" in df_wide.columns:
    ambient = df_wide[["ts", "sensors[0].ambient_temp"]].rename(
        columns={"sensors[0].ambient_temp": "ambient_temp"})
    df = df.merge(ambient, on="ts", how="left")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — CLEAN
# ══════════════════════════════════════════════════════════════════════════════
divider("Stage 3: Clean Plant 1 Data")

# Force numeric
numeric_cols = ["ac_power", "pv1_power", "inverter_temp", "grid_freq",
                "kwh_today", "kwh_total", "op_state",
                "v_ab", "v_bc", "v_ca"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Clamp physically impossible values
# Plant 1 inverters appear to be ~60kW class (12 inverters × 60kW = 720kW plant)
df["ac_power"]  = df["ac_power"].clip(lower=0, upper=100)
df["pv1_power"] = df["pv1_power"].clip(lower=0, upper=110)

# Clamp string currents (max ~15A per string physically)
string_cols = [c for c in df.columns if c.startswith("string")]
for col in string_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0, upper=20)

# op_state: treat -1 as 0 (communication loss → unknown, not fault)
# Create a separate flag for -1 (lost comms is its own signal)
df["comms_lost"] = (df["op_state"] == -1).astype(int)
df["op_state"]   = df["op_state"].replace(-1, 0)

# Grid voltages: v_ab/bc/ca are line-to-line 3-phase (~790V range)
for vcol in ["v_ab", "v_bc", "v_ca"]:
    if vcol in df.columns:
        df[vcol] = df[vcol].where(df[vcol].between(600, 900), np.nan)

# Daylight flag
df["is_daylight"] = (df["ac_power"] > 0.5).astype(int)

print(f"  Corrupted ac_power values clamped ✅")
print(f"  String current outliers clamped ✅")
print(f"  op_state=-1 → comms_lost flag ✅")
print(f"  Grid voltage range validated ✅")
print(f"\n  op_state distribution after fix:")
print(f"  {df['op_state'].value_counts().to_dict()}")
print(f"  comms_lost events: {df['comms_lost'].sum():,} "
      f"({df['comms_lost'].mean()*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
divider("Stage 4: Feature Engineering")

# Time features
df["hour"]       = df["ts"].dt.hour
df["month"]      = df["ts"].dt.month
df["day_of_week"]= df["ts"].dt.dayofweek
df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)

# 4a. Conversion efficiency (DC → AC)
df["conversion_efficiency"] = np.where(
    df["pv1_power"] > 0.5,
    (df["ac_power"] / df["pv1_power"]).clip(0, 1.05) * 100,
    np.nan
)

# 4b. Grid voltage imbalance (line-to-line voltages)
v_cols = [c for c in ["v_ab", "v_bc", "v_ca"] if c in df.columns]
if len(v_cols) == 3:
    # Memory fix: use float32 numpy for row-wise mean/std
    _v_vals = df[v_cols].to_numpy(dtype="float32")
    df["grid_v_mean"] = _v_vals.mean(axis=1)
    df["grid_v_std"]  = _v_vals.std(axis=1)
    del _v_vals
    df["grid_v_imbalance"] = np.where(
        df["grid_v_mean"] > 0,
        df["grid_v_std"] / df["grid_v_mean"] * 100,
        np.nan
    )

# 4c. String current imbalance
active_strings = [c for c in string_cols if df[c].max() > 0.5]
if len(active_strings) >= 2:
    # Memory fix: cast to float32 before row-wise ops to halve allocation
    # (~4.5M rows × 24 strings × 4 bytes = ~430 MB instead of ~1.79 GB)
    _str_vals = df[active_strings].to_numpy(dtype="float32")
    df["string_mean"] = _str_vals.mean(axis=1)
    df["string_std"]  = _str_vals.std(axis=1)
    del _str_vals  # free immediately
    df["string_imbalance"] = np.where(
        df["string_mean"] > 0.5,
        df["string_std"] / df["string_mean"],
        np.nan
    )
    # Count of strings producing near-zero current during daylight
    # (indicates dead strings — most powerful anomaly signal for Plant 1)
    # Memory + speed fix: vectorised numpy instead of row-wise apply()
    # apply() on 4.5M rows is ~100x slower and allocates a full copy
    _str_vals2 = df[active_strings].to_numpy(dtype="float32")
    df["dead_string_count"] = (_str_vals2 < 0.2).sum(axis=1) * df["is_daylight"].to_numpy()
    del _str_vals2  # free immediately

# 4d. KWh daily production rate (how fast is energy accumulating?)
df["kwh_today_rate"] = df.groupby(
    [df["inverter_id"], df["ts"].dt.date]
)["kwh_today"].transform(lambda x: x.diff().clip(lower=0))

# 4e. Temperature deviation from fleet mean at each timestamp
#     If one inverter is much hotter than siblings → thermal issue
df["fleet_temp_mean"] = df.groupby("ts")["inverter_temp"].transform("mean")
df["temp_vs_fleet"]   = df["inverter_temp"] - df["fleet_temp_mean"]

# 4f. Power deviation from fleet mean at each timestamp
#     Underproducing vs siblings during same weather → performance issue
df["fleet_power_mean"] = df.groupby("ts")["ac_power"].transform("mean")
df["power_vs_fleet"]   = df["ac_power"] - df["fleet_power_mean"]
# Normalised: how many % below fleet average?
df["power_vs_fleet_pct"] = np.where(
    df["fleet_power_mean"] > 0.5,
    df["power_vs_fleet"] / df["fleet_power_mean"] * 100,
    np.nan
)

# 4g. Rolling features per inverter
ROLL_COLS = {
    "ac_power":               ["mean", "std"],
    "conversion_efficiency":  ["mean", "std"],
    "inverter_temp":          ["mean", "std", "max"],
    "string_imbalance":       ["mean", "max"],
    "comms_lost":             ["sum"],
}

for col, stats in ROLL_COLS.items():
    if col not in df.columns:
        continue
    grp = df.groupby("inverter_id")[col]
    for wname, wsize in WINDOWS.items():
        for stat in stats:
            feat = f"{col}_{stat}_{wname}"
            df[feat] = grp.transform(
                lambda x, w=wsize, s=stat: getattr(
                    x.rolling(w, min_periods=w//4), s)()
            )

# 4h. Delta features (current vs own rolling baseline)
df["power_delta_7d"]      = df["ac_power"] - df.get("ac_power_mean_7d", pd.Series(dtype=float))
df["temp_delta_7d"]       = df["inverter_temp"] - df.get("inverter_temp_mean_7d", pd.Series(dtype=float))
df["efficiency_delta_7d"] = df["conversion_efficiency"] - df.get("conversion_efficiency_mean_7d", pd.Series(dtype=float))

# 4i. Lag change features
for col in ["ac_power", "inverter_temp"]:
    if col not in df.columns:
        continue
    for lag_name, lag_rows in [("1d", 24*ROWS_PER_HOUR), ("7d", 7*24*ROWS_PER_HOUR)]:
        df[f"{col}_change_{lag_name}"] = df[col] - df.groupby("inverter_id")[col].transform(
            lambda x: x.shift(lag_rows))

print(f"  Time features ✅")
print(f"  Conversion efficiency ✅")
print(f"  Grid voltage imbalance ✅")
print(f"  String imbalance + dead string count ✅")
print(f"  Fleet comparison features ✅")
print(f"  Rolling + delta + lag features ✅")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — ISOLATION FOREST ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
divider("Stage 5: IsolationForest Anomaly Detection")

"""
HOW ISOLATION FOREST WORKS (plain English):
  - It randomly picks a feature and splits the data at a random threshold
  - Anomalies are "isolated" (separated from the rest) in fewer splits
  - A normal reading needs many splits to isolate → high anomaly score
  - An unusual reading is isolated quickly → low anomaly score

  Output:
    anomaly_score: continuous score (higher = more anomalous)
    is_anomaly: 1 if this row is anomalous, 0 if normal
    anomaly_severity: 0/1/2/3 (None/Low/Medium/High)

WHY THIS WORKS FOR PLANT 1 WITHOUT LABELS:
  - The model learns what "normal" looks like from 12 inverters
  - When one inverter starts behaving differently from its own history
    AND from sibling inverters → flagged as anomalous
  - No fault labels needed!
"""

# Features used for anomaly detection
# We use daylight-only rows for cleaner signals
ANOMALY_FEATURES = [
    # Core performance
    "ac_power", "pv1_power", "conversion_efficiency",
    "inverter_temp",

    # Fleet comparison (key signals — is this inverter an outlier?)
    "temp_vs_fleet", "power_vs_fleet_pct",

    # String health
    "string_imbalance", "dead_string_count",

    # Grid quality
    "grid_v_imbalance",

    # Rolling trends
    "ac_power_mean_7d", "ac_power_std_7d",
    "conversion_efficiency_mean_7d", "conversion_efficiency_std_7d",
    "inverter_temp_mean_7d", "inverter_temp_std_7d",

    # Delta from baseline
    "power_delta_7d", "temp_delta_7d", "efficiency_delta_7d",

    # Communication loss rate (rolling sum)
    "comms_lost_sum_7d",

    # Time context
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]

# Keep only features that exist
ANOMALY_FEATURES = [f for f in ANOMALY_FEATURES if f in df.columns]
print(f"  Using {len(ANOMALY_FEATURES)} features for anomaly detection")

# Use daylight rows only (anomaly in dark = just nighttime, not interesting)
df_day = df[(df["is_daylight"] == 1) & df["ac_power_mean_7d"].notna()].copy()

# Fill NaN with median per inverter
X = df_day[ANOMALY_FEATURES].copy()
for col in X.columns:
    X[col] = X[col].fillna(X[col].median())

# Scale features (IsolationForest works better with scaled data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train IsolationForest
print(f"\n  Training IsolationForest on {len(X_scaled):,} daylight rows...")
iso = IsolationForest(
    n_estimators=200,        # more trees = more stable scores
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1                # use all CPU cores
)
iso.fit(X_scaled)

# Get scores
# score_samples returns negative values; more negative = more anomalous
# We flip and normalise to 0-100 (higher = more anomalous)
raw_scores = iso.score_samples(X_scaled)
df_day["anomaly_raw_score"] = raw_scores
df_day["is_anomaly"]        = (iso.predict(X_scaled) == -1).astype(int)

# Normalise to 0-100 for interpretability
min_score = raw_scores.min()
max_score = raw_scores.max()
df_day["anomaly_score"] = (
    (raw_scores - min_score) / (max_score - min_score + 1e-9) * 100
)
# Invert: high score = bad
df_day["anomaly_score"] = 100 - df_day["anomaly_score"]

# Severity tiers (matches the supervised model's 3-class output)
df_day["anomaly_severity"] = pd.cut(
    df_day["anomaly_score"],
    bins=[0, 40, 65, 85, 100],
    labels=[0, 1, 2, 3],    # 0=Normal, 1=Low, 2=Medium, 3=High
    include_lowest=True
).astype(int)

print(f"  IsolationForest training complete ✅")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — RESULTS & EXPORT
# ══════════════════════════════════════════════════════════════════════════════
divider("Stage 6: Anomaly Results")

print(f"\n  Anomaly detection summary:")
print(f"  {'─'*55}")
SEVERITY_LABELS = {0: "Normal  ", 1: "Low Risk", 2: "Med Risk", 3: "High Risk"}
for sev, cnt in df_day["anomaly_severity"].value_counts().sort_index().items():
    pct = cnt / len(df_day) * 100
    bar = "█" * int(pct / 2)
    print(f"  Severity {sev} ({SEVERITY_LABELS[sev]}): {cnt:>7,} rows ({pct:5.1f}%)  {bar}")

print(f"\n  Per-inverter anomaly rate:")
print(f"  {'─'*55}")
for inv_id, grp in df_day.groupby("inverter_id"):
    high = (grp["anomaly_severity"] >= 2).sum()
    total = len(grp)
    avg_score = grp["anomaly_score"].mean()
    bar = "█" * int(high/total*50)
    print(f"  {inv_id}: {high:>5,} high-risk rows "
          f"({high/total*100:.1f}%)  avg_score={avg_score:.1f}  {bar}")

# Merge scores back to full dataframe (night rows get NaN anomaly score)
df = df.merge(
    df_day[["inverter_id", "ts", "anomaly_score", "is_anomaly",
            "anomaly_severity"]],
    on=["inverter_id", "ts"],
    how="left"
)
df["anomaly_score"]    = df["anomaly_score"].fillna(0)
df["is_anomaly"]       = df["is_anomaly"].fillna(0).astype(int)
df["anomaly_severity"] = df["anomaly_severity"].fillna(0).astype(int)

# Drop raw wide-format columns we no longer need
drop_cols = ["_id", "mac", "createdAt", "fromServer", "dataLoggerModelId",
             "__v", "timestampDate", "timestamp"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Save
df.to_csv("plant1_anomaly_scores.csv", index=False)

print(f"\n  ✅ Saved → plant1_anomaly_scores.csv")
print(f"  Shape: {df.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — TOP ANOMALOUS EVENTS (for GenAI / Dashboard team)
# ══════════════════════════════════════════════════════════════════════════════
divider("Stage 7: Top Anomalous Events (for GenAI Layer)")

top_anomalies = df_day[df_day["anomaly_severity"] >= 2].nlargest(20, "anomaly_score")[
    ["inverter_id", "ts", "anomaly_score", "anomaly_severity",
     "ac_power", "conversion_efficiency", "inverter_temp",
     "temp_vs_fleet", "power_vs_fleet_pct", "dead_string_count"]
].reset_index(drop=True)

top_anomalies.to_csv("plant1_top_anomalies.csv", index=False)
print(f"\n  Top 20 anomalous events saved → plant1_top_anomalies.csv")
print(f"\n  Preview:")
print(top_anomalies[["inverter_id", "ts", "anomaly_score",
                      "anomaly_severity"]].head(10).to_string(index=False))

print(f"""
{'='*65}
  PLANT 1 PROCESSING COMPLETE
{'='*65}

  Outputs:
    plant1_anomaly_scores.csv   ← full dataset with anomaly scores
    plant1_top_anomalies.csv    ← top events for GenAI narrative

  Key columns for downstream teams:
    anomaly_score    → 0-100 continuous risk score (higher=worse)
    is_anomaly       → 1 if this row is flagged as anomalous
    anomaly_severity → 0=Normal, 1=Low, 2=Medium, 3=High

  How this maps to the supervised model output (Plants 2 & 3):
    anomaly_severity 0 ≈ risk_class 0 (No Risk)
    anomaly_severity 1 ≈ risk_class 1 (Degradation Risk)
    anomaly_severity 2 ≈ risk_class 2 (Shutdown Risk)
    anomaly_severity 3 → URGENT (no supervised equivalent)

  For the GenAI team:
    Pass top anomaly rows + feature values to LLM prompt
    The anomaly_score + contributing features make excellent
    narrative generation context
{'='*65}
""")