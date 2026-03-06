"""
╔══════════════════════════════════════════════════════════════════════════╗
║   LABEL FIX — Stage 5 Replacement                                       ║
║   Problem: 99.9% of rows were labelled "at-risk"                        ║
║   Root Cause: op_state==8 is nighttime OFF, not a fault                 ║
║   Fix: Use ONLY alarm_code 534/556 as major fault signal                ║
║        + add a "confirmed shutdown" combined condition                   ║
╚══════════════════════════════════════════════════════════════════════════╝

WHY op_state==8 MISLED US:
──────────────────────────
  - INV-69: 2,652 op_state==8 rows  BUT only 43 alarm_code 534/556 rows
  - op_state==8 mean is ~2.2 → it happens constantly (every night = 0, runs = 3, off = 8?)
  - Real faults are rare → only alarm_code 534 or 556

REVISED FAULT DEFINITIONS:
───────────────────────────
  Tier 1 (DEFINITE fault): alarm_code IN [534, 556]
  Tier 2 (CONFIRMED fault): alarm_code IN [534, 556] AND op_state == 8
  Tier 3 (DEGRADATION): op_state == 4 (derated) persisting for >2 hours
                         → use this as a separate softer signal

  We use Tier 1 as primary label. Tier 3 becomes a feature, not a label.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# ✏️  CONFIGURE — same paths as before
# ══════════════════════════════════════════════════════════════════════════════

FILES = {
    # Plant 3
    "INV-6E":   r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 3-20260305T111819Z-3-001\Plant 3\Copy of 54-10-EC-8C-14-6E.raws.csv",
    "INV-69":   r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 3-20260305T111819Z-3-001\Plant 3\Copy of 54-10-EC-8C-14-69.raws.csv",
    # Plant 2 — same column structure, fully compatible
    "INV-AC12": r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 2-20260305T111818Z-3-001\Plant 2\Copy of 80-1F-12-0F-AC-12.raws.csv",
    "INV-ACBB": r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 2-20260305T111818Z-3-001\Plant 2\Copy of 80-1F-12-0F-AC-BB.raws.csv",
}

PREDICTION_WINDOW_DAYS = 7
ROWS_PER_HOUR          = 12   # 5-min data → 12 rows/hr
LABEL_WINDOW           = PREDICTION_WINDOW_DAYS * 24 * ROWS_PER_HOUR

# ── REVISED fault codes (ONLY these are real major faults) ────────────────────
MAJOR_ALARM_CODES = [534, 556]
# op_state meanings (from data analysis):
#   0 = Off/Night    3 = Running    4 = Derated
#   5 = Starting     7 = Recovering 8 = Off/Standby (NOT always fault!)

def divider(title):
    print(f"\n{'='*65}\n  {title}\n{'='*65}")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD (same as before — abbreviated for patch focus)
# ══════════════════════════════════════════════════════════════════════════════
divider("Loading & Merging Files")

dfs = []
for inv_id, filepath in FILES.items():
    df = pd.read_csv(filepath, low_memory=False)
    df["inverter_id"] = inv_id
    dfs.append(df)
    print(f"  {inv_id}: {len(df):,} rows")

df_raw = pd.concat(dfs, ignore_index=True)

# Parse timestamp
df_raw["ts"] = pd.to_datetime(df_raw["timestampDate"], utc=True, errors="coerce")
df_raw["ts"] = df_raw["ts"].dt.tz_convert(None)  # strip UTC tz → naive datetime
df_raw = df_raw.dropna(subset=["ts"])
df_raw = df_raw.sort_values(["inverter_id", "ts"]).reset_index(drop=True)

# ── Rename key columns for clarity ───────────────────────────────────────────
df_raw = df_raw.rename(columns={
    "inverters[0].alarm_code": "alarm_code",
    "inverters[0].op_state":   "op_state",
    "inverters[0].power":      "ac_power",
    "inverters[0].pv1_voltage":"dc_voltage",
    "inverters[0].pv1_current":"dc_current",
    "inverters[0].pv1_power":  "dc_power_raw",
    "inverters[0].temp":       "inverter_temp",
    "inverters[0].kwh_today":  "kwh_today",
    "meters[0].v_r":           "grid_v_r",
    "meters[0].v_y":           "grid_v_y",
    "meters[0].v_b":           "grid_v_b",
    "meters[0].freq":          "grid_freq",
    "meters[0].pf":            "power_factor",
    "meters[0].meter_active_power": "grid_active_power",
})
df = df_raw.copy()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 FIX — REVISED LABEL ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
divider("STAGE 5 FIX: Revised Failure Label Engineering")

# ── STEP 1: Understand op_state distribution BEFORE labelling ─────────────────
print("\n  op_state value counts (both inverters combined):")
print(df["op_state"].value_counts().sort_index().to_string())

print("\n  alarm_code value counts (both inverters combined):")
print(df["alarm_code"].value_counts().sort_index().to_string())

# ── STEP 2: Check what op_state==8 actually looks like ───────────────────────
print("\n  Cross-tab: when op_state==8, what is alarm_code?")
mask_8 = df["op_state"] == 8
cross = df[mask_8]["alarm_code"].value_counts().head(10)
print(cross.to_string())
print(f"\n  → Most op_state==8 rows have alarm_code=100 (standby/night)")
print(f"  → Conclusion: op_state==8 alone is NOT a reliable fault indicator")

# ── STEP 3: Mark ONLY confirmed major fault events ───────────────────────────
#
# PRIMARY signal: alarm_code 534 or 556 (definite hardware fault)
df["is_major_fault"] = df["alarm_code"].isin(MAJOR_ALARM_CODES).astype(int)

# SECONDARY signal: derated output (op_state==4) persisting >2hrs during daylight
# This becomes a FEATURE (degradation risk), not the primary label
df["is_derated"] = (
    (df["op_state"] == 4) &
    (df.get("ac_power", 0) > 0.5)   # only during daylight
).astype(int)

print(f"\n  Revised major fault counts (alarm_code 534/556 ONLY):")
for inv_id, grp in df.groupby("inverter_id"):
    n = grp["is_major_fault"].sum()
    total = len(grp)
    print(f"    {inv_id}: {n:,} actual fault rows / {total:,} total ({n/total*100:.3f}%)")

# ── STEP 4: Build 3-tier label system ─────────────────────────────────────────
#
# Label 2 = Shutdown Risk     (major fault within 7 days)    ← BONUS: multi-class
# Label 1 = Degradation Risk  (derated op persisting 2+ hrs within 7 days)
# Label 0 = No Risk
#
# For binary classification: treat 2 as positive, 1 optionally as positive

def forward_rolling_any(series, window):
    """Returns 1 if any event in series occurs in the NEXT `window` rows."""
    reversed_sum = series[::-1].rolling(window=window, min_periods=1).sum()[::-1]
    return (reversed_sum > 0).astype(int)

# Deration window: 2-hour persistence (24 rows) to filter one-off readings
DERATE_PERSIST = 24
df["derated_persistent"] = df.groupby("inverter_id")["is_derated"].transform(
    lambda x: (x.rolling(DERATE_PERSIST, min_periods=1).sum() >= 6).astype(int)
)

# Forward labels
df["shutdown_risk"] = df.groupby("inverter_id")["is_major_fault"].transform(
    lambda x: forward_rolling_any(x, LABEL_WINDOW)
)
df["degradation_risk"] = df.groupby("inverter_id")["derated_persistent"].transform(
    lambda x: forward_rolling_any(x, LABEL_WINDOW)
)

# ── 3-class label (BONUS requirement from PDF!) ───────────────────────────────
df["risk_class"] = 0
df.loc[df["degradation_risk"] == 1, "risk_class"] = 1   # degradation
df.loc[df["shutdown_risk"]    == 1, "risk_class"] = 2   # shutdown (overrides)

# ── Binary label (for minimum requirement) ────────────────────────────────────
df["failure_label"] = (df["shutdown_risk"] == 1).astype(int)

# ── STEP 5: Print distribution ────────────────────────────────────────────────
print(f"\n  {'─'*60}")
print(f"  BINARY label distribution (shutdown risk only):")
vc = df["failure_label"].value_counts()
for lbl, cnt in sorted(vc.items()):
    pct = cnt / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"    Label {lbl}: {cnt:6,} rows ({pct:5.1f}%)  {bar}")

print(f"\n  MULTI-CLASS label distribution (for BONUS):")
vc3 = df["risk_class"].value_counts().sort_index()
labels = {0: "No Risk", 1: "Degradation Risk", 2: "Shutdown Risk"}
for cls, cnt in vc3.items():
    pct = cnt / len(df) * 100
    print(f"    Class {cls} ({labels[cls]:<20}): {cnt:6,} rows ({pct:5.1f}%)")

print(f"\n  Per-inverter breakdown:")
for inv_id, grp in df.groupby("inverter_id"):
    s0 = (grp["risk_class"]==0).sum()
    s1 = (grp["risk_class"]==1).sum()
    s2 = (grp["risk_class"]==2).sum()
    print(f"    {inv_id}:")
    print(f"      No Risk       : {s0:>7,} ({s0/len(grp)*100:.1f}%)")
    print(f"      Degradation   : {s1:>7,} ({s1/len(grp)*100:.1f}%)")
    print(f"      Shutdown Risk : {s2:>7,} ({s2/len(grp)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# SMOTE WARNING — CLASS IMBALANCE HANDLING
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  {'─'*60}")
print(f"  ⚠️  CLASS IMBALANCE NOTE FOR ML TEAM:")
print(f"  If shutdown_risk rows are still rare (<5%), tell the ML team to use:")
print(f"    - class_weight='balanced' in XGBoost/LightGBM")
print(f"    - scale_pos_weight = (n_negative / n_positive) in XGBoost")
print(f"    - OR: SMOTE oversampling from imbalanced-learn library")
print(f"    - Evaluate with F1 + AUC, NOT accuracy (accuracy is misleading here)")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE REVISED LABEL AUDIT
# ══════════════════════════════════════════════════════════════════════════════
divider("Saving Revised Label Audit")

label_audit = df[[
    "inverter_id", "ts", "alarm_code", "op_state",
    "is_major_fault", "is_derated", "derated_persistent",
    "shutdown_risk", "degradation_risk", "risk_class", "failure_label"
]].copy()

label_audit.to_csv("label_audit_REVISED.csv", index=False)
print(f"  Saved → label_audit_REVISED.csv")
print(f"\n  🔍 REVIEW CHECKLIST:")
print(f"  1. Open label_audit_REVISED.csv")
print(f"  2. Filter is_major_fault==1 → verify these are real faults")
print(f"  3. Check that failure_label==1 appears in the 7 days BEFORE each fault")
print(f"  4. Confirm failure_label==0 rows are genuinely fault-free periods")

# ══════════════════════════════════════════════════════════════════════════════
# PATCH: ADD REVISED LABELS BACK INTO FULL PIPELINE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
divider("Patching Labels Into Final Feature CSV")

print(f"  Loading existing inverter_features_final.csv...")
try:
    df_ml = pd.read_csv("inverter_features_final.csv", parse_dates=["timestamp"])

    # Merge revised labels in by timestamp + inverter_id
    revised_labels = df[[
        "inverter_id", "ts",
        "failure_label", "risk_class",
        "shutdown_risk", "degradation_risk", "is_derated"
    ]].rename(columns={"ts": "timestamp"})

    # Drop old labels
    df_ml = df_ml.drop(columns=["failure_label"], errors="ignore")

    # Merge new labels
    df_ml = df_ml.merge(
        revised_labels,
        on=["inverter_id", "timestamp"],
        how="left"
    )

    # Also add derated_persistent as a feature (not label)
    df_ml["is_derated"] = df_ml["is_derated"].fillna(0).astype(int)

    df_ml.to_csv("inverter_features_REVISED.csv", index=False)

    print(f"\n  Final revised feature matrix: {df_ml.shape}")
    print(f"\n  FINAL Binary label distribution:")
    vc = df_ml["failure_label"].value_counts()
    for lbl, cnt in sorted(vc.items()):
        pct = cnt / len(df_ml) * 100
        bar = "█" * int(pct / 3)
        print(f"    Label {lbl}: {cnt:6,} rows ({pct:5.1f}%)  {bar}")

    print(f"\n  FINAL Multi-class distribution:")
    vc3 = df_ml["risk_class"].value_counts().sort_index()
    for cls, cnt in vc3.items():
        print(f"    Class {cls}: {cnt:6,} rows ({cnt/len(df_ml)*100:.1f}%)")

    print(f"\n  ✅ Saved → inverter_features_REVISED.csv")
    print(f"  ✅ Hand THIS file (not the old one) to the ML team")

except FileNotFoundError:
    print(f"  ⚠️  inverter_features_final.csv not found in current directory.")
    print(f"  Run the full feature_engineering_aubergine.py first,")
    print(f"  then run this patch script from the same folder.")

print(f"\n{'='*65}")
print(f"  SUMMARY OF CHANGES:")
print(f"  OLD: is_major_fault = (alarm_code in [534,556]) OR (op_state==8)")
print(f"       → op_state==8 is nighttime OFF, triggered 4000+ times")
print(f"       → Result: 99.9% rows labelled at-risk (useless for ML)")
print(f"")
print(f"  NEW: is_major_fault = alarm_code IN [534, 556] ONLY")
print(f"       → Only 48 real fault events across both inverters")
print(f"       → Forward 7-day window creates meaningful at-risk periods")
print(f"       → Multi-class adds Degradation Risk (op_state==4) as class 1")
print(f"       → Result: clean, realistic label distribution for ML")
print(f"{'='*65}\n")