"""
╔══════════════════════════════════════════════════════════════════╗
║   DATA EXPLORER — Solar Inverter Dataset                        ║
║   HACKaMINeD 2026 · Run this FIRST before anything else         ║
║                                                                  ║
║   HOW TO RUN:                                                    ║
║     1. Put this file in the same folder as your CSV(s)          ║
║     2. Update FILE_PATHS below with your actual filenames       ║
║     3. Run: python data_explorer.py                             ║
║     4. Copy the printed output and share it                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import os
import glob

# ─────────────────────────────────────────────────────────────────
# ✏️  EDIT THIS SECTION — point to your actual CSV file(s)
# ─────────────────────────────────────────────────────────────────

# Option A: Single file
FILE_PATHS = [
    r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 1-20260305T111817Z-3-001\Plant 1\Copy of ICR2-LT1-Celestical-10000.73.raws.csv", 
    r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 1-20260305T111817Z-3-001\Plant 1\Copy of ICR2-LT2-Celestical-10000.73.raws.csv", 
    r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 2-20260305T111818Z-3-001\Plant 2\Copy of 80-1F-12-0F-AC-12.raws.csv", 
    r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 2-20260305T111818Z-3-001\Plant 2\Copy of 80-1F-12-0F-AC-BB.raws.csv",          # ← replace with your filename
    r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 3-20260305T111819Z-3-001\Plant 3\Copy of 54-10-EC-8C-14-6E.raws.csv",
    r"C:\Users\HP\Desktop\hackathon\Hackamined-2026\dataset_aubergine\Plant 3-20260305T111819Z-3-001\Plant 3\Copy of 54-10-EC-8C-14-69.raws.csv",
    # r"another_file.csv",     # ← add more if you have multiple
]

# Option B: Auto-detect all CSVs in the current folder
#           Uncomment the line below and comment out FILE_PATHS above
# FILE_PATHS = glob.glob("*.csv")

# How many rows to sample for the preview (keeps RAM usage low)
SAMPLE_ROWS = 5

# If your files are very large (>500MB), set this to True
# It will only read the first 50,000 rows for exploration
LARGE_FILE_MODE = False
LARGE_FILE_NROWS = 50_000

# ─────────────────────────────────────────────────────────────────

def divider(title=""):
    line = "=" * 65
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(line)
    else:
        print(line)


def explore_file(filepath):
    divider(f"FILE: {os.path.basename(filepath)}")

    # ── File size ─────────────────────────────────────────────────
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    print(f"\n  File size     : {size_mb:.1f} MB  ({size_bytes:,} bytes)")

    # ── Load (smart: sample if large) ────────────────────────────
    if LARGE_FILE_MODE and size_mb > 50:
        print(f"  Load mode     : LARGE FILE — reading first {LARGE_FILE_NROWS:,} rows only")
        df = pd.read_csv(filepath, nrows=LARGE_FILE_NROWS, low_memory=False)
        is_sample = True
    else:
        print(f"  Load mode     : Full file")
        df = pd.read_csv(filepath, low_memory=False)
        is_sample = False

    total_rows = len(df)
    print(f"  Rows loaded   : {total_rows:,}" + (" (sample)" if is_sample else ""))
    print(f"  Columns       : {len(df.columns)}")

    # ── Column names & types ──────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  COLUMN NAMES & DATA TYPES")
    print(f"{'─'*65}")
    print(f"  {'#':<4} {'Column Name':<35} {'Dtype':<12} {'Non-Null %'}")
    print(f"  {'─'*4} {'─'*35} {'─'*12} {'─'*10}")
    for i, col in enumerate(df.columns):
        non_null_pct = (df[col].notna().sum() / len(df)) * 100
        print(f"  {i:<4} {col:<35} {str(df[col].dtype):<12} {non_null_pct:.1f}%")

    # ── First few rows ────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  FIRST {SAMPLE_ROWS} ROWS (transposed for readability)")
    print(f"{'─'*65}")
    print(df.head(SAMPLE_ROWS).T.to_string())

    # ── Numeric column stats ──────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print(f"\n{'─'*65}")
        print(f"  NUMERIC COLUMN STATISTICS ({len(numeric_cols)} columns)")
        print(f"{'─'*65}")
        stats = df[numeric_cols].describe().T[["count", "mean", "min", "max"]]
        stats["missing"] = df[numeric_cols].isnull().sum()
        print(stats.round(3).to_string())

    # ── Non-numeric / categorical columns ────────────────────────
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        print(f"\n{'─'*65}")
        print(f"  TEXT / CATEGORICAL COLUMNS ({len(cat_cols)} columns)")
        print(f"{'─'*65}")
        for col in cat_cols:
            unique_vals = df[col].nunique()
            sample_vals = df[col].dropna().unique()[:8].tolist()
            print(f"\n  [{col}]")
            print(f"    Unique values : {unique_vals}")
            print(f"    Sample values : {sample_vals}")
            if unique_vals <= 20:
                vc = df[col].value_counts().head(10)
                print(f"    Value counts  :")
                for val, cnt in vc.items():
                    print(f"      {str(val):<30} {cnt:>6,} rows")

    # ── Missing value summary ─────────────────────────────────────
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print(f"\n{'─'*65}")
        print(f"  MISSING VALUES SUMMARY")
        print(f"{'─'*65}")
        for col, cnt in missing.items():
            pct = cnt / len(df) * 100
            bar = "█" * int(pct / 5)
            print(f"  {col:<35} {cnt:>6,} missing  ({pct:5.1f}%)  {bar}")
    else:
        print(f"\n  ✅ No missing values found!")

    # ── Timestamp / date column detection ────────────────────────
    print(f"\n{'─'*65}")
    print(f"  TIMESTAMP COLUMN DETECTION")
    print(f"{'─'*65}")
    possible_ts = [c for c in df.columns
                   if any(kw in c.lower() for kw in
                          ["time", "date", "ts", "datetime", "created", "updated", "stamp"])]
    if possible_ts:
        for col in possible_ts:
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else "N/A"
            print(f"  Likely timestamp column : [{col}]  →  sample value: {sample}")
    else:
        print(f"  ⚠️  No obvious timestamp column detected.")
        print(f"  Check column names manually from the list above.")

    # ── Inverter ID column detection ──────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  INVERTER ID COLUMN DETECTION")
    print(f"{'─'*65}")
    possible_id = [c for c in df.columns
                   if any(kw in c.lower() for kw in
                          ["inverter", "inv", "device", "unit", "id", "serial", "source"])]
    if possible_id:
        for col in possible_id:
            unique_count = df[col].nunique()
            sample_vals = df[col].dropna().unique()[:5].tolist()
            print(f"  Likely inverter ID col  : [{col}]")
            print(f"    Unique values : {unique_count}")
            print(f"    Sample values : {sample_vals}")
    else:
        print(f"  ⚠️  No obvious inverter ID column detected.")

    # ── Alarm / event column detection ───────────────────────────
    print(f"\n{'─'*65}")
    print(f"  ALARM / EVENT COLUMN DETECTION")
    print(f"{'─'*65}")
    possible_alarm = [c for c in df.columns
                      if any(kw in c.lower() for kw in
                             ["alarm", "event", "fault", "error", "alert", "status", "flag", "code"])]
    if possible_alarm:
        for col in possible_alarm:
            unique_count = df[col].nunique()
            sample_vals = df[col].dropna().unique()[:8].tolist()
            print(f"  Possible alarm col : [{col}]  —  {unique_count} unique values")
            print(f"    Sample : {sample_vals}")
    else:
        print(f"  ⚠️  No obvious alarm/event columns detected.")

    # ── Potential label / target column ──────────────────────────
    print(f"\n{'─'*65}")
    print(f"  POTENTIAL TARGET / LABEL COLUMN DETECTION")
    print(f"{'─'*65}")
    possible_label = [c for c in df.columns
                      if any(kw in c.lower() for kw in
                             ["label", "target", "failure", "fault", "shutdown",
                              "status", "class", "anomaly", "flag"])]
    if possible_label:
        for col in possible_label:
            vc = df[col].value_counts().head(5)
            print(f"  Possible label col : [{col}]")
            print(f"    Value distribution: {dict(vc)}")
    else:
        print(f"  ℹ️  No pre-existing label column found.")
        print(f"  A failure label will be engineered from alarm/event data.")

    print(f"\n{'─'*65}")
    print(f"  ✅ Exploration complete for: {os.path.basename(filepath)}")
    print(f"{'─'*65}\n")

    return df  # return for optional further use


# ─────────────────────────────────────────────────────────────────
# MAIN — run explorer on all files
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "=" * 65)
    print("  HACKAMINED 2026 — Dataset Explorer")
    print("  Solar Inverter Failure Prediction Pipeline")
    print("=" * 65)

    valid_paths = []
    for path in FILE_PATHS:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            print(f"\n  ⚠️  File not found: {path}")
            print(f"      Make sure the path is correct and the file exists.")

    if not valid_paths:
        print("\n  ❌ No valid files found. Please update FILE_PATHS at the top of this script.")
    else:
        print(f"\n  Found {len(valid_paths)} file(s) to explore...\n")
        for path in valid_paths:
            explore_file(path)

    print("\n" + "=" * 65)
    print("  NEXT STEP:")
    print("  Copy everything printed above and share it.")
    print("  The full feature engineering pipeline will then be")
    print("  adapted to your exact column names automatically.")
    print("=" * 65 + "\n")