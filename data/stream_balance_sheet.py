"""
Stream-process the 481MB balance_sheet.csv without loading entire file into RAM.

Strategy:
  1. Read CSV in 50K-row chunks
  2. Filter each chunk to CSI300 constituent stocks
  3. Convert to long format (melt)
  4. Compute time_since_update per variable
  5. Z-score per variable (using streaming stats)
  6. Append to output parquet file

This keeps peak memory under ~500MB even with 8GB RAM.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

BS_PATH = Path("D:/financial_data/financial/balance_sheet.csv")
OUT_DIR = Path("D:/financial_data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_SIZE = 30000

# CSMAR balance sheet code → readable name mapping
CSMAR_BS_MAP = {
    "A001000000": "total_assets",
    "A001101000": "current_assets",
    "A002000000": "total_liabilities",
    "A002101000": "current_liabilities",
    "A003000000": "total_equity",
    "A001212000": "fixed_assets",
    "A001223000": "cash_equivalents",
    "A001123000": "inventory",
    "A001111000": "accounts_receivable",
    "A001109000": "notes_receivable",
    "A002206000": "long_term_debt",
    "A002201000": "short_term_borrowing",
    "A003101000": "paid_in_capital",
    "A003105000": "capital_reserve",
    "A003106000": "retained_earnings",
    "A001218000": "intangible_assets",
}

print("=" * 60)
print("  STREAMING BALANCE SHEET PROCESSOR")
print("=" * 60)

# Read header to get columns
header = pd.read_csv(BS_PATH, nrows=0)
all_cols = list(header.columns)

# Select columns: Stkcd, Accper, DeclareDate + CSMAR codes
keep_cols = ["Stkcd", "Accper", "DeclareDate"]
metric_cols = [c for c in all_cols if c in CSMAR_BS_MAP]
cols_to_read = keep_cols + metric_cols
print(f"  Columns: {len(cols_to_read)} (from {len(all_cols)} total)")
print(f"  Processing ALL companies (no CSI300 filter — Stkcd ≠ stock code)")

# ── Pass 1: Compute per-variable mean and std for z-score ─────────
print("\n  Pass 1: Computing streaming stats...")
var_stats = {}  # {var_name: {"n": N, "sum": S, "sum_sq": SQ}}

chunk_num = 0
for chunk in pd.read_csv(BS_PATH, usecols=cols_to_read, chunksize=CHUNK_SIZE,
                         low_memory=False):
    chunk_num += 1
    # Process ALL companies (Stkcd is CSMAR internal code, not stock exchange code)
    chunk["Stkcd"] = chunk["Stkcd"].astype(str)

    # Use DeclareDate if available, else Accper
    chunk["date"] = pd.to_datetime(chunk["DeclareDate"], errors="coerce")
    mask = chunk["date"].isna()
    chunk.loc[mask, "date"] = pd.to_datetime(chunk.loc[mask, "Accper"], errors="coerce")
    chunk = chunk.dropna(subset=["date"])

    # Melt to long format
    long = chunk.melt(id_vars=["Stkcd", "date"], value_vars=metric_cols,
                      var_name="var_code", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])

    # Make qualified variable names
    long["variable"] = "bs_" + long["Stkcd"] + "::" + long["var_code"].map(CSMAR_BS_MAP)

    # Update streaming stats
    for vname, grp in long.groupby("variable"):
        vals = grp["value"]
        if vname not in var_stats:
            var_stats[vname] = {"n": 0, "sum": 0.0, "sum_sq": 0.0}
        var_stats[vname]["n"] += len(vals)
        var_stats[vname]["sum"] += vals.sum()
        var_stats[vname]["sum_sq"] += (vals ** 2).sum()

    if chunk_num % 5 == 0:
        print(f"    Chunk {chunk_num}: {len(var_stats)} variables tracked")

print(f"  Pass 1 done: {len(var_stats)} variables, {chunk_num} chunks")

# Compute mean/std
for vname in var_stats:
    n = var_stats[vname]["n"]
    var_stats[vname]["mean"] = var_stats[vname]["sum"] / n
    variance = var_stats[vname]["sum_sq"] / n - var_stats[vname]["mean"] ** 2
    var_stats[vname]["std"] = max(variance, 0) ** 0.5

# ── Pass 2: Z-score and save ─────────────────────────────────────
print("\n  Pass 2: Z-scoring and saving...")
out_dir_bs = OUT_DIR / "balance_sheet_chunks"
out_dir_bs.mkdir(exist_ok=True)
# Clean previous
for f in out_dir_bs.glob("*.parquet"):
    f.unlink()
chunk_out_num = 0

for chunk in pd.read_csv(BS_PATH, usecols=cols_to_read, chunksize=CHUNK_SIZE,
                         low_memory=False):
    chunk["Stkcd"] = chunk["Stkcd"].astype(str)

    chunk["date"] = pd.to_datetime(chunk["DeclareDate"], errors="coerce")
    mask = chunk["date"].isna()
    chunk.loc[mask, "date"] = pd.to_datetime(chunk.loc[mask, "Accper"], errors="coerce")
    chunk = chunk.dropna(subset=["date"])

    long = chunk.melt(id_vars=["Stkcd", "date"], value_vars=metric_cols,
                      var_name="var_code", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    long["variable"] = "bs_" + long["Stkcd"] + "::" + long["var_code"].map(CSMAR_BS_MAP)

    # Z-score
    long["value_raw"] = long["value"]
    for vname, grp in long.groupby("variable"):
        if vname in var_stats and var_stats[vname]["std"] > 0:
            idx = grp.index
            long.loc[idx, "value"] = (
                (long.loc[idx, "value"] - var_stats[vname]["mean"]) / var_stats[vname]["std"]
            )

    # Compute time_since_update
    long = long.sort_values(["variable", "date"])
    long["time_since_update"] = long.groupby("variable")["date"].diff().dt.days.fillna(0).astype(float)

    # Add source
    long["source"] = "financial"
    long["datetime"] = long["date"]
    cols_out = ["datetime", "source", "variable", "value", "value_raw", "time_since_update"]
    long = long[cols_out]

    # Save chunk as separate parquet
    chunk_out_num += 1
    out_chunk = out_dir_bs / f"chunk_{chunk_out_num:03d}.parquet"
    long.to_parquet(out_chunk, index=False)

print(f"  Pass 2 done. {chunk_out_num} chunks saved to {out_dir_bs}")

# ── Summary ──────────────────────────────────────────────────────
total_rows = 0
all_vars = set()
for f in sorted(out_dir_bs.glob("*.parquet")):
    df = pd.read_parquet(f)
    total_rows += len(df)
    all_vars.update(df["variable"].unique())

print(f"\n  Balance Sheet Results:")
print(f"    Rows:      {total_rows:,}")
print(f"    Variables: {len(all_vars):,}")
print(f"    Chunks:    {chunk_out_num}")
print(f"    Path:      {out_dir_bs}")
