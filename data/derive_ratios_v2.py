"""
Derive financial ratios v2 — uses raw balance sheet CSV directly.

Computes ~15 ratios per company per report date from balance sheet columns.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
DATA_DIR = Path("D:/financial_data")

print("=" * 60)
print("  DERIVING FINANCIAL RATIOS (v2 — from raw CSV)")
print("=" * 60)

# Load balance sheet
bs_path = DATA_DIR / "financial" / "balance_sheet.csv"
if not bs_path.exists():
    print("  balance_sheet.csv not found")
    exit(1)

# Read column names
df_header = pd.read_csv(bs_path, nrows=0)
cols = list(df_header.columns)
print(f"  Balance sheet: {len(cols)} columns")

# CSMAR Balance Sheet standard codes (CSMAR database)
CSMAR_BS_CODES = {
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
    "A002201000": "short_term_debt",
    "A003101000": "paid_in_capital",
    "A003105000": "capital_reserve",
    "A003106000": "retained_earnings",
    "A001218000": "intangible_assets",
}

CSMAR_IS_CODES = {
    "B001100000": "revenue",
    "B001101000": "operating_revenue",
    "B001200000": "operating_cost",
    "B001300000": "operating_profit",
    "B002000000": "net_profit",
    "B002500000": "eps",
    "B001210000": "selling_expense",
    "B001220000": "admin_expense",
    "B001230000": "rd_expense",
    "B001240000": "interest_expense",
}

# Map by CSMAR code
col_map = {}
for c in cols:
    if c in CSMAR_BS_CODES:
        col_map[CSMAR_BS_CODES[c]] = c

print(f"  Balance sheet: {len(col_map)}/{len(CSMAR_BS_CODES)} CSMAR codes mapped")
for k, v in sorted(col_map.items()):
    print(f"    {k}: {v}")

# ── Now load data in chunks and compute ratios ────────────────────
# We also need income statement data. Let's load the full income statement.
is_path = DATA_DIR / "financial" / "income_statement_full.csv"
is_header = pd.read_csv(is_path, nrows=0)

# Map IS columns by CSMAR code
is_col_map = {}
for c in is_header.columns:
    if c in CSMAR_IS_CODES:
        is_col_map[CSMAR_IS_CODES[c]] = c

print(f"\n  Income statement: {len(is_col_map)}/{len(CSMAR_IS_CODES)} CSMAR codes mapped")
for k, v in sorted(is_col_map.items()):
    print(f"    {k}: {v}")

# ── Compute ratios per chunk ──────────────────────────────────────
print("\n  Computing ratios in chunks...")

all_ratios = []
chunk_count = 0
date_col = "DeclareDate" if "DeclareDate" in cols else None
# Also check for EndDate
end_date_col = None
for c in cols:
    if "EndDate" in str(c) or "EndDt" in str(c):
        end_date_col = c
        break

# Read BS in chunks
for bs_chunk in pd.read_csv(bs_path, chunksize=100000, low_memory=False):
    chunk_count += 1

    # Use DeclareDate if available, else EndDate
    date_col_actual = date_col if (date_col and bs_chunk[date_col].notna().any()) else end_date_col
    if not date_col_actual:
        continue

    # Rename columns to standard names
    rename_map = {v: k for k, v in col_map.items() if v in bs_chunk.columns}
    bs = bs_chunk.rename(columns=rename_map)

    # Ensure numeric
    for c in rename_map.values():
        bs[c] = pd.to_numeric(bs[c], errors="coerce")

    # Filter rows with date
    bs["date"] = pd.to_datetime(bs[date_col_actual], errors="coerce")
    bs = bs.dropna(subset=["date"])

    # Compute ratios
    ratios = []
    for _, row in bs.iterrows():
        r = {"date": row["date"]}

        def div(a, b):
            return a / b if (pd.notna(a) and pd.notna(b) and b != 0) else np.nan

        # Profitability
        ta = row.get("total_assets")
        te = row.get("total_equity")
        tl = row.get("total_liab")
        ca = row.get("total_cur_assets")
        cl = row.get("total_cur_liab")

        if ta and te:
            r["debt_to_assets"] = div(tl, ta) if tl else np.nan
            r["equity_to_assets"] = div(te, ta)
            r["debt_to_equity"] = div(tl, te) if tl else np.nan

        if ca and cl:
            r["current_ratio"] = div(ca, cl)
            r["quick_ratio"] = div(row.get("cash", np.nan), cl)

        r["tangible_ratio"] = div(row.get("fixed_assets", np.nan), ta) if ta else np.nan

        ratios.append(r)

    all_ratios.extend(ratios)
    if chunk_count % 2 == 0:
        print(f"    Chunk {chunk_count}: {len(all_ratios):,} ratios so far")

print(f"  Total: {len(all_ratios):,} ratio records")

# ── Convert to long format and save ───────────────────────────────
if all_ratios:
    ratios_df = pd.DataFrame(all_ratios)
    ratio_cols = [c for c in ratios_df.columns if c != "date"]
    long = ratios_df.melt(id_vars=["date"], value_vars=ratio_cols,
                          var_name="variable", value_name="value")
    long = long.dropna(subset=["value"])
    long["source"] = "financial_ratio"
    long["time_since_update"] = 0.0

    out_path = Path("data/processed/financial_ratios.parquet")
    long.to_parquet(out_path, index=False)
    print(f"\n  Saved: {out_path} ({len(long):,} data points)")

    for rc in ratio_cols:
        vals = long[long["variable"] == rc]["value"]
        if len(vals) > 0:
            print(f"    {rc}: n={len(vals):>6,}, med={vals.median():.3f}, std={vals.std():.3f}")
