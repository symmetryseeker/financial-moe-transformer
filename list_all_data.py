"""Complete data inventory — every file in D:/financial_data/ with details."""
import pandas as pd
from pathlib import Path
import os

DATA_DIR = Path("D:/financial_data")

print("=" * 90)
print("  COMPLETE DATA INVENTORY — D:/financial_data/")
print("=" * 90)

total_files = 0
total_rows = 0
total_size = 0

for cat_dir in sorted(DATA_DIR.iterdir()):
    if not cat_dir.is_dir():
        continue
    cat_files = sorted(cat_dir.rglob("*.csv"))
    if not cat_files:
        continue

    cat_rows = 0
    cat_size = 0
    print(f"\n{'─'*90}")
    print(f"  [{cat_dir.name.upper()}]  ({len(cat_files)} files)")
    print(f"{'─'*90}")
    print(f"  {'File':<45s} {'Rows':>10s} {'Cols':>6s} {'Size':>8s}  {'Date Range'}")
    print(f"  {'-'*45} {'-'*10} {'-'*6} {'-'*8}  {'-'*30}")

    for f in cat_files:
        size_mb = f.stat().st_size / 1e6
        cat_size += size_mb
        total_size += size_mb
        total_files += 1

        try:
            df = pd.read_csv(f, nrows=0)
            n_cols = len(df.columns)
            cols_preview = ", ".join(df.columns[:6].tolist())
            if len(df.columns) > 6:
                cols_preview += f", ... +{len(df.columns)-6}"

            # Get full row count
            try:
                # Fast line count for large files
                with open(f, 'r', encoding='utf-8', errors='replace') as fh:
                    n_rows = sum(1 for _ in fh) - 1  # minus header
            except:
                n_rows = 0
            cat_rows += n_rows

            # Try to get date range
            try:
                df_full = pd.read_csv(f, nrows=3)
                date_col = None
                for c in df_full.columns:
                    if any(kw in str(c).lower() for kw in
                           ['date', 'time', '日期', '月份', 'trade', 'end', 'year', 'stat']):
                        date_col = c
                        break
                if date_col:
                    sample = str(df_full[date_col].iloc[0])
                else:
                    sample = str(df_full.iloc[0, 0])
                date_info = f"[{sample[:30]}]"
            except:
                date_info = ""

            print(f"  {f.name:<45s} {n_rows:>10,} {n_cols:>6} {size_mb:>7.1f}MB  {date_info}")

        except Exception as e:
            print(f"  {f.name:<45s} {'?':>10} {'?':>6} {size_mb:>7.1f}MB  ERROR: {str(e)[:30]}")

    print(f"  {'─'*45} {'─'*10} {'─'*6} {'─'*8}")
    print(f"  CATEGORY TOTAL: {cat_rows:>10,} rows, {cat_size:>7.1f} MB")

total_rows = "—"  # too slow to count all
print(f"\n{'='*90}")
print(f"  GRAND TOTAL: {total_files} files, {total_size:.1f} MB")
print(f"  Categories: market, financial, macro, alternative, sentiment, text")
print(f"{'='*90}")
