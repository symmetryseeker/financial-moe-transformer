"""Check if financial data has actual disclosure dates (not just period end dates)."""
import pandas as pd
from pathlib import Path

print("=" * 60)
print("  CHECKING DISCLOSURE DATE COLUMNS")
print("=" * 60)

files = [
    ("D:/financial_data/financial/balance_sheet.csv", "资产负债表"),
    ("D:/financial_data/financial/income_statement_full.csv", "利润表(完整版)"),
    ("D:/financial_data/financial/income_statement_local.csv", "利润表(本地)"),
]

for path, label in files:
    p = Path(path)
    if not p.exists():
        print(f"\n  {label}: NOT FOUND")
        continue

    print(f"\n  [{label}] {p.name} ({p.stat().st_size/1e6:.0f} MB)")

    # Read header only
    df = pd.read_csv(p, nrows=0)
    cols = list(df.columns)
    print(f"  Total columns: {len(cols)}")

    # Find date-related columns
    date_cols = [c for c in cols if any(kw in str(c).lower() for kw in
        ['date', 'declar', 'publ', 'info', 'ann', '披露', '公布', '公告', '申报', 'end'])]

    if date_cols:
        print(f"  Date-related columns:")
        for c in date_cols[:15]:
            print(f"    - {c}")
    else:
        print(f"  NO date-related columns found!")

    # Read first 3 rows to check values
    df_sample = pd.read_csv(p, nrows=3)
    for c in date_cols[:5]:
        vals = df_sample[c].tolist()
        print(f"    {c}: {vals}")

print(f"\n{'='*60}")
print("  CHECKING: Does data use period-end or disclosure dates?")
print("=" * 60)
print("""
  If the balance sheet has 'DeclareDate' or 'InfoPublDate' columns,
  we can fix the look-ahead bias by using disclosure date instead
  of period end date (EndDate).

  Expected columns from CSMAR:
  - EndDate (截止日期) = period end date (e.g., 2024-03-31 for Q1)
  - DeclareDate (披露日期) = actual disclosure date (e.g., 2024-04-25)
  - InfoPublDate (信息发布日期) = when the public saw it
""")
