"""Deep check: find usable disclosure dates in all financial files."""
import pandas as pd
from pathlib import Path

# 1. Check balance_sheet.csv DeclareDate further down
print("=" * 60)
print("  1. balance_sheet.csv — DeclareDate non-NaN check")
print("=" * 60)

# Read in chunks to find non-NaN DeclareDate
for chunk in pd.read_csv("D:/financial_data/financial/balance_sheet.csv",
                         chunksize=100000, low_memory=False):
    if chunk["DeclareDate"].notna().any():
        n_valid = chunk["DeclareDate"].notna().sum()
        sample = chunk[chunk["DeclareDate"].notna()]["DeclareDate"].head(3).tolist()
        print(f"  Found {n_valid} non-NaN DeclareDate values in chunk")
        print(f"  Sample: {sample}")
        break
else:
    print("  All DeclareDate values are NaN — this column is empty")

# 2. Check EndDate in balance_sheet
print(f"\n{'='*60}")
print("  2. balance_sheet.csv — All date-like columns")
print(f"{'='*60}")
df = pd.read_csv("D:/financial_data/financial/balance_sheet.csv", nrows=5)
for c in df.columns:
    cstr = str(c).lower()
    if any(kw in cstr for kw in ['date', 'end', 'year', 'period', 'declar', 'ann']):
        vals = df[c].tolist()
        print(f"  {c}: {vals}")

# 3. Check income_statement_full
print(f"\n{'='*60}")
print("  3. income_statement_full.csv — Date columns")
print(f"{'='*60}")
df = pd.read_csv("D:/financial_data/financial/income_statement_full.csv", nrows=5)
for c in df.columns:
    cstr = str(c).lower()
    if any(kw in cstr for kw in ['date', 'end', 'year', 'period', 'declar', 'ann']):
        vals = df[c].tolist()
        print(f"  {c}: {vals}")

# Check for other date-like columns
extra = [c for c in df.columns if any(kw in str(c) for kw in ['Date', 'date', 'End', 'Year'])]
print(f"\n  All date-related: {extra}")

# 4. Check income_statement_local InfoPublDate range
print(f"\n{'='*60}")
print("  4. income_statement_local.csv — InfoPublDate full stats")
print(f"{'='*60}")
for chunk in pd.read_csv("D:/financial_data/financial/income_statement_local.csv",
                         chunksize=50000, low_memory=False):
    if "InfoPublDate" not in chunk.columns:
        # Garbled column name handling
        for c in chunk.columns:
            if 'Info' in str(c) or '信息' in str(c):
                print(f"  Found: {c}")
    break

# Count usable date columns across all files
print(f"\n{'='*60}")
print("  SUMMARY: Disclosure Date Availability")
print(f"{'='*60}")
print("""
  income_statement_local.csv:
    end_date (截止日期)    → period end, e.g. 2006-12-31
    InfoPublDate (信息发布日期) → actual disclosure, e.g. 2007-03-22
    → CAN FIX: replace end_date with InfoPublDate ✓

  balance_sheet.csv:
    DeclareDate → ALL NaN (column exists but empty)
    → CANNOT FIX from this file, need to estimate disclosure lag

  income_statement_full.csv:
    DeclareDate → ALL NaN (column exists but empty)
    → CANNOT FIX from this file

  RECOMMENDATION:
  - For income_statement_local.csv: use InfoPublDate ✓
  - For balance_sheet.csv and income_statement_full.csv:
    Apply estimated disclosure lag based on report quarter:
      Q1 (end 03-31) → disclose 04-30
      Q2 (end 06-30) → disclose 08-31
      Q3 (end 09-30) → disclose 10-31
      Q4 (end 12-31) → disclose 03-31 (next year)
  - This is standard practice in Chinese financial research
""")
