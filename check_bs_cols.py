"""Check balance sheet column names — find the actual data columns."""
import pandas as pd

# Read balance sheet column names
df = pd.read_csv("D:/financial_data/financial/balance_sheet.csv", nrows=3)
print("Balance Sheet columns (first 30):")
for i, c in enumerate(df.columns[:30]):
    sample = str(df[c].iloc[0])[:40] if len(df) > 0 else ""
    print(f"  [{i:3d}] {c[:60]:60s}  sample={sample}")

print(f"\n... total {len(df.columns)} columns")

# Check: which columns have numeric values (not NaN) in first row?
print(f"\nColumns with non-NaN values in first 3 rows:")
for c in df.columns:
    n_valid = df[c].notna().sum()
    if n_valid > 0:
        sample = str(df[c].iloc[0])[:50]
        print(f"  [{n_valid}/3] {c[:50]:50s}  {sample}")

# Check DeclareDate specifically
print(f"\nDeclareDate check:")
for i in range(3):
    print(f"  row {i}: DeclareDate={df['DeclareDate'].iloc[i]}")

# Check if there are EndDt/EndDate type columns
for c in df.columns:
    if "end" in str(c).lower() or "End" in str(c):
        print(f"  END COL: {c} = {df[c].iloc[0]}")

# Also check income_statement_full
print(f"\n\nIncome Statement Full columns:")
df2 = pd.read_csv("D:/financial_data/financial/income_statement_full.csv", nrows=3)
for i, c in enumerate(df2.columns[:20]):
    sample = str(df2[c].iloc[0])[:40] if len(df2) > 0 else ""
    print(f"  [{i:3d}] {c[:60]:60s}  sample={sample}")
print(f"  ... total {len(df2.columns)} columns")
