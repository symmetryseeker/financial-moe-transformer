"""Check Stkcd format in balance sheet vs CSI300 codes."""
import pandas as pd

# Balance sheet Stkcd sample
bs = pd.read_csv("D:/financial_data/financial/balance_sheet.csv",
                 usecols=["Stkcd"], nrows=20)
print("Balance Sheet Stkcd (first 20):")
print(bs["Stkcd"].unique()[:20])
print(f"  Type: {bs['Stkcd'].dtype}")

# CSI300 codes
for path in ["D:/financial_data/market/csi300_constituents.csv",
             "D:/financial_data/market/csi300_stock_list.csv"]:
    df = pd.read_csv(path, nrows=10)
    print(f"\n{path}:")
    print(f"  Cols: {list(df.columns)}")
    for c in df.columns:
        print(f"  [{c}]: {df[c].head(5).tolist()}")
