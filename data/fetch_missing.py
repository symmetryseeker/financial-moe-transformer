"""
Fetch missing data identified in the audit:
1. Social financing (社融) — AKShare
2. International macro (VIX, Fed rate, USD index) — yfinance
3. SZSE margin trading — AKShare
4. Commodity futures (copper, oil, gold, rebar) — AKShare
5. PMI sub-indices — AKShare
"""

import time, sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
DATA_DIR = Path("D:/financial_data")

def save(df, category, name):
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return False
    outdir = DATA_DIR / category
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{name}.csv"
    df.to_csv(outpath, index=False)
    print(f"    -> {outpath} ({len(df)} rows)")
    return True

print("=" * 60)
print("  FETCHING MISSING DATA")
print("=" * 60)

# ── 1. Social Financing (社融) via AKShare ───────────────────────
print("\n[1] Social Financing (社融)")

import akshare as ak
try:
    df = ak.macro_china_shrzgm()
    if isinstance(df, pd.DataFrame) and len(df) > 0:
        save(df, "macro", "social_financing_ak")
        print(f"    OK: {len(df)} rows, cols={list(df.columns)[:8]}")
except Exception as e:
    print(f"    FAIL: {e}")

# ── 2. International macro via yfinance ─────────────────────────
print("\n[2] International Macro (VIX, Fed, USD)")

try:
    import yfinance as yf

    # VIX
    vix = yf.download("^VIX", start="2015-01-01", end="2025-12-31", progress=False)
    if not vix.empty:
        vix_flat = vix.reset_index()
        vix_flat.columns = ["date"] + [f"vix_{c.lower()}" for c in vix_flat.columns[1:]]
        save(vix_flat, "macro", "vix_daily")

    # USD/CNY
    usdcny = yf.download("CNY=X", start="2015-01-01", end="2025-12-31", progress=False)
    if not usdcny.empty:
        usdcny_flat = usdcny.reset_index()
        usdcny_flat.columns = ["date"] + [f"usdcny_{c.lower()}" for c in usdcny_flat.columns[1:]]
        save(usdcny_flat, "macro", "usdcny_daily")

    # 10Y US Treasury
    us10y = yf.download("^TNX", start="2015-01-01", end="2025-12-31", progress=False)
    if not us10y.empty:
        us10y_flat = us10y.reset_index()
        us10y_flat.columns = ["date"] + [f"us10y_{c.lower()}" for c in us10y_flat.columns[1:]]
        save(us10y_flat, "macro", "us10y_daily")

    # S&P 500
    spx = yf.download("^GSPC", start="2015-01-01", end="2025-12-31", progress=False)
    if not spx.empty:
        spx_flat = spx.reset_index()
        spx_flat.columns = ["date"] + [f"spx_{c.lower()}" for c in spx_flat.columns[1:]]
        save(spx_flat, "macro", "sp500_daily")

    print("    OK: VIX + USDCNY + US10Y + SP500")
except Exception as e:
    print(f"    yfinance FAIL: {e}")

# ── 3. SZSE Margin Trading ──────────────────────────────────────
print("\n[3] SZSE Margin Trading")

try:
    df = ak.stock_margin_detail_szse(date="20240517")
    if isinstance(df, pd.DataFrame) and len(df) > 0:
        save(df, "sentiment", "margin_szse")
        print(f"    OK: {len(df)} rows")
except Exception as e:
    print(f"    FAIL: {e}")

# ── 4. Commodity Futures ────────────────────────────────────────
print("\n[4] Commodity Futures (main contracts)")

futures_contracts = {
    "RB": "rebar",       # 螺纹钢
    "CU": "copper",      # 铜
    "AU": "gold",        # 黄金
    "SC": "crude_oil",   # 原油
    "AL": "aluminum",    # 铝
    "ZN": "zinc",        # 锌
    "MA": "methanol",    # 甲醇
    "TA": "pta",         # PTA
    "M": "soymeal",      # 豆粕
    "Y": "soyoil",       # 豆油
}

for code, name in futures_contracts.items():
    try:
        df = ak.futures_main_sina(symbol=code)
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            df["contract"] = code
            save(df, "alternative", f"futures_{name}")
    except Exception as e:
        print(f"    {code}({name}): {str(e)[:50]}")

# ── Summary ──────────────────────────────────────────────────────
files = list(DATA_DIR.rglob("*.csv"))
total_size = sum(f.stat().st_size for f in files) / 1e6
print(f"\n{'='*60}")
print(f"  DONE: {len(files)} files, {total_size:.1f} MB")
