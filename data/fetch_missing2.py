"""Fetch remaining missing data — yfinance + alternative sources."""
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
print("  FETCHING MISSING DATA (round 2)")
print("=" * 60)

# ── 1. International Macro via yfinance ─────────────────────────
print("\n[1] International Macro (yfinance)")

try:
    import yfinance as yf

    # VIX
    vix = yf.download("^VIX", start="2015-01-01", progress=False)
    if not vix.empty:
        df = vix.reset_index()
        df.columns = ["date"] + [f"vix_{str(c).lower()}" for c in vix.columns]
        save(df, "macro", "vix_daily")

    # USD Index
    dxy = yf.download("DX-Y.NYB", start="2015-01-01", progress=False)
    if not dxy.empty:
        df = dxy.reset_index()
        df.columns = ["date"] + [f"dxy_{str(c).lower()}" for c in dxy.columns]
        save(df, "macro", "dxy_daily")

    # 10Y US Treasury
    us10y = yf.download("^TNX", start="2015-01-01", progress=False)
    if not us10y.empty:
        df = us10y.reset_index()
        df.columns = ["date"] + [f"us10y_{str(c).lower()}" for c in us10y.columns]
        save(df, "macro", "us10y_daily")

    # S&P 500
    spx = yf.download("^GSPC", start="2015-01-01", progress=False)
    if not spx.empty:
        df = spx.reset_index()
        df.columns = ["date"] + [f"spx_{str(c).lower()}" for c in spx.columns]
        save(df, "macro", "sp500_daily")

    # CSI 300 (from yfinance as backup)
    csi300 = yf.download("000300.SS", start="2015-01-01", progress=False)
    if not csi300.empty:
        df = csi300.reset_index()
        df.columns = ["date"] + [f"csi300_yf_{str(c).lower()}" for c in csi300.columns]
        save(df, "market", "csi300_yfinance")

    # HSI (Hang Seng)
    hsi = yf.download("^HSI", start="2015-01-01", progress=False)
    if not hsi.empty:
        df = hsi.reset_index()
        df.columns = ["date"] + [f"hsi_{str(c).lower()}" for c in hsi.columns]
        save(df, "macro", "hsi_daily")

    print("    OK: VIX + DXY + US10Y + SP500 + CSI300 + HSI")
except Exception as e:
    print(f"    FAIL: {e}")

# ── 2. Social Financing via Baostock ────────────────────────────
print("\n[2] Social Financing (via Baostock)")

import baostock as bs
lg = bs.login()
if lg.error_code == "0":
    # Try various macro query functions
    for func_name in ["query_sf_data", "query_macro_data", "query_social_financing_data"]:
        func = getattr(bs, func_name, None)
        if func:
            try:
                rs = func()
                if rs.error_code == "0" and rs.data:
                    df = pd.DataFrame(rs.data, columns=rs.fields)
                    save(df, "macro", f"social_financing_bs")
                    print(f"    OK via {func_name}")
                    break
            except Exception as e:
                print(f"    {func_name}: {e}")
    else:
        print("    No social financing function found in Baostock")
    bs.logout()

# ── 3. Commodity futures via AKShare (non-Sina source) ──────────
print("\n[3] Commodity Futures (attempting alternative sources)")

import akshare as ak
# Try futures_em (EastMoney-based, might work)
for code, name in [("RB0", "rebar"), ("CU0", "copper"), ("AU0", "gold")]:
    try:
        df = ak.futures_main_sina(symbol=code)
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            df["contract"] = code
            save(df, "alternative", f"futures_{name}")
            print(f"    {name}: OK")
    except:
        pass

# Try futures_foreign (COMEX, LME)
try:
    df = ak.futures_foreign_hist(symbol="黄金")
    if isinstance(df, pd.DataFrame) and len(df) > 0:
        save(df, "alternative", "futures_gold_comex")
        print(f"    gold COMEX: OK")
except Exception as e:
    print(f"    COMEX gold: {str(e)[:50]}")

# Summary
files = list(DATA_DIR.rglob("*.csv"))
total_size = sum(f.stat().st_size for f in files) / 1e6
print(f"\n{'='*60}")
print(f"  DONE: {len(files)} files, {total_size:.1f} MB")
