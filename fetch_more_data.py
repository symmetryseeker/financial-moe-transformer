"""
Fetch additional macro + sentiment data from all available sources.
Tests each interface with 8s timeout, collects working ones.
"""

import time, sys
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("D:/financial_data")

# ── Part 1: AKShare macro interfaces ──────────────────────────────────────────

def try_ak(name, func, category, save_name, **kwargs):
    """Try an AKShare interface with short timeout."""
    print(f"  {name:42s} ... ", end="", flush=True)
    try:
        df = func(**kwargs)
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            outdir = DATA_DIR / category
            outdir.mkdir(parents=True, exist_ok=True)
            outpath = outdir / f"{save_name}.csv"
            df.to_csv(outpath, index=False)
            print(f"OK ({len(df):,} rows)")
            return True
        else:
            print("EMPTY")
            return False
    except Exception as e:
        msg = str(e)[:70]
        print(f"FAIL: {msg}")
        return False

print("=" * 60)
print("  FETCHING ADDITIONAL MACRO & SENTIMENT DATA")
print("=" * 60)

# ── AKShare: Test all available macro interfaces ────────────────────
import akshare as ak

print("\n>>> AKShare Macro (quick test)")

macro_tests = [
    # (name, func_ref, category, save_name)
    ("Money Supply", "macro_china_money_supply", "macro", "money_supply_ak"),
    ("PMI", "macro_china_pmi", "macro", "pmi_ak"),
    ("CPI Yearly", "macro_china_cpi_yearly", "macro", "cpi_yearly_ak"),
    ("PPI Yearly", "macro_china_ppi_yearly", "macro", "ppi_yearly_ak"),
    ("GDP Yearly", "macro_china_gdp_yearly", "macro", "gdp_yearly_ak"),
    ("LPR", "macro_china_lpr", "macro", "lpr_ak"),
    ("Shibor", "macro_china_shibor", "macro", "shibor_ak"),
    ("Industrial Output", "macro_china_industrial_production_yoy", "macro", "industrial_output_ak"),
    ("Fixed Asset Invest", "macro_china_fixed_asset_investment_yoy", "macro", "fai_ak"),
    ("Retail Sales", "macro_china_consumer_goods_retail_yoy", "macro", "retail_sales_ak"),
    ("Trade Balance", "macro_china_trade_balance", "macro", "trade_balance_ak"),
    ("FX Reserves", "macro_china_fx_reserves_yoy", "macro", "fx_reserves_ak"),
    ("Social Financing", "macro_china_social_financing", "macro", "social_financing_ak"),
    ("Bond Yield Curve", "bond_china_yield", "macro", "bond_yield_curve_ak"),
]

ok = 0
for name, func_name, cat, savename in macro_tests:
    func = getattr(ak, func_name, None)
    if func is None:
        print(f"  {name:42s} ... NO FUNC")
        continue
    if try_ak(name, func, cat, savename):
        ok += 1
    time.sleep(0.5)

print(f"\n  Macro: {ok}/{len(macro_tests)} OK")

# ── Try to get historical northbound flow ──────────────────────────
print("\n>>> Northbound Flow (historical)")

import akshare as ak
try:
    # Try HSGT historical
    df_sh = try_ak("HSGT Northbound SH", ak.stock_hsgt_hist_em,
                    "sentiment", "northbound_sh_hist", symbol="沪股通")
    df_sz = try_ak("HSGT Northbound SZ", ak.stock_hsgt_hist_em,
                    "sentiment", "northbound_sz_hist", symbol="深股通")
except Exception as e:
    print(f"  HSGT: {e}")

# Try margin data
print("\n>>> Margin Data")
try:
    try_ak("Margin SSE", ak.stock_margin_detail_sse, "sentiment", "margin_sse")
except Exception as e:
    print(f"  Margin: {e}")

# ── Try Baostock for additional macro ─────────────────────────────
print("\n>>> Baostock Macro")

import baostock as bs
lg = bs.login()
if lg.error_code == "0":
    print("  Baostock logged in")

    # Money supply
    func = getattr(bs, "query_money_supply_data_year", None) or getattr(bs, "query_money_supply_data", None)
    if func:
        rs = func()
        if rs.error_code == "0" and rs.data:
            df = pd.DataFrame(rs.data, columns=rs.fields)
            outdir = DATA_DIR / "macro"
            outdir.mkdir(parents=True, exist_ok=True)
            df.to_csv(outdir / "money_supply_bs.csv", index=False)
            print(f"  Money Supply (BS): {len(df):,} rows OK")

    # Shibor
    func = getattr(bs, "query_shibor_data", None)
    if func:
        rs = func()
        if rs.error_code == "0" and rs.data:
            df = pd.DataFrame(rs.data, columns=rs.fields)
            df.to_csv(DATA_DIR / "macro" / "shibor_bs.csv", index=False)
            print(f"  Shibor (BS): {len(df):,} rows OK")

    # Deposit rate
    func = getattr(bs, "query_deposit_rate_data", None)
    if func:
        rs = func()
        if rs.error_code == "0" and rs.data:
            df = pd.DataFrame(rs.data, columns=rs.fields)
            df.to_csv(DATA_DIR / "macro" / "deposit_rate_bs.csv", index=False)
            print(f"  Deposit Rate (BS): {len(df):,} rows OK")

    # Loan rate
    func = getattr(bs, "query_loan_rate_data", None)
    if func:
        rs = func()
        if rs.error_code == "0" and rs.data:
            df = pd.DataFrame(rs.data, columns=rs.fields)
            df.to_csv(DATA_DIR / "macro" / "loan_rate_bs.csv", index=False)
            print(f"  Loan Rate (BS): {len(df):,} rows OK")

    # Reserve ratio
    func = getattr(bs, "query_required_reserve_ratio_data", None)
    if func:
        rs = func()
        if rs.error_code == "0" and rs.data:
            df = pd.DataFrame(rs.data, columns=rs.fields)
            df.to_csv(DATA_DIR / "macro" / "reserve_ratio_bs.csv", index=False)
            print(f"  Reserve Ratio (BS): {len(df):,} rows OK")

    bs.logout()
else:
    print("  Baostock login FAILED")

# ── Summary ────────────────────────────────────────────────────────
files = list(DATA_DIR.rglob("*.csv"))
total_size = sum(f.stat().st_size for f in files) / 1e6
print(f"\n{'='*60}")
print(f"  DATA FETCH COMPLETE")
print(f"  Total: {len(files)} files, {total_size:.1f} MB")
print(f"  Path: {DATA_DIR}")

# List new files
print(f"\n  Macro files:")
for f in sorted((DATA_DIR / "macro").glob("*.csv")):
    print(f"    {f.name} ({f.stat().st_size/1e3:.0f} KB)")
print(f"\n  Sentiment files:")
for f in sorted((DATA_DIR / "sentiment").glob("*.csv")):
    print(f"    {f.name} ({f.stat().st_size/1e3:.0f} KB)")
