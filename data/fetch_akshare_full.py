"""
AKShare FULL data collection on DOMESTIC network.
Covers: PMI sub-items, all futures, social financing detail, etc.
"""
import time, sys
from pathlib import Path
import pandas as pd
import akshare as ak

sys.path.insert(0, str(Path(__file__).parent.parent))
DATA_DIR = Path("D:/financial_data")
ok = 0; fail = 0

def fetch(name, func, category, save_name, **kwargs):
    global ok, fail
    print(f"  {name:45s} ... ", end="", flush=True)
    try:
        df = func(**kwargs) if callable(func) else func
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            outdir = DATA_DIR / category; outdir.mkdir(parents=True, exist_ok=True)
            df.to_csv(outdir / f"{save_name}.csv", index=False)
            print(f"OK ({len(df):,} rows)")
            ok += 1; return True
        else:
            print("EMPTY"); fail += 1; return False
    except Exception as e:
        print(f"FAIL: {str(e)[:70]}"); fail += 1; return False

print("=" * 60)
print("  AKShare FULL — DOMESTIC NETWORK")
print("=" * 60)

# ── 1. PMI Sub-indices ──────────────────────────────────────────
print("\n[1] PMI Sub-Indices")
try:
    df = ak.macro_china_pmi()
    if isinstance(df, pd.DataFrame) and len(df) > 0:
        fetch("PMI Detail", lambda: df, "macro", "pmi_detail_full")
except: pass

try:
    df = ak.macro_china_cx_pmi()  # Caixin PMI
    fetch("Caixin PMI", lambda: df, "macro", "caixin_pmi")
except: pass

try:
    df = ak.macro_china_pmi_yearly()
    fetch("PMI Yearly", lambda: df, "macro", "pmi_yearly")
except: pass

# ── 2. Social Financing Detail ───────────────────────────────────
print("\n[2] Social Financing Detail")
try:
    df = ak.macro_china_shrzgm()  # 社会融资规模增量
    fetch("Social Financing", lambda: df, "macro", "social_financing")
except: pass

# ── 3. ALL Commodity Futures ─────────────────────────────────────
print("\n[3] Commodity Futures (full set)")

futures_map = {
    "RB": "rebar", "CU": "copper", "AU": "gold", "SC": "crude_oil",
    "AL": "aluminum", "ZN": "zinc", "MA": "methanol", "TA": "pta",
    "M": "soymeal", "Y": "soyoil", "CF": "cotton", "SR": "sugar",
    "RU": "rubber", "FU": "fuel_oil", "BU": "asphalt",
    "IF": "csi300_futures", "IC": "csi500_futures",
    "T": "treasury_10y", "TF": "treasury_5y",
}

for code, name in futures_map.items():
    try:
        df = ak.futures_main_sina(symbol=code)
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            df["contract"] = code
            fetch(f"Futures {name}", lambda d=df: d, "alternative", f"futures_{name}")
    except Exception as e:
        fetch(f"Futures {name}", lambda: None, "alternative", f"futures_{name}")
    time.sleep(0.3)

# ── 4. Additional Macro ──────────────────────────────────────────
print("\n[4] Additional Macro")

try:
    df = ak.macro_china_industrial_production_yoy()
    fetch("Industrial Production", lambda: df, "macro", "industrial_production")
except: pass

try:
    df = ak.macro_china_consumer_goods_retail_yoy()
    fetch("Retail Sales", lambda: df, "macro", "retail_sales")
except: pass

try:
    df = ak.macro_china_fixed_asset_investment_yoy()
    fetch("Fixed Asset Investment", lambda: df, "macro", "fixed_asset_invest")
except: pass

try:
    df = ak.macro_china_trade_balance()
    fetch("Trade Balance", lambda: df, "macro", "trade_balance")
except: pass

try:
    df = ak.macro_china_fx_reserves_yoy()
    fetch("FX Reserves", lambda: df, "macro", "fx_reserves")
except: pass

try:
    df = ak.macro_china_cpi_monthly()
    fetch("CPI Monthly", lambda: df, "macro", "cpi_monthly")
except: pass

try:
    df = ak.macro_china_ppi()
    fetch("PPI Detail", lambda: df, "macro", "ppi_detail")
except: pass

# ── Summary ──────────────────────────────────────────────────────
files = list(DATA_DIR.rglob("*.csv"))
size = sum(f.stat().st_size for f in files) / 1e6
print(f"\n{'='*60}")
print(f"  AKShare FULL: {ok} OK, {fail} FAILED")
print(f"  Total: {len(files)} files, {size:.1f} MB")
