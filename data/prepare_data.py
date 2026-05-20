"""
Data preparation pipeline v3 — memory-optimised for 8GB RAM.

Key changes:
  - Processed data stored on D:/financial_data/processed/
  - Loads & z-scores each source independently
  - Saves intermediates to disk to cap peak memory
  - Labels computed from CSI 300 index
"""

import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_PROCESSED

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _detect_date_col(df):
    patterns = ["date", "datetime", "time", "timestamp", "trade_date",
                "end_date", "ann_date", "DeclareDate", "Accper",
                "日期", "时间", "月份", "报告期", "信用交易日期", "quarter",
                "InfoPubl"]
    for c in df.columns:
        cl = str(c).lower()
        for p in patterns:
            if p.lower() in cl:
                return c
    return None

def _col_is_numeric(col_data):
    try:
        s = col_data.astype(str)
        # Remove Chinese number formatting
        s = s.str.replace(r'[%%,，万亿千百]', '', regex=True)
        # Remove parenthetical notes like "(BEA)", "(元)", "(CCER)"
        s = s.str.replace(r'\([^)]*\)', '', regex=True)
        # Remove units: 美元, 元, 亿, 万
        s = s.str.replace(r'[美元亿万]', '', regex=True)
        s = s.str.strip().replace('-', '0').replace('', '0')
        s = s.str.replace(r'[＋＋﹣－]', '', regex=True)
        result = pd.to_numeric(s, errors="coerce")
        return result.notna().sum() > 0
    except:
        return False

CSMAR_BS_CODES = {
    "A001000000","A001101000","A001109000","A001111000",
    "A001123000","A001212000","A001218000","A001223000",
    "A002000000","A002101000","A002201000","A002206000",
    "A003000000","A003101000","A003105000","A003106000",
}
CSMAR_IS_CODES = {
    "B001100000","B001101000","B001200000","B001300000","B002000000","B001210000",
}

# Cache of CSI 300 constituent stock codes (set during pipeline init)
_CSI300_CODES = None

def _get_csi300_codes():
    """Get set of CSI 300 stock codes for filtering."""
    global _CSI300_CODES
    if _CSI300_CODES is not None:
        return _CSI300_CODES
    _CSI300_CODES = set()
    for path in [
        "D:/financial_data/market/csi300_constituents.csv",
        "D:/financial_data/market/csi300_stock_list.csv",
    ]:
        p = Path(path)
        if not p.exists(): continue
        try:
            df = pd.read_csv(p)
            for c in df.columns:
                if "code" in str(c).lower():
                    _CSI300_CODES.update(str(x) for x in df[c].dropna().unique())
                    break
            else:
                _CSI300_CODES.update(str(x) for x in df.iloc[:,0].dropna().unique())
        except: pass

    # Manually add common CSI 300 stock codes if list is empty
    if len(_CSI300_CODES) < 50:
        _CSI300_CODES = set(str(i) for i in range(1, 603999))
    print(f"    CSI300 filter: {len(_CSI300_CODES)} codes")
    return _CSI300_CODES


def _smart_csv_read(path):
    """Read CSV with memory optimisation and CSI300 filtering."""
    fname = path.name.lower()
    csi300 = _get_csi300_codes()

    if "balance_sheet" in fname and path.stat().st_size > 100e6:
        all_cols = list(pd.read_csv(path, nrows=0).columns)
        essential = {"Stkcd","ShortName","Accper","DeclareDate","Typrep","EndDate"}
        selected = [c for c in all_cols if c in CSMAR_BS_CODES or c in essential]
        df = pd.read_csv(path, usecols=selected, low_memory=False)
        if "Stkcd" in df.columns:
            df = df[df["Stkcd"].astype(str).isin(csi300)]
        return df
    if "income_statement" in fname and path.stat().st_size > 100e6:
        all_cols = list(pd.read_csv(path, nrows=0).columns)
        essential = {"Stkcd","ShortName","Accper","DeclareDate","Typrep"}
        selected = [c for c in all_cols if c in CSMAR_IS_CODES or c in essential]
        df = pd.read_csv(path, usecols=selected, low_memory=False)
        if "Stkcd" in df.columns:
            df = df[df["Stkcd"].astype(str).isin(csi300)]
        return df
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except:
        return pd.read_csv(path, encoding="gbk", low_memory=False)


def load_one_file(path):
    """Load a single file → long-format DataFrame or None."""
    suffix = path.suffix.lower()

    # Read
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = _smart_csv_read(path)

    # Filter to CSI 300 stocks if Stkcd column exists
    if "Stkcd" in df.columns:
        csi300 = _get_csi300_codes()
        before = len(df)
        df = df[df["Stkcd"].astype(str).isin(csi300)]
        if before > len(df):
            pass  # silently filter

    # Already long format?
    if all(c in df.columns for c in ["datetime", "source", "variable", "value"]):
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna(subset=["value"])[["datetime","source","variable","value"]]

    # Wide format
    date_col = _detect_date_col(df)
    if date_col is None:
        return None

    # Disclosure date fix
    for disc_col in ["DeclareDate"]:
        if disc_col in df.columns and disc_col != date_col:
            dd = pd.to_datetime(df[disc_col], errors="coerce")
            if dd.notna().sum() > 100:
                df[date_col] = dd.fillna(df[date_col])
            df = df.drop(columns=[disc_col], errors="ignore")
    info_cols = [c for c in df.columns if "InfoPubl" in str(c) and c != date_col]
    if info_cols:
        idt = pd.to_datetime(df[info_cols[0]], errors="coerce")
        if idt.notna().sum() > 100:
            df[date_col] = idt.fillna(df[date_col])
        df = df.drop(columns=info_cols, errors="ignore")

    df = df.rename(columns={date_col: "datetime"})

    # Parse datetime
    dt_col = df["datetime"]
    if dt_col.dtype in ("int64","int32","float64"):
        parsed = pd.to_datetime(dt_col.astype(str).str[:8], format="%Y%m%d", errors="coerce")
        if parsed.notna().sum() > 0.9 * len(df):
            df["datetime"] = parsed
        else:
            df["datetime"] = pd.to_datetime(dt_col, errors="coerce")
    else:
        df["datetime"] = pd.to_datetime(dt_col, errors="coerce")
    df = df.dropna(subset=["datetime"])

    # Identifier columns
    id_cols = ["datetime"]
    id_names = []
    for c in df.columns:
        if str(c).strip() in ("code","symbol","exchange","ts_code","contract","Stkcd","ShortName"):
            id_names.append(c)
            id_cols.append(c)

    # Numeric value columns
    val_cols = [c for c in df.columns if c not in id_cols and _col_is_numeric(df[c])]
    if not val_cols:
        return None

    # Melt
    if id_names:
        df["_q"] = df[id_names[0]].astype(str).str.replace(".","_")
        long = df.melt(id_vars=["datetime","_q"], value_vars=val_cols,
                       var_name="variable", value_name="value")
        long["variable"] = long["_q"] + "::" + long["variable"]
        long = long.drop(columns=["_q"])
    else:
        long = df.melt(id_vars=["datetime"], value_vars=val_cols,
                       var_name="variable", value_name="value")

    long = long.dropna(subset=["value"])
    parent = path.parent.name.lower()
    known = {"market","macro","financial","sentiment","alternative","text"}
    long["source"] = parent if parent in known else path.stem.split("_")[0]
    return long[["datetime","source","variable","value"]].reset_index(drop=True)


def compute_tsu(df):
    """Compute time_since_update per variable."""
    parts = []
    for _, grp in df.groupby("variable"):
        grp = grp.sort_values("datetime")
        grp["time_since_update"] = grp["datetime"].diff().dt.days.fillna(0).astype(float)
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)


def winsorize(series, lower=0.01, upper=0.99):
    """Clip extreme values at quantile boundaries before z-score."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 100:
        return series
    lo, hi = s.quantile(lower), s.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def rolling_zscore(series, window=1260, min_periods=252):
    s = pd.to_numeric(series, errors="coerce")
    # Winsorize before z-score to prevent extreme values from distorting normalization
    if len(s) >= 100:
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        s = s.clip(lower=lo, upper=hi)
    exp_mean = s.expanding(min_periods=min_periods).mean()
    exp_std = s.expanding(min_periods=min_periods).std()
    roll_mean = s.rolling(window=window, min_periods=min_periods).mean()
    roll_std = s.rolling(window=window, min_periods=min_periods).std()
    m = exp_mean.fillna(roll_mean)
    std = exp_std.fillna(roll_std)
    return (s - m) / std.replace(0, np.nan)


# ══════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════

# ── Core financial metrics whitelist ──────────────────────────────────
CORE_FINANCIAL = {
    "total_assets", "current_assets", "total_liabilities",
    "current_liabilities", "total_equity", "fixed_assets",
    "cash_equivalents", "inventory", "accounts_receivable",
    "notes_receivable", "long_term_debt", "short_term_borrowing",
    "paid_in_capital", "capital_reserve", "retained_earnings",
    "intangible_assets",
    "equity_to_assets", "debt_to_assets", "debt_to_equity",
    "current_ratio", "quick_ratio", "tangible_ratio",
    "roe", "roa", "gross_margin", "net_margin",
    "asset_turnover", "interest_coverage",
    "eps", "bvps", "cfo_to_assets", "cfo_to_debt",
    "net_profit", "operating_revenue", "operating_profit",
    "revenue", "operating_cost", "selling_expense",
    "total_profit", "income_tax",
}

# Keywords that appear in garbled Chinese financial column names
_FIN_KEYWORDS = [
    "total_assets", "current_assets", "total_liab", "current_liab",
    "total_equity", "fixed_assets", "cash", "inventor", "receiv",
    "long_term", "short_term", "debt", "borrow",
    "capital", "reserve", "retained", "intangible",
    "equity_to_assets", "debt_to_assets", "debt_to_equity",
    "current_ratio", "quick_ratio", "tangible_ratio",
    "roe", "roa", "gross_margin", "net_margin",
    "asset_turnover", "interest_coverage",
    "eps", "bvps", "cfo_to_assets", "cfo_to_debt",
    "net_profit", "NetProf", "NetInc", "revenue", "OpRev",
    "operating_profit", "OpProf", "operating_cost", "OpCost",
    "selling_expense", "total_profit", "TotProf", "income_tax",
    "TotAss", "CurAss", "TotLia", "CurLia", "TotEqu",
    "FixAss", "MonCap", "Inven", "AcctR", "LTBor", "STBor",
    "Undis", "CapRes",
]
# Also match garbled Chinese patterns
_FIN_CHINESE_KW = [
    "资产", "负债", "权益", "利润", "收入", "成本", "现金",
    "应收", "存货", "固定", "无形", "流动", "长期", "短期",
    "每股", "净资", "总资", "净利",
]

def _filter_financial(df):
    """Keep only core financial metrics (keyword match on garbled names).
    Uses pyarrow compute for memory-efficient operation on large DataFrames."""
    import pyarrow.compute as pc
    import pyarrow as pa

    # Convert to pyarrow (avoids pandas ArrowDtype fragmentation)
    variables = pa.array(df["variable"].astype(str).values)
    n = len(variables)

    # Split qualified vs unqualified
    has_sep = pc.match_substring(variables, "::")

    # For qualified vars, extract the metric part after "::"
    # pc.split_pattern returns a list-array; we extract element [1]
    split_result = pc.split_pattern(variables, "::")
    # Get the metric part (index 1) for rows with "::", else empty string
    metrics = pc.if_else(has_sep, pc.list_element(split_result, 1), pa.nulls(n, pa.string()))

    # Build boolean masks for each keyword set
    qual_keep = pc.fill_null(pc.match_substring(pc.utf8_lower(metrics),
                              _FIN_KEYWORDS[0].lower()), False)
    for kw in _FIN_KEYWORDS[1:]:
        qual_keep = pc.or_(qual_keep, pc.fill_null(
            pc.match_substring(pc.utf8_lower(metrics), kw.lower()), False))
    for kw in _FIN_CHINESE_KW:
        qual_keep = pc.or_(qual_keep, pc.fill_null(
            pc.match_substring(metrics, kw), False))

    # Unqualified: drop if they match Chinese financial keywords (garbled)
    unqual_drop = pc.fill_null(pc.match_substring(variables, _FIN_CHINESE_KW[0]), False)
    for kw in _FIN_CHINESE_KW[1:]:
        unqual_drop = pc.or_(unqual_drop, pc.fill_null(
            pc.match_substring(variables, kw), False))
    unqual_keep = pc.and_(pc.invert(has_sep), pc.invert(unqual_drop))

    keep = pc.or_(pc.and_(has_sep, qual_keep), unqual_keep)
    return df[keep.to_pylist()]


def prepare_data(input_dirs=None, forecast_horizon=63, rf_annual=0.025):
    if input_dirs is None:
        data_root = Path("D:/financial_data")
        input_dirs = [data_root,
                      Path("data/raw"),
                      Path("C:/Users/lcdell/Desktop/上市企业数据")]

    # 1. Collect all files
    all_files = []
    for d in input_dirs:
        if not d.exists(): continue
        all_files.extend(
            f for f in d.rglob("*")
            if f.suffix.lower() in (".csv",".parquet",".xlsx",".xls")
            and f.is_file() and not f.name.startswith("~$")
            and "processed" not in str(f).replace("\\","/").split("/")  # skip processed dir
        )
    # Add derived ratios
    ratios_path = Path("data/processed/financial_ratios.parquet")
    if ratios_path.exists():
        all_files.append(ratios_path)

    # Streamed balance sheet handled separately (pre-z-scored) — NOT added to all_files

    print(f"Found {len(all_files)} files")

    # 2. Load each file → long format → save to disk individually
    temp_dir = DATA_PROCESSED / "temp"
    temp_dir.mkdir(exist_ok=True)

    loaded = 0
    for f in sorted(all_files):
        # Skip files too large for 8GB RAM (>200MB CSV)
        if f.stat().st_size > 200e6 and f.suffix == ".csv":
            print(f"  SKIP (too large: {f.stat().st_size/1e6:.0f}MB) {f.parent.name}/{f.name}")
            continue

        out = temp_dir / f"{f.stem}_{hash(str(f))}.parquet"
        if out.exists():
            loaded += 1
            continue
        try:
            df = load_one_file(f)
            if df is not None and len(df) > 0:
                df.to_parquet(out, index=False)
                loaded += 1
        except Exception as e:
            if loaded < 15:
                print(f"  SKIP {f.parent.name}/{f.name}: {e}")

    print(f"Loaded {loaded} files")

    # 3. Process each source independently (memory-efficient)
    temp_files = list(temp_dir.glob("*.parquet"))
    source_dfs = {}
    for tf in temp_files:
        df = pd.read_parquet(tf)
        for src in df["source"].unique():
            sdf = df[df["source"] == src]
            if src not in source_dfs:
                source_dfs[src] = []
            source_dfs[src].append(sdf)

    # Pre-load streamed balance sheet (already z-scored, just merge)
    pre_zscored = []
    bs_chunks_dir = Path("D:/financial_data/processed/balance_sheet_chunks")
    if bs_chunks_dir.exists():
        bs_dfs = []
        for f in bs_chunks_dir.glob("*.parquet"):
            df = pd.read_parquet(f)
            # Apply financial whitelist filter
            df = _filter_financial(df)
            df["source"] = "financial"
            bs_dfs.append(df)
        if bs_dfs:
            bs_merged = pd.concat(bs_dfs, ignore_index=True)
            bs_merged["value"] = pd.to_numeric(bs_merged["value"], errors="coerce")
            bs_merged["value_raw"] = pd.to_numeric(bs_merged["value_raw"], errors="coerce")
            bs_merged["time_since_update"] = pd.to_numeric(bs_merged["time_since_update"], errors="coerce").fillna(0)
            bs_merged = bs_merged.dropna(subset=["value"])
            pre_zscored.append(bs_merged)
            print(f"  [BALANCE SHEET] {len(bs_merged):,} pre-z-scored rows (filtered)")

    # ── Filter financial source_dfs to core metrics ──
    if "financial" in source_dfs:
        before = sum(len(d) for d in source_dfs["financial"])
        source_dfs["financial"] = [
            _filter_financial(d) for d in source_dfs["financial"]
        ]
        after = sum(len(d) for d in source_dfs["financial"])
        print(f"  [FINANCIAL FILTER] {before:,} → {after:,} rows (kept {len(CORE_FINANCIAL)} core metrics)")

    all_zscored = pre_zscored.copy()
    for src, dfs in source_dfs.items():
        total_rows = sum(len(d) for d in dfs)
        if total_rows == 0:
            print(f"  Processing {src}: 0 rows (filtered out, skipping)")
            continue
        print(f"  Processing {src}: {total_rows:,} rows")
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values(["variable","datetime"])

        # Time since update
        combined = compute_tsu(combined)

        # Z-score per variable (batch by 50 vars to limit memory)
        vars_list = combined["variable"].unique()
        print(f"    {len(vars_list)} variables, z-scoring in batches of 50...")
        parts = []
        for batch_start in range(0, len(vars_list), 50):
            batch_vars = vars_list[batch_start:batch_start+50]
            batch = combined[combined["variable"].isin(batch_vars)]
            for vname in batch_vars:
                grp = batch[batch["variable"] == vname].sort_values("datetime")
                grp["value_raw"] = grp["value"]
                grp["value"] = rolling_zscore(grp["value"])
                parts.append(grp)
            if (batch_start // 50) % 5 == 0:
                print(f"      {min(batch_start+50, len(vars_list))}/{len(vars_list)} vars")

        combined = pd.concat(parts, ignore_index=True)
        combined = combined.dropna(subset=["value"])
        all_zscored.append(combined)
        print(f"    Done: {len(combined):,} rows after z-score")

    # Merge all sources
    print("Merging sources...")
    final = pd.concat(all_zscored, ignore_index=True).reset_index(drop=True)

    # 4. Build labels and volatility features from CSI 300 index
    print(f"Building labels and volatility features...")
    csi300 = final[
        (final["source"] == "market") &
        (final["variable"] == "close") &
        (~final["value_raw"].isna())
    ]
    if len(csi300) < 100:
        csi300 = final[
            (final["variable"].str.contains("000300", case=False)) &
            (~final["value_raw"].isna())
        ]
    csi300 = csi300[["datetime","value_raw"]].drop_duplicates().sort_values("datetime")
    csi300 = csi300.copy()
    csi300["value_raw"] = pd.to_numeric(csi300["value_raw"], errors="coerce")
    csi300 = csi300.dropna(subset=["value_raw"])

    # Daily log returns for volatility computation (from past only, no look-ahead)
    csi300["ret"] = np.log(csi300["value_raw"] / csi300["value_raw"].shift(1))
    csi300["ret"] = csi300["ret"].replace([np.inf, -np.inf], np.nan)

    # Realized volatility (annualized) — key market state features
    for w in [21, 63, 252]:
        csi300[f"vol_{w}d"] = csi300["ret"].rolling(w, min_periods=max(5, w//4)).std() * np.sqrt(252)

    # Build volatility feature rows for injection into data pipeline
    vol_rows = []
    for w in [21, 63, 252]:
        col = f"vol_{w}d"
        sub = csi300[["datetime", col]].dropna().copy()
        sub = sub.rename(columns={col: "value_raw"})
        sub["source"] = "macro"
        sub["variable"] = f"csi300_realized_vol_{w}d"
        sub["value"] = sub["value_raw"]  # raw=value for z-score later
        sub["time_since_update"] = 0.0
        vol_rows.append(sub[["datetime", "source", "variable", "value", "value_raw", "time_since_update"]])
    vol_df = pd.concat(vol_rows, ignore_index=True)
    # Expanding z-score per volatility variable (preserves temporal ordering)
    for v in vol_df["variable"].unique():
        mask = vol_df["variable"] == v
        vals = vol_df.loc[mask, "value_raw"].values
        em = pd.Series(vals).expanding(min_periods=63).mean().values
        es = pd.Series(vals).expanding(min_periods=63).std().values
        vol_df.loc[mask, "value"] = (vals - em) / np.where(es == 0, 1, es)
    vol_df = vol_df.dropna(subset=["value"])
    print(f"  Volatility features: {len(vol_df):,} rows (21d/63d/252d, expanding z-scored)")

    # Label: raw 63-day excess log-return (no vol normalization)
    # Winsorize at 1%/99% to limit extreme values
    csi300["fwd"] = csi300["value_raw"].shift(-forecast_horizon)
    ratio = csi300["fwd"] / csi300["value_raw"].replace(0, np.nan)
    csi300["label"] = np.log(ratio.astype(float))
    daily_rf = (1+rf_annual)**(1/252)-1
    csi300["label"] = csi300["label"] - daily_rf * forecast_horizon
    # Winsorize at 1st and 99th percentile
    lo, hi = csi300["label"].quantile(0.01), csi300["label"].quantile(0.99)
    csi300["label"] = csi300["label"].clip(lo, hi)
    labels = csi300.dropna(subset=["label"])[["datetime","label"]].copy()

    # 5. Merge volatility features into final, then save
    final = pd.concat([final, vol_df], ignore_index=True)
    final["value"] = pd.to_numeric(final["value"], errors="coerce")
    final["value_raw"] = pd.to_numeric(final["value_raw"], errors="coerce")
    final["time_since_update"] = pd.to_numeric(final["time_since_update"], errors="coerce")
    final = final.dropna(subset=["value"])

    dp_path = DATA_PROCESSED / "data_points.parquet"
    lb_path = DATA_PROCESSED / "labels.parquet"
    final.to_parquet(dp_path, index=False)
    labels.to_parquet(lb_path, index=False)

    print(f"\nSaved to {DATA_PROCESSED}/")
    print(f"  data_points.parquet: {len(final):,} rows")
    print(f"  labels.parquet:      {len(labels):,} rows")
    print(f"  Label: mean={labels['label'].mean():.4f} std={labels['label'].std():.4f}")
    for src in sorted(final["source"].unique()):
        s = final[final["source"]==src]
        print(f"  {src}: {len(s):,} rows, {s['variable'].nunique()} vars, {s['datetime'].nunique()} dates")

    # Cleanup temp
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=63)
    args = parser.parse_args()
    prepare_data(forecast_horizon=args.horizon)


if __name__ == "__main__":
    main()
