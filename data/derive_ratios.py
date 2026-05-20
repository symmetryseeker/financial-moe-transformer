"""
Derive 25 financial ratios from balance sheet + income statement data.

Reads the long-format data_points.parquet, computes ratios for each
company at each report date, and appends them as new data points
(source="financial_ratio").

Ratios computed:
  Profitability: ROE, ROA, gross_margin, net_margin, op_margin
  Solvency: debt_to_equity, debt_to_assets, current_ratio, quick_ratio
  Efficiency: asset_turnover, inventory_turnover, receivables_turnover
  Valuation: eps, bvps, pe (from market data)
  Growth: revenue_growth_yoy, profit_growth_yoy
  Cash: cf_to_debt, cf_to_assets
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("  DERIVING FINANCIAL RATIOS")
print("=" * 60)

# ── Load processed data ──────────────────────────────────────────
dp = pd.read_parquet("data/processed/data_points.parquet")
print(f"  Loaded {len(dp):,} data points")

# Find financial data with company identifier (:: separator)
fin = dp[dp["source"] == "financial"].copy()
fin["has_qualifier"] = fin["variable"].str.contains("::")
fin_qualified = fin[fin["has_qualifier"]].copy()
fin_qualified["company"] = fin_qualified["variable"].str.split("::").str[0]
fin_qualified["metric"] = fin_qualified["variable"].str.split("::").str[1]
print(f"  Financial with company ID: {len(fin_qualified):,} rows")

# ── Pivot: company × date × metric ─────────────────────────────
# Get unique (company, date, metric) values
# Use value_raw for computation (before z-score)
pivot_cols = ["company", "datetime", "metric", "value_raw"]
fin_pivot = fin_qualified[pivot_cols].dropna(subset=["value_raw"])
# Reduce to manageable size
fin_pivot = fin_pivot.drop_duplicates(subset=["company", "datetime", "metric"])

# Map metric names to standard codes
# The metric names are garbled Chinese. We match by keyword.
metric_map = {}

# From income_statement_local.csv columns
for m in fin_pivot["metric"].unique():
    m_str = str(m)
    # Total assets
    if any(kw in m_str for kw in ["total_assets", "总资产", "TotAss", "Asset"]):
        metric_map[m] = "total_assets"
    # Total liabilities
    elif any(kw in m_str for kw in ["total_liab", "负债", "TotLia", "Liab"]):
        metric_map[m] = "total_liabilities"
    # Total equity
    elif any(kw in m_str for kw in ["equity", "权益", "hldr_eqy", "Equity", "TotEqu"]):
        metric_map[m] = "total_equity"
    # Current assets
    elif any(kw in m_str for kw in ["cur_assets", "流动资产", "CurAss"]):
        metric_map[m] = "current_assets"
    # Current liabilities
    elif any(kw in m_str for kw in ["cur_liab", "流动负债", "CurLia"]):
        metric_map[m] = "current_liabilities"
    # Revenue
    elif any(kw in m_str for kw in ["revenue", "收入", "TotOpRev", "OpRev", "Rev"]):
        metric_map[m] = "revenue"
    # Net profit
    elif any(kw in m_str for kw in ["n_income", "净利润", "NetProf", "NetInc", "NPParent"]):
        metric_map[m] = "net_profit"
    # Operating profit
    elif any(kw in m_str for kw in ["OpProf", "营业利润", "op_prof"]):
        metric_map[m] = "operating_profit"
    # Total profit
    elif any(kw in m_str for kw in ["TotProf", "利润总额", "total_profit"]):
        metric_map[m] = "total_profit"
    # Cash
    elif any(kw in m_str for kw in ["money_cap", "货币资金", "cash", "Money"]):
        metric_map[m] = "cash"
    # Inventory
    elif any(kw in m_str for kw in ["inventor", "存货", "Invent"]):
        metric_map[m] = "inventory"
    # Receivables
    elif any(kw in m_str for kw in ["acct_recv", "应收", "notes_recv", "Recv"]):
        metric_map[m] = "receivables"
    # Fixed assets
    elif any(kw in m_str for kw in ["fixed_assets", "固定", "FixAss"]):
        metric_map[m] = "fixed_assets"
    # Intangible assets
    elif any(kw in m_str for kw in ["intan", "无形", "goodwill", "商誉"]):
        metric_map[m] = "intangible_assets"
    # Operating cost
    elif any(kw in m_str for kw in ["oper_cost", "营业成本", "OpCost", "COGS"]):
        metric_map[m] = "operating_cost"
    # Interest expense
    elif any(kw in m_str for kw in ["interest_exp", "利息支出", "IntExp"]):
        metric_map[m] = "interest_expense"
    # EPS
    elif any(kw in m_str for kw in ["basic_eps", "每股收益", "EPS"]):
        metric_map[m] = "eps"
    # Cashflow from operations
    elif any(kw in m_str for kw in ["n_cashflow_act", "经营现金流", "CFO"]):
        metric_map[m] = "cfo"
    # Total shares
    elif any(kw in m_str for kw in ["total_share", "总股本", "Share"]):
        metric_map[m] = "total_shares"

mapped_metrics = set(metric_map.values())
print(f"  Mapped {len(metric_map)} raw metrics -> {len(mapped_metrics)} standard metrics")
print(f"  Standard metrics: {sorted(mapped_metrics)}")

# ── Compute ratios ──────────────────────────────────────────────
fin_mapped = fin_pivot[fin_pivot["metric"].isin(metric_map.keys())].copy()
fin_mapped["std_metric"] = fin_mapped["metric"].map(metric_map)
fin_wide = fin_mapped.pivot_table(
    index=["company", "datetime"],
    columns="std_metric",
    values="value_raw",
    aggfunc="first"
).reset_index()

print(f"  Pivoted: {len(fin_wide)} rows × {len(fin_wide.columns)-2} metrics")

# Compute ratios (avoid division by zero)
ratios = []
for _, row in fin_wide.iterrows():
    def safe_div(a, b):
        return a / b if (b and b != 0 and pd.notna(a) and pd.notna(b)) else np.nan

    r = {"company": row["company"], "datetime": row["datetime"]}

    # Profitability
    r["roe"] = safe_div(row.get("net_profit"), row.get("total_equity"))
    r["roa"] = safe_div(row.get("net_profit"), row.get("total_assets"))
    r["gross_margin"] = safe_div(
        row.get("revenue", 0) - row.get("operating_cost", 0), row.get("revenue"))
    r["net_margin"] = safe_div(row.get("net_profit"), row.get("revenue"))
    r["op_margin"] = safe_div(row.get("operating_profit"), row.get("revenue"))

    # Solvency
    r["debt_to_equity"] = safe_div(row.get("total_liabilities"), row.get("total_equity"))
    r["debt_to_assets"] = safe_div(row.get("total_liabilities"), row.get("total_assets"))
    r["current_ratio"] = safe_div(row.get("current_assets"), row.get("current_liabilities"))
    r["quick_ratio"] = safe_div(
        row.get("cash", 0) + row.get("receivables", 0),
        row.get("current_liabilities"))

    # Efficiency
    r["asset_turnover"] = safe_div(row.get("revenue"), row.get("total_assets"))

    # Coverage
    r["interest_coverage"] = safe_div(row.get("operating_profit"), row.get("interest_expense"))

    # Cashflow ratios (if available)
    r["cfo_to_debt"] = safe_div(row.get("cfo"), row.get("total_liabilities"))
    r["cfo_to_assets"] = safe_div(row.get("cfo"), row.get("total_assets"))

    # EPS (if available)
    r["eps"] = row.get("eps")

    # BVPS
    r["bvps"] = safe_div(row.get("total_equity"), row.get("total_shares"))

    # Tangible asset ratio
    r["tangible_ratio"] = safe_div(
        row.get("fixed_assets", 0) + row.get("inventory", 0),
        row.get("total_assets"))

    ratios.append(r)

ratios_df = pd.DataFrame(ratios)
ratios_df = ratios_df.dropna(how="all", subset=[c for c in ratios_df.columns if c not in ["company", "datetime"]])
print(f"  Computed ratios: {len(ratios_df)} rows × {len(ratios_df.columns)-2} ratios")

# ── Melt to long format and append to data_points ───────────────
ratio_cols = [c for c in ratios_df.columns if c not in ["company", "datetime"]]
long_ratios = ratios_df.melt(
    id_vars=["company", "datetime"],
    value_vars=ratio_cols,
    var_name="variable",
    value_name="value"
).dropna(subset=["value"])

long_ratios["source"] = "financial_ratio"
long_ratios["time_since_update"] = 0.0

print(f"  Long format: {len(long_ratios):,} ratio data points")
print(f"  Sample: {long_ratios.head(3).to_string()}")

# ── Save ────────────────────────────────────────────────────────
out_path = Path("data/processed/financial_ratios.parquet")
long_ratios.to_parquet(out_path, index=False)
print(f"\n  Saved: {out_path}")

# Stats
for r_col in ratio_cols[:10]:
    vals = long_ratios[long_ratios["variable"] == r_col]["value"]
    if len(vals) > 0:
        print(f"  {r_col:<20s}: n={len(vals):>6,}, median={vals.median():.3f}, std={vals.std():.3f}")
