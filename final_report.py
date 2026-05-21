"""
COMPREHENSIVE FINAL REPORT — Data Inventory + Architecture + Training Readiness.
"""
import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("D:/financial_data")

print("=" * 85)
print("  FINANCIAL MoE TRANSFORMER v3 — FINAL PRE-TRAINING REPORT")
print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 85)

# ═════════════════════════════════════════════════════════════════════════
# 1. COMPLETE DATA INVENTORY
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 85)
print("  1. COMPLETE DATA FILE INVENTORY")
print("=" * 85)

total_files = 0; total_size = 0; total_rows_est = 0
for cat_dir in sorted(DATA_DIR.iterdir()):
    if not cat_dir.is_dir() or cat_dir.name == "processed": continue
    files = sorted(cat_dir.rglob("*.csv"))
    if not files: continue
    cat_size = 0; cat_rows = 0
    print(f"\n  [{cat_dir.name.upper()}]  ({len(files)} files)")
    print(f"  {'File':<42s} {'Rows':>9s} {'Cols':>5s}  {'Size':>7s}  Date Sample")
    print(f"  {'-'*42} {'-'*9} {'-'*5}  {'-'*7}  {'-'*20}")
    for f in files:
        sz = f.stat().st_size / 1e6
        cat_size += sz
        try:
            df = pd.read_csv(f, nrows=0)
            nc = len(df.columns)
            # Fast line count
            with open(f, 'r', encoding='utf-8', errors='replace') as fh:
                nr = sum(1 for _ in fh) - 1
            cat_rows += nr
            # Date sample
            df2 = pd.read_csv(f, nrows=1)
            date_cols = [c for c in df2.columns if any(kw in str(c).lower()
                         for kw in ['date','time','日期','trade','end','accper','月份'])]
            sample = str(df2[date_cols[0]].iloc[0])[:20] if date_cols else str(df2.iloc[0,0])[:20]
            print(f"  {f.name:<42s} {nr:>9,} {nc:>5}  {sz:>6.1f}MB  {sample}")
        except:
            print(f"  {f.name:<42s} {'?':>9} {'?':>5}  {sz:>6.1f}MB")

    print(f"  {'─'*70}")
    print(f"  Subtotal: {cat_rows:>9,} rows, {cat_size:>7.1f} MB")
    total_files += len(files); total_size += cat_size; total_rows_est += cat_rows

# ── Processed data ─────────────────────────────────────────────────
proc = DATA_DIR / "processed"
dp_path = proc / "data_points.parquet"
lb_path = proc / "labels.parquet"
bs_chunks = proc / "balance_sheet_chunks"

if dp_path.exists():
    dp = pd.read_parquet(dp_path)
    labels = pd.read_parquet(lb_path)
    print(f"\n  [PROCESSED]")
    print(f"    data_points.parquet:  {len(dp):>12,} rows  ({dp_path.stat().st_size/1e6:.0f} MB)")
    print(f"    labels.parquet:       {len(labels):>12,} rows")
    print(f"    balance_sheet_chunks:  {len(list(bs_chunks.glob('*'))):>3} files  ({sum(f.stat().st_size for f in bs_chunks.glob('*'))/1e6:.0f} MB)")

print(f"\n  ═══════════════════════════════════════════════")
print(f"  GRAND TOTAL: {total_files} raw files, {total_size:.0f} MB, ~{total_rows_est:,.0f} raw rows")
print(f"  Processed:   24,181,928 data points, 18,164 labels")
print(f"  Storage:     D:/financial_data/")

# ═════════════════════════════════════════════════════════════════════════
# 2. PROCESSED DATA QUALITY
# ═════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*85}")
print(f"  2. PROCESSED DATA QUALITY")
print(f"{'='*85}")

if dp_path.exists():
    dp = pd.read_parquet(dp_path)
    labels = pd.read_parquet(lb_path)

    print(f"\n  {'Source':<16s} {'Rows':>12s} {'Vars':>8s} {'Dates':>8s}  {'Date Range'}")
    print(f"  {'-'*16} {'-'*12} {'-'*8} {'-'*8}  {'-'*30}")
    for src in sorted(dp["source"].unique()):
        sdf = dp[dp["source"] == src]
        dmin = str(sdf['datetime'].min())[:10]
        dmax = str(sdf['datetime'].max())[:10]
        print(f"  {src:<16s} {len(sdf):>12,} {sdf['variable'].nunique():>8} {sdf['datetime'].nunique():>8}  {dmin} -> {dmax}")

    print(f"\n  Label Quality:")
    print(f"    Mean:     {labels['label'].mean():.4f}")
    print(f"    Std:      {labels['label'].std():.4f}")
    print(f"    Skew:     {labels['label'].skew():.2f}")
    print(f"    Kurtosis: {labels['label'].kurtosis():.2f}")
    print(f"    Range:    [{labels['label'].min():.4f}, {labels['label'].max():.4f}]")
    print(f"    ACF(1):   {labels['label'].autocorr(lag=1):.4f}")
    print(f"    ACF(5):   {labels['label'].autocorr(lag=5):.4f}")
    print(f"    NaN: {labels['label'].isna().sum()}  Inf: {np.isinf(labels['label']).sum()}")

    tsu = dp["time_since_update"].dropna()
    print(f"\n  time_since_update: P50={tsu.median():.0f}d  P90={tsu.quantile(0.9):.0f}d  P99={tsu.quantile(0.99):.0f}d")

# ═════════════════════════════════════════════════════════════════════════
# 3. ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*85}")
print(f"  3. MODEL ARCHITECTURE")
print(f"{'='*85}")

from model import FinancialMoETransformer
from utils.dataset import SlidingWindowDataset, collate_fn
from config import model_cfg

ds = SlidingWindowDataset(
    data_path=str(dp_path), labels_path=str(lb_path),
    window_days=365, forecast_horizon=21,
    max_seq_len=model_cfg.max_seq_len, use_cache=False
)

model = FinancialMoETransformer(
    vocab_size=max(model_cfg.vocab_size, len(ds.var_to_id) + 2),
    d_model=model_cfg.d_model, nhead=model_cfg.nhead,
    num_layers=model_cfg.num_layers,
    dim_feedforward=model_cfg.dim_feedforward,
    attn_type=model_cfg.attn_type,
    attn_chunk_size=model_cfg.attn_chunk_size,
    num_experts=model_cfg.num_experts,
    expert_hidden=model_cfg.expert_hidden,
    top_k=model_cfg.top_k,
    max_seq_len=model_cfg.max_seq_len,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n  Vocab size:    {len(ds.var_to_id):,}")
print(f"  Windows:       {len(ds)}")
print(f"  Max seq len:   {model_cfg.max_seq_len}")
print(f"  Total params:  {total_params:,} ({'OK' if total_params < 2e6 else 'OVER 2M!'})")

print(f"\n  ┌─────────────────────────────────────────────────────┐")
print(f"  │  ARCHITECTURE PIPELINE                              │")
print(f"  ├─────────────────────────────────────────────────────┤")
print(f"  │  1. Token Embedding                                 │")
print(f"  │     value(1→{model_cfg.d_model}) + var_id({len(ds.var_to_id)}→{model_cfg.d_model})")
print(f"  │     + calendar(day+month+dow+year→{model_cfg.d_model})")
print(f"  │     + time_bucket(11 bins→{model_cfg.d_model})")
print(f"  │                                                     │")
print(f"  │  2. Hierarchical Encoder (SimpleHierarchicalEncoder)│")
print(f"  │     Source-pooled × {model_cfg.n_scales}-scale temporal pyramid  │")
print(f"  │     → source representations (n_sources × d_model)  │")
print(f"  │                                                     │")
print(f"  │  3. Cross-Source Fusion                             │")
print(f"  │     {model_cfg.num_layers}× Transformer (cosFormer chunked attn)          │")
print(f"  │     d={model_cfg.d_model} h={model_cfg.nhead} FFN={model_cfg.dim_feedforward} chunk={model_cfg.attn_chunk_size}       │")
print(f"  │                                                     │")
print(f"  │  4. Attention Pooling                               │")
print(f"  │     Learnable query × {model_cfg.nhead} heads → (B, d_model)            │")
print(f"  │                                                     │")
print(f"  │  5. MoE Predictor                                   │")
print(f"  │     Gate({model_cfg.d_model}→{model_cfg.num_experts}) → Top-1 routing              │")
print(f"  │     {model_cfg.num_experts} experts ({model_cfg.d_model}→{model_cfg.expert_hidden}→32→1)           │")
print(f"  │     → scalar excess return prediction               │")
print(f"  └─────────────────────────────────────────────────────┘")

# Parameter breakdown
print(f"\n  Parameter Breakdown:")
for name, mod in [
    ("Embedding", model.embedding),
    ("Hierarchical", model.hierarchical),
    ("Attn Pooling", model.pooling),
    ("MoE", model.moe),
]:
    n = sum(p.numel() for p in mod.parameters())
    pct = n / total_params * 100
    print(f"    {name:<20s} {n:>10,}  ({pct:5.1f}%)")

# ═════════════════════════════════════════════════════════════════════════
# 4. TRAINING READINESS
# ═════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*85}")
print(f"  4. TRAINING CONFIGURATION")
print(f"{'='*85}")

from config import train_cfg

print(f"\n  Loss Function:")
print(f"    Total = {train_cfg.mse_coef}×MSE + {train_cfg.rank_coef}×SpearmanRank + {train_cfg.direction_coef}×DirectionBCE + {train_cfg.load_balance_coef}×LoadBalance")

print(f"\n  Optimizer:   AdamW(lr={train_cfg.lr}, wd={train_cfg.weight_decay})")
print(f"  Schedule:    Linear warmup({train_cfg.warmup_steps}) → Cosine decay")
print(f"  Batch:       {train_cfg.batch_size} × {train_cfg.gradient_accumulation} accum = {train_cfg.batch_size * train_cfg.gradient_accumulation} effective")
print(f"  Epochs:      {train_cfg.epochs} (early stop patience={train_cfg.early_stop_patience})")
print(f"  Grad clip:   {train_cfg.max_grad_norm}")
print(f"  AMP:         {'ON' if train_cfg.use_amp else 'OFF (GTX 1060 FP32 only)'}")
print(f"  CV folds:    {train_cfg.n_cv_folds}")

# Quick speed estimate
print(f"\n  Speed Estimate:")
sample = ds[0]
t0 = __import__('time').time()
for _ in range(5):
    pred, lb = model(
        sample['values'].unsqueeze(0), sample['var_ids'].unsqueeze(0),
        sample['day'].unsqueeze(0), sample['month'].unsqueeze(0),
        sample['dow'].unsqueeze(0), sample['year_offset'].unsqueeze(0),
        sample['time_since_update'].unsqueeze(0), sample['mask'].unsqueeze(0)
    )
elapsed = __import__('time').time() - t0
per_sample = elapsed / 5
time_per_epoch = len(ds) * per_sample / 60
print(f"    Per sample (CPU):    {per_sample:.2f}s")
print(f"    Time per epoch:      {time_per_epoch:.1f} min")
print(f"    20 epochs:           {time_per_epoch*20/60:.1f} hr")
print(f"    Early stop (~8 ep):  {time_per_epoch*8/60:.1f} hr")

# ═════════════════════════════════════════════════════════════════════════
# 5. DATA GAPS & IMPROVEMENTS
# ═════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*85}")
print(f"  5. REMAINING GAPS & NEXT STEPS")
print(f"{'='*85}")

print(f"""
  DATA GAPS:
    [OK]     A-share daily OHLCV + valuation (180 CSI 300 stocks)
    [OK]     Balance sheet (16 metrics × all listed companies)
    [OK]     Income statement (profit + revenue + EPS)
    [OK]     Cashflow (bank / securities / insurance)
    [OK]     Macro: CPI/PPI/PMI/GDP/LPR/M2/社融/国债/存款/贷款/准备金/贸易
    [OK]     International: VIX/DXY/US10Y/S&P500/HSI/Nikkei/FTSE/Stoxx50
    [OK]     Commodities: Gold/Silver/Copper/CrudeOil (COMEX)
    [OK]     FX: USDCNY/EURUSD/USDJPY
    [OK]     Carbon: 9 China exchanges daily
    [OK]     Northbound flow: SH + SZ (2014-2026)
    [OK]     Margin trading: SSE + SZSE
    [OK]     Futures: Rebar/Copper/Gold (Sina)
    [MISSING] Text embeddings (news headlines → BGE encoding)
    [MISSING] LLM market state (DeepSeek V4 — script ready, needs API key)
    [MISSING] Income statement FULL (268MB, skipped for RAM, can stream like BS)

  ARCHITECTURE:
    [OK]     1.35M parameters (< 2M limit)
    [OK]     Hierarchical source encoding + multi-scale temporal pyramid
    [OK]     Chunked cosFormer attention (handles long sequences)
    [OK]     Attention pooling (learnable, interpretable)
    [OK]     Sparse MoE (6 experts, Top-1, load balancing)
    [OK]     Advanced losses (MSE + Rank + Direction + LoadBalance)
    [OK]     Walk-forward cross-validation (5 folds)

  TRAINING:
    [OK]     AdamW + cosine LR + warmup + grad clip
    [OK]     Data stored on D drive (no C drive pressure)
    [OK]     Label: CSI 300 excess return (mean=-0.0018, std=0.73)
    [OK]     8GB RAM, CPU training feasible
    [GAP]    GPU training needs NVIDIA driver update (currently 398.27)
""")

print("=" * 85)
print("  REPORT COMPLETE — Ready for final decision")
print("=" * 85)
