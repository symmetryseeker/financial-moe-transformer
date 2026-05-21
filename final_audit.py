"""
Complete end-to-end audit: data quality, architecture, training readiness.
"""
import sys; sys.path.insert(0, '.')
import time, json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import Counter

print("=" * 75)
print("  COMPREHENSIVE SYSTEM AUDIT — Financial MoE Transformer v2")
print("=" * 75)

# ═════════════════════════════════════════════════════════════════════════
# 1. DATA AUDIT
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("  1. DATA QUALITY AUDIT")
print("=" * 75)

dp = pd.read_parquet("data/processed/data_points.parquet")
labels = pd.read_parquet("data/processed/labels.parquet")

print(f"\n  Data points:      {len(dp):>12,} rows")
print(f"  Unique variables: {dp['variable'].nunique():>12,}")
print(f"  Unique dates:     {dp['datetime'].nunique():>12,}")
print(f"  Date range:       {dp['datetime'].min()} -> {dp['datetime'].max()}")
print(f"  Labels:           {len(labels):>12,} rows")

# Per-source breakdown
print(f"\n  --- Per-Source Breakdown ---")
print(f"  {'Source':<16s} {'Rows':>10s} {'Vars':>6s} {'Dates':>8s} {'Date Range':>30s}")
print(f"  {'-'*16} {'-'*10} {'-'*6} {'-'*8} {'-'*30}")
for src in sorted(dp["source"].unique()):
    sdf = dp[dp["source"] == src]
    dmin = str(sdf['datetime'].min())[:10]
    dmax = str(sdf['datetime'].max())[:10]
    print(f"  {src:<16s} {len(sdf):>10,} {sdf['variable'].nunique():>6} "
          f"{sdf['datetime'].nunique():>8} {dmin} -> {dmax}")

# Variable naming audit
print(f"\n  --- Variable Naming ---")
for src in sorted(dp["source"].unique()):
    sdf = dp[dp["source"] == src]
    vars_list = sorted(sdf["variable"].unique())
    # Show a few examples
    examples = vars_list[:5]
    has_qualifier = sum(1 for v in vars_list if "::" in v)
    print(f"  {src}: {len(vars_list)} vars, {has_qualifier} qualified (::)")
    for ex in examples:
        print(f"    eg: {ex}")

# Label quality
print(f"\n  --- Label Quality ---")
print(f"  Mean:     {labels['label'].mean():.4f}")
print(f"  Std:      {labels['label'].std():.4f}")
print(f"  Skew:     {labels['label'].skew():.2f}")
print(f"  Kurtosis: {labels['label'].kurtosis():.2f}")
print(f"  Min:      {labels['label'].min():.4f}")
print(f"  Max:      {labels['label'].max():.4f}")
print(f"  P1/P99:   {labels['label'].quantile(0.01):.4f} / {labels['label'].quantile(0.99):.4f}")
# Check for NaN
print(f"  NaN:      {labels['label'].isna().sum()}")
print(f"  Inf:      {np.isinf(labels['label']).sum()}")
# Label autocorrelation
acf1 = labels['label'].autocorr(lag=1)
acf5 = labels['label'].autocorr(lag=5) if len(labels) > 5 else np.nan
print(f"  ACF(1):   {acf1:.4f}")
print(f"  ACF(5):   {acf5:.4f}")

# Time-since-update quality
print(f"\n  --- Time-Since-Update Distribution ---")
tsu = dp["time_since_update"].dropna()
for pct in [50, 75, 90, 95, 99, 100]:
    print(f"  P{pct}: {np.percentile(tsu, pct):.0f} days")

# Value distribution after z-score
print(f"\n  --- Value Distribution (post z-score) ---")
vals = dp["value"].dropna()
print(f"  Mean: {vals.mean():.4f}")
print(f"  Std:  {vals.std():.4f} (should be ~1.0)")
print(f"  P1:   {vals.quantile(0.01):.3f}")
print(f"  P99:  {vals.quantile(0.99):.3f}")

# Data density by source
print(f"\n  --- Data Density (avg rows per trading day) ---")
for src in sorted(dp["source"].unique()):
    sdf = dp[dp["source"] == src]
    n_dates = sdf["datetime"].nunique()
    avg_rows = len(sdf) / n_dates if n_dates > 0 else 0
    print(f"  {src}: {avg_rows:.0f} rows/date")


# ═════════════════════════════════════════════════════════════════════════
# 2. DATASET & SEQUENCE AUDIT
# ═════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*75}")
print(f"  2. DATASET & SEQUENCE AUDIT")
print(f"{'='*75}")

from utils.dataset import SlidingWindowDataset, collate_fn
from config import model_cfg

ds = SlidingWindowDataset(
    data_path='data/processed/data_points.parquet',
    labels_path='data/processed/labels.parquet',
    window_days=365, forecast_horizon=21,
    max_seq_len=model_cfg.max_seq_len, use_cache=False
)

print(f"\n  Total windows:     {len(ds)}")
print(f"  Vocabulary size:   {len(ds.var_to_id)}")
print(f"  Max seq len:       {model_cfg.max_seq_len}")

# Sample sequence length distribution
seq_lens = []
valid_counts = []
for i in range(min(300, len(ds))):
    s = ds[i]
    if s is not None:
        seq_lens.append(s["seq_len"])
        valid_counts.append(s["mask"].sum().item())

seq_lens = np.array(seq_lens)
if len(seq_lens) > 0:
    print(f"\n  --- Sequence Lengths (n={len(seq_lens)}) ---")
    print(f"  Mean:   {seq_lens.mean():.0f}")
    print(f"  Min:    {seq_lens.min()}")
    print(f"  P50:    {np.percentile(seq_lens, 50):.0f}")
    print(f"  P90:    {np.percentile(seq_lens, 90):.0f}")
    print(f"  Max:    {seq_lens.max()}")
    trunc_pct = (seq_lens == model_cfg.max_seq_len).mean() * 100
    print(f"  Truncated: {trunc_pct:.0f}% at max ({model_cfg.max_seq_len})")

    # Token count per window
    total_tokens = seq_lens.sum()
    avg_valid = np.mean(valid_counts)
    print(f"  Avg valid tokens/window: {avg_valid:.0f}")
    print(f"  Total tokens (est):      {total_tokens:,.0f}")


# ═════════════════════════════════════════════════════════════════════════
# 3. MODEL ARCHITECTURE AUDIT
# ═════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*75}")
print(f"  3. MODEL ARCHITECTURE AUDIT")
print(f"{'='*75}")

from model import FinancialMoETransformer

model = FinancialMoETransformer(
    vocab_size=max(model_cfg.vocab_size, len(ds.var_to_id) + 2),
    d_model=model_cfg.d_model, nhead=model_cfg.nhead,
    num_layers=model_cfg.num_layers,
    dim_feedforward=model_cfg.dim_feedforward,
    attn_type=model_cfg.attn_type,
    num_experts=model_cfg.num_experts,
    expert_hidden=model_cfg.expert_hidden,
    top_k=model_cfg.top_k,
    max_seq_len=model_cfg.max_seq_len,
    attn_chunk_size=model_cfg.attn_chunk_size,
)

print(f"\n  --- Parameter Breakdown ---")
total = 0
details = []
for name, mod in model.named_children():
    n = sum(p.numel() for p in mod.parameters())
    total += n
    details.append((name, n))
    # Sub-modules
    for sub_name, sub_mod in mod.named_children():
        sn = sum(p.numel() for p in sub_mod.parameters())
        if sn > 1000:
            details.append((f"  └ {sub_name}", sn))

for name, n in details:
    print(f"  {name:<35s} {n:>10,}")

print(f"  {'─'*45}")
print(f"  {'TOTAL':<35s} {total:>10,}")
print(f"  GPU memory (FP32 params):        {total*4/1e6:.1f} MB")
print(f"  GPU memory (+grad+optim):        {total*16/1e6:.1f} MB")
print(f"  Under 2M limit:                  {'YES' if total < 2e6 else 'NO'}")

# Architecture summary
print(f"\n  --- Architecture Summary ---")
print(f"  Embedding:   Value(1→128) + Variable({model.embedding.var_embed.embed.weight.shape[0]},128)")
print(f"             + Calendar(day+month+dow+year→128) + TimeBucket(11 bins→128)")
print(f"  Encoder:     {model_cfg.num_layers} layers × MultiheadLinearAttention(d={model_cfg.d_model}, h={model_cfg.nhead})")
print(f"             + FFN({model_cfg.d_model}→{model_cfg.dim_feedforward}→{model_cfg.d_model})")
print(f"  Attention:   cosFormer (chunked, chunk={model_cfg.attn_chunk_size})")
print(f"  Pooling:     AttentionPooling (learnable query, {model_cfg.nhead} heads)")
print(f"  MoE:         {model_cfg.num_experts} experts ({model_cfg.d_model}→{model_cfg.expert_hidden}→32→1)")
print(f"             + Top-{model_cfg.top_k} sparse gating + load-balancing loss")


# ═════════════════════════════════════════════════════════════════════════
# 4. TRAINING READINESS
# ═════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*75}")
print(f"  4. TRAINING READINESS AUDIT")
print(f"{'='*75}")

from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW

# Speed test
loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
model.train()
optimizer = AdamW(model.parameters(), lr=3e-4)

# Warmup
print("\n  Warming up...")
for i, batch in enumerate(loader):
    if i >= 3: break
    pred, lb = model(batch['values'], batch['var_ids'], batch['day'],
                     batch['month'], batch['dow'], batch['year_offset'],
                     batch['time_since_update'], batch['mask'].bool())
    loss = torch.nn.functional.mse_loss(pred.squeeze(-1), batch['label']) + 0.01 * lb
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Benchmark
print("  Benchmarking 15 batches...")
times = []
model.train()
for i, batch in enumerate(loader):
    if i >= 15: break
    t0 = time.time()
    pred, lb = model(batch['values'], batch['var_ids'], batch['day'],
                     batch['month'], batch['dow'], batch['year_offset'],
                     batch['time_since_update'], batch['mask'].bool())
    loss = torch.nn.functional.mse_loss(pred.squeeze(-1), batch['label']) + 0.01 * lb
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    times.append(time.time() - t0)

avg_time = np.mean(times)
batches_per_epoch = len(ds) // 2
time_per_epoch = batches_per_epoch * avg_time

print(f"\n  --- Speed Estimate (CPU) ---")
print(f"  Per batch (2 samples):   {avg_time:.2f}s")
print(f"  Batches per epoch:       {batches_per_epoch}")
print(f"  Time per epoch:          {time_per_epoch/60:.1f} min")
print(f"  Time for 20 epochs:      {time_per_epoch*20/60:.1f} min ({time_per_epoch*20/3600:.1f} hr)")
print(f"  Early stop (~8 epochs):  {time_per_epoch*8/60:.1f} min")

# Split estimate
train_dates, val_dates, test_dates = ds.train_val_test_split()
print(f"\n  --- Data Split ---")
print(f"  Train: {len(train_dates)} windows ({len(train_dates)/len(ds)*100:.0f}%)")
print(f"  Val:   {len(val_dates)} windows ({len(val_dates)/len(ds)*100:.0f}%)")
print(f"  Test:  {len(test_dates)} windows ({len(test_dates)/len(ds)*100:.0f}%)")


# ═════════════════════════════════════════════════════════════════════════
# 5. GAPS & IMPROVEMENT SPACE
# ═════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*75}")
print(f"  5. GAPS & IMPROVEMENT ANALYSIS")
print(f"{'='*75}")

print(f"""
  DATA GAPS:
    [MISSING] 资产负债表 — 桌上已有ZIP(152MB), 未集成
    [MISSING] 现金流量表 — 桌上已有ZIP
    [MISSING] 新闻标题 → BGE编码 — 网络封锁, 需代理或离线采集
    [MISSING] 研报摘要 → BGE编码 — 同上
    [MISSING] LLM市场状态 — DeepSeek脚本已就绪, 待API key
    [THIN]    北向资金 — 仅294行, 日期解析需修复
    [THIN]    macro数据 — 11指标偏少, 缺少进出口/工业增加值/社消
    [GAP]     无港股/美股数据 — yfinance可补充

  ARCHITECTURE GAPS:
    [OK]      参数量 1.29M < 2M 上限
    [OK]      分块注意力量 已支持最长8K序列
    [GAP]     8K序列仍截断~98%数据(每窗口实际>50万tokens)
    [GAP]     无分层聚合 — 180只股票平铺, 无先验结构
    [GAP]     注意力无法可视化 — 线性注意力不产生L×L矩阵
    [GAP]     概念正则化未启用

  TRAINING GAPS:
    [OK]      MSE + 负载均衡损失
    [GAP]     无排序损失(Spearman/IC)
    [GAP]     无方向准确率损失
    [OK]      早停 patience=5
    [GAP]     无Walk-Forward交叉验证
    [GAP]     无集成/多模型对比
""")

print(f"\n{'='*75}")
print(f"  AUDIT COMPLETE")
print(f"{'='*75}")
