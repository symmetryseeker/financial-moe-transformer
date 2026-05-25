"""
5-Fold Walk-Forward CV for Index Market Timing Strategy.
Each fold: train from scratch -> validate -> test on next period.
Signal: pred < 0 -> long CSI300 (inverted, proven effective).
"""
import sys; sys.path.insert(0,'.')
import time, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from utils.dataset import SlidingWindowDataset, collate_fn
from model import FinancialMoETransformer
from config import model_cfg, train_cfg, MODEL_DIR
from train import train_epoch, validate, EMA
from utils.losses import CombinedLoss
from scipy.stats import pearsonr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = "cuda"; RF = 0.025
N_FOLDS = 5

# ── Load data ──
ds = SlidingWindowDataset(
    data_path="data/processed/data_points.parquet",
    labels_path="data/processed/labels.parquet",
    max_seq_len=8192, use_cache=True, cache_dir="data/processed/cache_wf",
    multi_window=False)
all_dates = pd.DatetimeIndex(ds.window_dates)
print(f"Dataset: {len(ds)} windows, {all_dates[0].date()} -> {all_dates[-1].date()}")

# ── CSI 300 returns ──
csi = pd.read_csv("D:/financial_data/market/csi300_daily.csv")
csi["date"] = pd.to_datetime(csi["date"]); csi = csi.sort_values("date").set_index("date")
csi_ret = csi["close"].pct_change().dropna()

# ── WF-CV ──
results = []
all_equity_curves = {}

for fold in range(N_FOLDS):
    # Determine train/val/test date ranges
    test_years = [2020, 2021, 2022, 2023, 2024][fold]
    test_start = pd.Timestamp(f"{test_years}-01-01")
    test_end = pd.Timestamp(f"{test_years+1}-01-01") if fold < 4 else all_dates[-1]

    train_end = test_start - pd.Timedelta(days=1)
    train_start = all_dates[0]
    val_end = train_end
    val_start = train_end - pd.Timedelta(days=int((train_end - train_start).days * 0.15))

    train_idx = [i for i, d in enumerate(ds.window_dates) if d <= train_end]
    val_idx = [i for i, d in enumerate(ds.window_dates) if val_start <= d <= val_end]
    test_idx = [i for i, d in enumerate(ds.window_dates) if test_start <= d <= test_end]

    # Ensure val is 15% of train
    if len(val_idx) == 0:
        split = int(len(train_idx) * 0.85)
        val_idx = train_idx[split:]
        train_idx = train_idx[:split]

    print(f"\n{'='*55}")
    print(f"Fold {fold+1}/{N_FOLDS}: Test={test_years}")
    print(f"  Train: {pd.Timestamp(ds.window_dates[train_idx[0]]).date()} -> {pd.Timestamp(ds.window_dates[train_idx[-1]]).date()} ({len(train_idx)} windows)")
    print(f"  Val:   {pd.Timestamp(ds.window_dates[val_idx[0]]).date()} -> {pd.Timestamp(ds.window_dates[val_idx[-1]]).date()} ({len(val_idx)} windows)")
    print(f"  Test:  {pd.Timestamp(ds.window_dates[test_idx[0]]).date()} -> {pd.Timestamp(ds.window_dates[test_idx[-1]]).date()} ({len(test_idx)} windows)")
    print(f"{'='*55}")

    # ── Train ──
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=4,
        shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=4,
        shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

    model = FinancialMoETransformer(
        n_companies=ds.n_companies+2, n_metrics=ds.n_metrics+2,
        d_model=128, nhead=4, num_layers=4, dim_feedforward=512,
        dropout=0.2, attn_type="cosformer", attn_chunk_size=2048,
        num_experts=6, expert_hidden=64, top_k=1).to(DEVICE)

    opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    sp = max(1, len(train_loader)); total_steps = sp * 15
    ws = min(500, max(1, total_steps // 5))
    warmup = LinearLR(opt, start_factor=1e-3, total_iters=ws)
    cosine = CosineAnnealingLR(opt, T_max=total_steps - ws, eta_min=3e-6)
    scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[ws])
    loss_fn = CombinedLoss(mse_coef=1.0, load_balance_coef=0.01, rank_coef=0.3, direction_coef=0.0)
    ema = EMA(model, decay=0.999)
    best_ic = -float("inf"); patience = 0
    ckpt_path = MODEL_DIR / f"wf_fold{fold+1}.pt"

    for epoch in range(15):
        t0 = time.time()
        tl = train_epoch(model, train_loader, opt, scheduler, DEVICE, train_cfg, loss_fn)
        ema.update(); ema.apply_shadow()
        vm = validate(model, val_loader, DEVICE)
        ema.restore()
        print(f"  Ep {epoch+1:2d} ({time.time()-t0:.0f}s) MSE={tl.get('mse',0):.3f} "
              f"Rank={tl.get('rank',0):.3f} Val IC={vm['ic']:.4f}", flush=True)
        if vm["ic"] > best_ic:
            best_ic = vm["ic"]; patience = 0
            ema.apply_shadow()
            torch.save({"epoch": epoch, "model": model.state_dict(), "best_ic": best_ic, "fold": fold}, ckpt_path)
            ema.restore()
        else:
            patience += 1
        if patience >= 5: print(f"  Early stop epoch {epoch+1}"); break

    print(f"  Fold {fold+1} Best Val IC: {best_ic:.4f}")

    # ── Test inference ──
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ms = model.state_dict()
    for k, t in ckpt["model"].items():
        if k in ms and ms[k].shape == t.shape: ms[k].copy_(t)
    model.eval()

    test_loader = DataLoader(Subset(ds, test_idx), batch_size=32,
        shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    preds, labels = [], []
    for batch in test_loader:
        if not batch: continue
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            p, _ = model(batch['values'], batch['company_ids'], batch['metric_ids'],
                        batch['day'], batch['month'], batch['dow'], batch['year_offset'],
                        batch['time_since_update'], batch['mask'].bool(),
                        batch['source_ids'], batch['time_bins'])
        preds.extend(p.cpu().numpy().flatten())
        labels.extend(batch['label'].cpu().numpy().flatten())

    n = min(len(preds), len(test_idx))
    dates = [pd.Timestamp(ds.window_dates[test_idx[i]]) for i in range(n)]
    preds_arr = np.array(preds[:n]); labels_arr = np.array(labels[:n])

    test_ic, _ = pearsonr(preds_arr, labels_arr)
    print(f"  Fold {fold+1} Test IC (63d label): {test_ic:+.4f}")

    # ── Timing strategy (inverted: pred < 0 -> long) ──
    df = csi_ret.to_frame("csi300").join(
        pd.DataFrame({"pred": preds_arr}, index=pd.DatetimeIndex(dates)), how="inner").dropna()

    df["signal"] = df["pred"] < 0  # inverted signal
    df["strat"] = df["csi300"] * df["signal"].astype(float)

    def metrics(rets):
        rets = rets.dropna(); n = len(rets); ny = n / 252
        if ny == 0 or n < 10: return (0, 0, 0, 0, 0, 0, pd.Series([1.0]))
        tr = (1 + rets).prod() - 1; ar = (1 + tr) ** (1 / ny) - 1 if ny > 0 else 0
        vol = rets.std() * np.sqrt(252); sh = (ar - RF) / vol if vol > 0 else 0
        cum = (1 + rets).cumprod()
        dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
        wr = (rets > 0).mean()
        return ar, vol, sh, dd, wr, tr, cum

    ar_s, vo_s, sh_s, dd_s, wr_s, tr_s, cum_s = metrics(df["strat"])
    ar_b, vo_b, sh_b, dd_b, wr_b, tr_b, cum_b = metrics(df["csi300"])

    results.append({
        "fold": fold + 1, "test_year": test_years,
        "train_n": len(train_idx), "test_n": len(test_idx),
        "val_ic": best_ic, "test_ic": test_ic,
        "sharpe": sh_s, "bench_sharpe": sh_b,
        "maxdd": dd_s, "bench_maxdd": dd_b,
        "ann_ret": ar_s, "bench_ann_ret": ar_b,
        "tot_ret": tr_s, "bench_tot_ret": tr_b,
        "win_rate": wr_s, "bench_win_rate": wr_b,
    })
    all_equity_curves[f"Fold {fold+1}"] = cum_s
    print(f"    Sharpe={sh_s:.2f} MaxDD={dd_s*100:.1f}% AnnRet={ar_s*100:+.1f}% TotRet={tr_s*100:+.1f}%")

    del model; torch.cuda.empty_cache()

# ── Report ──
report = []
report.append("=" * 70)
report.append("  5-FOLD WALK-FORWARD CV: INDEX MARKET TIMING STRATEGY")
report.append("=" * 70)
report.append(f"  Signal: pred < 0 -> long CSI300 (inverted, validated in Phase 1)")
report.append(f"  Risk-free rate: {RF*100:.1f}%")
report.append("")
report.append(f"  {'Fold':<6} {'Years':<8} {'Train':>6} {'ValIC':>7} {'TestIC':>7} {'Sharpe':>7} {'BmkSh':>7} {'MaxDD':>7} {'AnnRet':>7} {'TotRet':>7} {'Win':>5}")
report.append(f"  {'-'*78}")

sharpe_list, dd_list, ic_list = [], [], []
for r in results:
    report.append(f"  {r['fold']:<6} {r['test_year']:<8} {r['train_n']:>6} {r['val_ic']:>+7.4f} {r['test_ic']:>+7.4f} {r['sharpe']:>7.2f} {r['bench_sharpe']:>7.2f} {r['maxdd']*100:>6.1f}% {r['ann_ret']*100:>+6.1f}% {r['tot_ret']*100:>+6.1f}% {r['win_rate']*100:>4.0f}%")
    sharpe_list.append(r['sharpe']); dd_list.append(r['maxdd']); ic_list.append(r['test_ic'])

report.append(f"  {'-'*78}")
report.append(f"  {'Mean':<6} {'':<8} {'':>6} {'':>7} {np.mean(ic_list):>+7.4f} {np.mean(sharpe_list):>7.2f} {'':>7} {np.mean(dd_list)*100:>6.1f}%")
report.append(f"  {'Std':<6} {'':<8} {'':>6} {'':>7} {np.std(ic_list):>7.4f} {np.std(sharpe_list):>7.2f} {'':>7} {np.std(dd_list)*100:>6.1f}%")
report.append("")
report.append(f"  Positive Sharpe folds: {sum(1 for s in sharpe_list if s > 0)}/{N_FOLDS} ({sum(1 for s in sharpe_list if s > 0)/N_FOLDS*100:.0f}%)")
report.append(f"  Sharpe > Benchmark: {sum(1 for i in range(N_FOLDS) if sharpe_list[i] > results[i]['bench_sharpe'])}/{N_FOLDS}")
report.append(f"  Mean Test IC: {np.mean(ic_list):+.4f} ± {np.std(ic_list):.4f}")
report.append("")
if sum(1 for s in sharpe_list if s > 0) >= 3:
    report.append("  VERDICT: STRATEGY IS ROBUST across market regimes.")
else:
    report.append("  VERDICT: Strategy NOT robust — inconsistent across folds.")
report.append("=" * 70)

rtext = "\n".join(report)
with open("reports/walkforward_report.txt", "w") as f: f.write(rtext)
print(rtext)

# ── Chart ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: equity curves
colors = ['#1a73e8', '#e37400', '#0d904f', '#d93025', '#9334e6']
for i, (label, cum) in enumerate(all_equity_curves.items()):
    ax1.plot(cum.index, cum.values, color=colors[i], label=label, lw=2)
ax1.axhline(y=1.0, color='black', ls='--', alpha=0.3)
ax1.set_title('Walk-Forward Equity Curves (per Fold)', fontweight='bold')
ax1.set_ylabel('Cumulative Return'); ax1.legend(); ax1.grid(True, alpha=0.3)

# Right: Sharpe bar chart
folds = [f"F{i+1}" for i in range(N_FOLDS)]
x = np.arange(N_FOLDS)
w = 0.35
ax2.bar(x - w/2, sharpe_list, w, label='Strategy', color='#1a73e8')
ax2.bar(x + w/2, [r['bench_sharpe'] for r in results], w, label='B&H', color='gray', alpha=0.7)
ax2.axhline(y=0, color='black', ls='-', alpha=0.3)
ax2.set_xticks(x); ax2.set_xticklabels(folds)
ax2.set_title('Sharpe Ratio by Fold'); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/walkforward_equity.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: reports/walkforward_report.txt + reports/walkforward_equity.png")
