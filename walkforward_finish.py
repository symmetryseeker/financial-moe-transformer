"""Complete WF-CV: backtest Folds 1-4 from checkpoints, train Fold 5."""
import sys; sys.path.insert(0,'.')
import torch, numpy as np, pandas as pd
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

DEVICE = "cuda"; RF = 0.025; N_FOLDS = 5

ds = SlidingWindowDataset(
    data_path="data/processed/data_points.parquet",
    labels_path="data/processed/labels.parquet",
    max_seq_len=8192, use_cache=True, cache_dir="data/processed/cache_wf2",
    multi_window=False)

csi = pd.read_csv("D:/financial_data/market/csi300_daily.csv")
csi["date"] = pd.to_datetime(csi["date"]); csi = csi.sort_values("date").set_index("date")
csi_ret = csi["close"].pct_change().dropna()

def build_model():
    return FinancialMoETransformer(
        n_companies=ds.n_companies+2, n_metrics=ds.n_metrics+2,
        d_model=128, nhead=4, num_layers=4, dim_feedforward=512,
        dropout=0.2, attn_type="cosformer", attn_chunk_size=2048,
        num_experts=6, expert_hidden=64, top_k=1).to(DEVICE)

def metrics(rets):
    rets = rets.dropna(); n = len(rets); ny = n / 252
    if ny == 0 or n < 10: return (0, 0, 0, 0, 0, 0, pd.Series([1.0]))
    tr = (1 + rets).prod() - 1; ar = (1 + tr) ** (1 / ny) - 1 if ny > 0 else 0
    vol = rets.std() * np.sqrt(252); sh = (ar - RF) / vol if vol > 0 else 0
    cum = (1 + rets).cumprod(); dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    wr = (rets > 0).mean(); return ar, vol, sh, dd, wr, tr, cum

def backtest_fold(fold, test_idx):
    ckpt = torch.load(MODEL_DIR / f"wf_fold{fold}.pt", map_location="cpu", weights_only=False)
    model = build_model()
    ms = model.state_dict()
    for k, t in ckpt["model"].items():
        if k in ms and ms[k].shape == t.shape: ms[k].copy_(t)
        elif k in ms and ms[k].dim() == 2:
            m0, m1 = min(ms[k].shape[0], t.shape[0]), min(ms[k].shape[1], t.shape[1])
            ms[k][:m0, :m1].copy_(t[:m0, :m1])
    model.eval()

    loader = DataLoader(Subset(ds, test_idx), batch_size=32, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    preds, labels = [], []
    for batch in loader:
        if not batch: continue
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            p, _ = model(batch['values'], batch['company_ids'], batch['metric_ids'],
                        batch['day'], batch['month'], batch['dow'], batch['year_offset'],
                        batch['time_since_update'], batch['mask'].bool(),
                        batch['source_ids'], batch['time_bins'])
        preds.extend(p.cpu().numpy().flatten()); labels.extend(batch['label'].cpu().numpy().flatten())

    n = min(len(preds), len(test_idx))
    dates = [pd.Timestamp(ds.window_dates[test_idx[i]]) for i in range(n)]
    preds_arr = np.array(preds[:n]); labels_arr = np.array(labels[:n])
    test_ic, _ = pearsonr(preds_arr, labels_arr)

    df = csi_ret.to_frame("csi300").join(
        pd.DataFrame({"pred": preds_arr}, index=pd.DatetimeIndex(dates)), how="inner").dropna()
    df["strat"] = df["csi300"] * (df["pred"] < 0).astype(float)
    ar_s, vo_s, sh_s, dd_s, wr_s, tr_s, cum_s = metrics(df["strat"])
    ar_b, vo_b, sh_b, dd_b, wr_b, tr_b, cum_b = metrics(df["csi300"])
    return {
        "fold": fold, "val_ic": ckpt.get("best_ic", 0), "test_ic": test_ic,
        "sharpe": sh_s, "bench_sharpe": sh_b, "maxdd": dd_s, "ann_ret": ar_s,
        "tot_ret": tr_s, "win_rate": wr_s, "cum": cum_s, "test_n": len(test_idx)
    }

results = []
test_years = [2020, 2021, 2022, 2023, 2024]
test_ranges = [("2020-01-01","2020-12-31"),("2021-01-01","2021-12-31"),
               ("2022-01-01","2022-12-31"),("2023-01-01","2023-12-31"),
               ("2024-01-01","2025-12-26")]

# Backtest Folds 1-4
for fold in range(1, 5):
    t0, t1 = test_ranges[fold-1]
    test_idx = [i for i, d in enumerate(ds.window_dates) if pd.Timestamp(t0) <= d <= pd.Timestamp(t1)]
    r = backtest_fold(fold, test_idx)
    results.append(r)
    print(f"Fold {fold}: Val IC={r['val_ic']:+.4f} Test IC={r['test_ic']:+.4f} "
          f"Sharpe={r['sharpe']:+.2f} MaxDD={r['maxdd']*100:.1f}% AnnRet={r['ann_ret']*100:+.1f}%")
    del r['cum']

# Train Fold 5
fold = 5; t0, t1 = test_ranges[fold-1]
test_start = pd.Timestamp(t0)
train_end = test_start - pd.Timedelta(days=1)
train_idx = [i for i, d in enumerate(ds.window_dates) if d <= train_end]
split = int(len(train_idx) * 0.85)
val_idx = train_idx[split:]; train_idx = train_idx[:split]
test_idx = [i for i, d in enumerate(ds.window_dates) if pd.Timestamp(t0) <= d <= pd.Timestamp(t1)]
print(f"\nFold 5: Train={len(train_idx)} Val={len(val_idx)} Test={len(test_idx)}")

tr_loader = DataLoader(Subset(ds, train_idx), batch_size=4, shuffle=True,
    collate_fn=collate_fn, num_workers=0, pin_memory=True, drop_last=True)
v_loader = DataLoader(Subset(ds, val_idx), batch_size=4, shuffle=False,
    collate_fn=collate_fn, num_workers=0, pin_memory=True)

model = build_model()
opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
sp = max(1, len(tr_loader)); total = sp * 15
ws = min(500, max(1, total // 5))
warmup = LinearLR(opt, start_factor=1e-3, total_iters=ws)
cosine = CosineAnnealingLR(opt, T_max=total - ws, eta_min=3e-6)
scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[ws])
loss_fn = CombinedLoss(mse_coef=1.0, load_balance_coef=0.01, rank_coef=0.3, direction_coef=0.0)
ema = EMA(model, decay=0.999)
best_ic = -float("inf"); patience = 0; ckpt_path = MODEL_DIR / "wf_fold5.pt"

for epoch in range(15):
    import time
    t0_t = time.time()
    tl = train_epoch(model, tr_loader, opt, scheduler, DEVICE, train_cfg, loss_fn)
    ema.update(); ema.apply_shadow()
    vm = validate(model, v_loader, DEVICE)
    ema.restore()
    print(f"  Ep {epoch+1} ({time.time()-t0_t:.0f}s) MSE={tl.get('mse',0):.3f} Val IC={vm['ic']:.4f}", flush=True)
    if vm["ic"] > best_ic:
        best_ic = vm["ic"]; patience = 0
        ema.apply_shadow(); torch.save({"epoch": epoch, "model": model.state_dict(), "best_ic": best_ic, "fold": 5}, ckpt_path); ema.restore()
    else:
        patience += 1
    if patience >= 5: print(f"  Early stop epoch {epoch+1}"); break

r5 = backtest_fold(5, test_idx)
r5["fold"] = 5; results.append(r5)
print(f"Fold 5: Val IC={r5['val_ic']:+.4f} Test IC={r5['test_ic']:+.4f} "
      f"Sharpe={r5['sharpe']:+.2f} MaxDD={r5['maxdd']*100:.1f}% AnnRet={r5['ann_ret']*100:+.1f}%")

# ── Report ──
report = []
report.append("=" * 70)
report.append("  5-FOLD WALK-FORWARD CV: INDEX MARKET TIMING STRATEGY")
report.append("=" * 70)
report.append(f"  Signal: pred < 0 -> long CSI300, else cash")
report.append(f"  Risk-free rate: {RF*100:.1f}%")
report.append("")
report.append(f"  {'Fold':<6} {'Year':<8} {'ValIC':>7} {'TestIC':>7} {'Sharpe':>7} {'BmkSh':>7} {'MaxDD':>7} {'AnnRet':>8} {'TotRet':>8} {'Win':>5}")
report.append(f"  {'-'*80}")
sh_list, dd_list = [], []
for r in results:
    report.append(f"  {r['fold']:<6} {test_years[r['fold']-1]:<8} {r['val_ic']:>+7.4f} {r['test_ic']:>+7.4f} {r['sharpe']:>7.2f} {r['bench_sharpe']:>7.2f} {r['maxdd']*100:>6.1f}% {r['ann_ret']*100:>+7.1f}% {r['tot_ret']*100:>+7.1f}% {r['win_rate']*100:>4.0f}%")
    sh_list.append(r['sharpe']); dd_list.append(r['maxdd'])
report.append(f"  {'-'*80}")
report.append(f"  {'Mean':<6} {'':<8} {'':>7} {'':>7} {np.mean(sh_list):>7.2f} {'':>7} {np.mean(dd_list)*100:>6.1f}%")
report.append(f"  {'Std':<6} {'':<8} {'':>7} {'':>7} {np.std(sh_list):>7.2f} {'':>7} {np.std(dd_list)*100:>6.1f}%")
positive = sum(1 for s in sh_list if s > 0)
report.append(f"\n  Positive Sharpe: {positive}/{N_FOLDS} folds ({positive/N_FOLDS*100:.0f}%)")
if positive >= 3: report.append("  VERDICT: STRATEGY IS ROBUST across market regimes.")
else: report.append("  VERDICT: Strategy NOT robust.")
report.append("=" * 70)

rtext = "\n".join(report)
with open("reports/walkforward_report.txt", "w") as f: f.write(rtext)
print("\n" + rtext)

# Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
colors = ['#1a73e8', '#e37400', '#0d904f', '#d93025', '#9334e6']
for i, r in enumerate(results):
    ax1.plot(r['cum'].index, r['cum'].values, color=colors[i], label=f"Fold {i+1} ({test_years[i]})", lw=2)
ax1.axhline(y=1.0, color='black', ls='--', alpha=0.3)
ax1.set_title('WF-CV Equity Curves', fontweight='bold'); ax1.legend(); ax1.grid(True, alpha=0.3)
x = np.arange(N_FOLDS); w = 0.35
ax2.bar(x - w/2, sh_list, w, label='Strategy', color='#1a73e8')
ax2.bar(x + w/2, [r['bench_sharpe'] for r in results], w, label='B&H', color='gray', alpha=0.7)
ax2.axhline(y=0, color='black', ls='-', alpha=0.3)
ax2.set_xticks(x); ax2.set_xticklabels([f"F{i+1}\n({test_years[i]})" for i in range(N_FOLDS)])
ax2.set_title('Sharpe by Fold'); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("reports/walkforward_equity.png", dpi=150, bbox_inches='tight'); plt.close()
print("Saved: reports/walkforward_report.txt + reports/walkforward_equity.png")
