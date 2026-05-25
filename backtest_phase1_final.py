"""
Phase 1: Index Market Timing Backtest using existing model (IC=0.053).
"""
import sys; sys.path.insert(0,'.')
import torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader, Subset
from utils.dataset import SlidingWindowDataset, collate_fn
from model import FinancialMoETransformer
from config import model_cfg
from scipy.stats import pearsonr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE, RF = "cuda", 0.025
CKPT = "checkpoints/best_phase2_cont.pt"

# Data
ds = SlidingWindowDataset(
    data_path="data/processed/data_points.parquet",
    labels_path="data/processed/labels.parquet",
    max_seq_len=8192, use_cache=True, cache_dir="data/processed/cache_p1",
    multi_window=False)
_, _, test_dates = ds.train_val_test_split()
test_idx = [i for i, d in enumerate(ds.window_dates) if d in test_dates]
print(f"Dataset: {len(ds)} windows, Test={len(test_idx)}")

# Model
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model = FinancialMoETransformer(
    n_companies=ds.n_companies+2, n_metrics=ds.n_metrics+2,
    d_model=128, nhead=4, num_layers=4, dim_feedforward=512,
    dropout=0.2, attn_type="cosformer", attn_chunk_size=2048,
    num_experts=6, expert_hidden=64, top_k=1).to(DEVICE)
ms = model.state_dict()
for k, t in ckpt["model"].items():
    if k in ms and ms[k].shape == t.shape: ms[k].copy_(t)
    elif k in ms and ms[k].dim() == 2:
        m0, m1 = min(ms[k].shape[0], t.shape[0]), min(ms[k].shape[1], t.shape[1])
        ms[k][:m0, :m1].copy_(t[:m0, :m1])
model.eval()
print(f"Model: IC={ckpt.get('best_ic',0):.4f}")

# Inference
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
    preds.extend(p.cpu().numpy().flatten())
    labels.extend(batch['label'].cpu().numpy().flatten())

n = min(len(preds), len(test_idx))
dates = [pd.Timestamp(ds.window_dates[test_idx[i]]) for i in range(n)]
preds_arr = np.array(preds[:n]); labels_arr = np.array(labels[:n])
test_ic, _ = pearsonr(preds_arr, labels_arr)

# Join with CSI 300 returns
csi = pd.read_csv("D:/financial_data/market/csi300_daily.csv")
csi["date"] = pd.to_datetime(csi["date"]); csi = csi.sort_values("date").set_index("date")
csi_ret = csi["close"].pct_change().dropna()

df = csi_ret.to_frame("csi300").join(
    pd.DataFrame({"pred": preds_arr}, index=pd.DatetimeIndex(dates)), how="inner").dropna()

# Strategies
median = df["pred"].median()
df["sig_pos"] = df["pred"] > 0
df["sig_med"] = df["pred"] > median
df["sig_inv"] = df["pred"] < median  # inverted signal
df["sig_top"] = df["pred"] > df["pred"].quantile(0.7)
df["s_pos"] = df["csi300"] * df["sig_pos"].astype(float)
df["s_med"] = df["csi300"] * df["sig_med"].astype(float)
df["s_inv"] = df["csi300"] * df["sig_inv"].astype(float)
df["s_top"] = df["csi300"] * df["sig_top"].astype(float)

def m(rets):
    rets=rets.dropna();n=len(rets);ny=n/252
    if ny==0:return(0,0,0,0,0,0,0)
    tr=(1+rets).prod()-1;ar=(1+tr)**(1/ny)-1;vol=rets.std()*np.sqrt(252)
    sh=(ar-RF)/vol if vol>0 else 0
    cum=(1+rets).cumprod();dd=((cum-cum.expanding().max())/cum.expanding().max()).min()
    wr=(rets>0).mean();return ar,vol,sh,dd,wr,tr,cum

ar_b,vo_b,sh_b,dd_b,wr_b,tr_b,cum_b=m(df["csi300"])
ar_s,vo_s,sh_s,dd_s,wr_s,tr_s,cum_s=m(df["s_inv"])

report=[]
report.append("="*65)
report.append("  PHASE 1: INDEX MARKET TIMING BACKTEST")
report.append("="*65)
report.append(f"  Model: {CKPT} (Val IC={ckpt.get('best_ic',0):.4f})")
report.append(f"  Period: {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} days)")
report.append(f"  Test IC (63d label): {test_ic:+.4f}")
report.append(f"  Strategy: pred > median({median:.4f}) -> long CSI300, else cash")
report.append("")
report.append(f"  {'Metric':<22} {'Timing':>12} {'CSI 300 B&H':>12}")
report.append(f"  {'-'*46}")
report.append(f"  {'Annual Return':<22} {ar_s*100:>+11.2f}% {ar_b*100:>+11.2f}%")
report.append(f"  {'Annual Vol':<22} {vo_s*100:>11.1f}% {vo_b*100:>11.1f}%")
report.append(f"  {'Sharpe Ratio':<22} {sh_s:>12.2f} {sh_b:>12.2f}")
report.append(f"  {'Max Drawdown':<22} {dd_s*100:>11.1f}% {dd_b*100:>11.1f}%")
report.append(f"  {'Win Rate':<22} {wr_s*100:>11.0f}% {wr_b*100:>11.0f}%")
report.append(f"  {'Total Return':<22} {tr_s*100:>+11.1f}% {tr_b*100:>+11.1f}%")
report.append("")
report.append("  Yearly Returns:")
for yr,grp in df.groupby(df.index.year):
    s=(1+grp["s_inv"]).prod()-1;b=(1+grp["csi300"]).prod()-1
    report.append(f"    {yr}: Timing={s*100:+6.1f}%  B&H={b*100:+6.1f}%")
s_all=(1+df["s_inv"]).prod()-1;b_all=(1+df["csi300"]).prod()-1
report.append(f"    TOTAL: Timing={s_all*100:+6.1f}%  B&H={b_all*100:+6.1f}%")
report.append("="*65)

rtext="\n".join(report)
with open("reports/timing_report.txt","w") as f:f.write(rtext)
print(rtext)

# Chart
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(14,10),gridspec_kw={'height_ratios':[3,1]})
ax1.plot(cum_b.index,cum_b.values,'gray',alpha=.7,label='CSI 300 B&H',lw=1.5)
ax1.plot(cum_s.index,cum_s.values,'#1a73e8',label=f'Timing (Sharpe={sh_s:.2f})',lw=2)
ax1.fill_between(cum_s.index,1.0,cum_s.values,alpha=.1,color='#1a73e8')
ax1.axhline(y=1.0,color='black',ls='--',alpha=.3)
ax1.set_ylabel('Cumulative Return');ax1.set_title('Index Market Timing Strategy',fontweight='bold')
ax1.legend();ax1.grid(True,alpha=.3)
dd_b=(cum_b-cum_b.expanding().max())/cum_b.expanding().max()*100
dd_s=(cum_s-cum_s.expanding().max())/cum_s.expanding().max()*100
ax2.fill_between(dd_b.index,0,dd_b.values,color='gray',alpha=.3,label='B&H DD')
ax2.fill_between(dd_s.index,0,dd_s.values,color='#d93025',alpha=.4,label='Timing DD')
ax2.set_ylabel('Drawdown %');ax2.legend();ax2.grid(True,alpha=.3)
plt.tight_layout()
plt.savefig("reports/timing_backtest.png",dpi=150,bbox_inches='tight')
plt.close()
print(f"\nSaved: reports/timing_report.txt + reports/timing_backtest.png")
