"""
Test eval: use pre-computed windows from dataset cache.
Reads only windows directly from dataset, no full parquet load.
"""
import sys; sys.path.insert(0,'.')
import numpy as np, torch
from torch.utils.data import DataLoader
from utils.dataset import SlidingWindowDataset, collate_fn
from model import FinancialMoETransformer
from config import model_cfg

print("Reading labels...")
import pandas as pd
labels = pd.read_parquet("D:/financial_data/processed/labels.parquet")
labels["datetime"] = pd.to_datetime(labels["datetime"])
labels = labels.sort_values("datetime")
n = len(labels)
test_start = int(n * 0.75)

# Only read data from ~2022 onwards to fit memory
min_dt = labels["datetime"].iloc[max(0, test_start - 300)]
print(f"Data from {min_dt.date()}")

# Read only needed columns + rows
import pyarrow.parquet as pq
table = pq.read_table("D:/financial_data/processed/data_points.parquet",
                       filters=[("datetime", ">=", pd.Timestamp(min_dt))])
dp = table.to_pandas()
dp["datetime"] = pd.to_datetime(dp["datetime"])
print(f"Loaded: {len(dp):,} rows")

# Build dataset
ds = SlidingWindowDataset.__new__(SlidingWindowDataset)
ds.data = dp.sort_values("datetime").reset_index(drop=True)
ds.labels = labels
ds.window_days = 365; ds.forecast_horizon = 63
ds.max_seq_len = 8192; ds.base_year = 2015
ds.use_cache = False; ds.multi_window = False
ds._window_options = [365]
ds.company_to_id, ds.metric_to_id = ds._build_dual_vocab()
ds.n_companies = len(ds.company_to_id)
ds.n_metrics = len(ds.metric_to_id)
ds.window_dates = ds._build_windows()
ds.var_to_id = ds.company_to_id

# Keep only test windows (most recent ~25%)
all_dt = pd.to_datetime(ds.window_dates)
cutoff = labels["datetime"].iloc[test_start]
test_windows = all_dt[all_dt >= cutoff]
ds.window_dates = np.array(test_windows)
print(f"Vocab: {len(ds.company_to_id)}c x {len(ds.metric_to_id)}m | Test windows: {len(ds.window_dates)}")

# Model
# Use checkpoint's vocab sizes, not filtered dataset's
model = FinancialMoETransformer(
    n_companies=6269, n_metrics=126,
    d_model=128, nhead=4, num_layers=model_cfg.num_layers,
    dim_feedforward=model_cfg.dim_feedforward,
    num_experts=model_cfg.num_experts, expert_hidden=model_cfg.expert_hidden,
)
ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"]); model.eval()
print(f"Model: epoch {ckpt['epoch']+1}, Val IC={ckpt['best_ic']:.4f}")

# Streaming eval
class StreamingStats:
    def __init__(self):
        self.n=0; self.sx=0.0; self.sy=0.0; self.sxy=0.0; self.sx2=0.0; self.sy2=0.0
        self.gates=np.zeros(6)
    def add(self,p,l,g):
        n=len(p); self.n+=n; self.sx+=p.sum(); self.sy+=l.sum()
        self.sxy+=(p*l).sum(); self.sx2+=(p**2).sum(); self.sy2+=(l**2).sum()
        self.gates+=np.bincount(np.argmax(g,axis=1),minlength=6)
    def ic(self):
        n,sx,sy=self.n,self.sx,self.sy
        num=n*self.sxy-sx*sy
        den=np.sqrt(max(0,n*self.sx2-sx**2)*max(0,n*self.sy2-sy**2))
        return num/den if den>0 else 0.0

stats = StreamingStats()
AP, AL = [], []
loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)
for i,batch in enumerate(loader):
    if not batch or 'values' not in batch: continue
    with torch.no_grad():
        p,_ = model(batch['values'],batch['company_ids'],batch['metric_ids'],
                    batch['day'],batch['month'],batch['dow'],batch['year_offset'],
                    batch['time_since_update'],batch['mask'].bool(),
                    batch['source_ids'],batch['time_bins'])
        x=model.embedding(batch['values'],batch['company_ids'],batch['metric_ids'],
                         batch['day'],batch['month'],batch['dow'],batch['year_offset'],
                         batch['time_since_update'])
        x,hm=model.hierarchical(x,batch['mask'].bool(),batch['source_ids'],batch['time_bins'])
        g=model.moe.get_expert_weights(model.pooling(x,hm))
    P=p.numpy().flatten(); L=batch['label'].numpy().flatten(); G=g.numpy()
    stats.add(P,L,G)
    del G  # avoid confusion with g
    AP.extend(P); AL.extend(L)
    if (i+1)%30==0: print(f"  {i+1}/{len(loader)} | IC={stats.ic():+.4f} | n={stats.n}")

AP=np.array(AP); AL=np.array(AL)
ic=stats.ic(); ric=np.corrcoef(AP.argsort().argsort(),AL.argsort().argsort())[0,1]
print(f"\nTest IC: {ic:+.4f}  Rank IC: {ric:+.4f}  ({stats.n} samples)")
for e in range(6):
    u=stats.gates[e]/stats.gates.sum()*100
    print(f"  Expert {e+1}: {u:5.1f}%  {'#'*int(u)}")
cv=np.std(stats.gates/stats.gates.sum())/(1/6)
print(f"  CV: {cv:.3f}")
