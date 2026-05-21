import sys; sys.path.insert(0,'.')
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from utils.dataset import SlidingWindowDataset, collate_fn
from model import FinancialMoETransformer
from config import model_cfg
ds = SlidingWindowDataset(max_seq_len=8192, use_cache=False)
_, _, td = ds.train_val_test_split()
ti = [i for i, d in enumerate(ds.window_dates) if d in td]
m = FinancialMoETransformer(n_companies=ds.n_companies+2, n_metrics=ds.n_metrics+2,
    d_model=128, nhead=4, num_layers=model_cfg.num_layers,
    dim_feedforward=model_cfg.dim_feedforward,
    num_experts=model_cfg.num_experts, expert_hidden=model_cfg.expert_hidden)
m.load_state_dict(torch.load('checkpoints/best.pt',map_location='cpu',weights_only=False)['model'])
m.eval()
l = DataLoader(Subset(ds,ti),batch_size=2,shuffle=False,collate_fn=collate_fn,num_workers=0)
P,L,G=[],[],[]
for b in l:
    if not b or 'values' not in b: continue
    with torch.no_grad():
        p,_ = m(b['values'],b['company_ids'],b['metric_ids'],b['day'],b['month'],b['dow'],b['year_offset'],b['time_since_update'],b['mask'].bool(),b['source_ids'],b['time_bins'])
        x=m.embedding(b['values'],b['company_ids'],b['metric_ids'],b['day'],b['month'],b['dow'],b['year_offset'],b['time_since_update'])
        x,hm=m.hierarchical(x,b['mask'].bool(),b['source_ids'],b['time_bins'])
        G.append(m.moe.get_expert_weights(m.pooling(x,hm)).numpy())
    P.extend(p.numpy().flatten()); L.extend(b['label'].numpy().flatten())
P=np.array(P); L=np.array(L); G=np.concatenate(G); T=G.argmax(1)
ic=np.corrcoef(P,L)[0,1]; ric=np.corrcoef(P.argsort().argsort(),L.argsort().argsort())[0,1]
print(f'Test IC: {ic:+.4f}  Rank IC: {ric:+.4f}  ({len(P)} samples)')
for e in range(6): print(f'  Expert {e+1}: {(T==e).mean()*100:5.1f}%')
cv=np.std([(T==e).mean() for e in range(6)])/(1/6)
print(f'  CV: {cv:.3f}')
