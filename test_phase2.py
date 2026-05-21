"""Phase 2 test set evaluation — quick."""
import sys; sys.path.insert(0,'.')
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from utils.dataset import SlidingWindowDataset, collate_fn
from model import FinancialMoETransformer
from config import model_cfg

ds = SlidingWindowDataset(max_seq_len=8192, use_cache=False)
_, _, test_dates = ds.train_val_test_split()
test_idx = [i for i, d in enumerate(ds.window_dates) if d in test_dates]

model = FinancialMoETransformer(
    n_companies=ds.n_companies+2, n_metrics=ds.n_metrics+2,
    d_model=128, nhead=4, num_layers=model_cfg.num_layers,
    dim_feedforward=model_cfg.dim_feedforward,
    num_experts=model_cfg.num_experts, expert_hidden=model_cfg.expert_hidden,
)
ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

loader = DataLoader(Subset(ds, test_idx), batch_size=2, shuffle=False,
                    collate_fn=collate_fn, num_workers=0, pin_memory=False)

preds, labels, gates_l = [], [], []
with torch.no_grad():
    for batch in loader:
        if not batch or 'values' not in batch: continue
        pred, _ = model(
            batch['values'], batch['company_ids'], batch['metric_ids'],
            batch['day'], batch['month'], batch['dow'],
            batch['year_offset'], batch['time_since_update'],
            batch['mask'].bool(), batch['source_ids'], batch['time_bins'],
        )
        x = model.embedding(
            batch['values'], batch['company_ids'], batch['metric_ids'],
            batch['day'], batch['month'], batch['dow'],
            batch['year_offset'], batch['time_since_update'],
        )
        x, hm = model.hierarchical(x, batch['mask'].bool(),
                                   batch['source_ids'], batch['time_bins'])
        xp = model.pooling(x, hm)
        gates_l.append(model.moe.get_expert_weights(xp).numpy())
        preds.extend(pred.numpy().flatten())
        labels.extend(batch['label'].numpy().flatten())

preds = np.array(preds); labels = np.array(labels)
gates = np.concatenate(gates_l, axis=0); top1 = gates.argmax(axis=1)

ic = np.corrcoef(preds, labels)[0,1]
print(f"Phase 2 Test IC: {ic:+.4f}  ({len(preds)} samples)")
for e in range(6):
    print(f"  Expert {e+1}: {(top1==e).mean()*100:5.1f}%")
cv = np.std([(top1==e).mean() for e in range(6)]) / (1/6)
print(f"  CV: {cv:.3f}")
