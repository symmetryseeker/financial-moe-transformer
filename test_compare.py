"""Compare V2 vs Phase1 on held-out test set."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Subset
from utils.dataset import SlidingWindowDataset, collate_fn
from model import FinancialMoETransformer
from config import model_cfg

ds = SlidingWindowDataset(max_seq_len=8192, use_cache=False)
_, _, test_dates = ds.train_val_test_split()
test_idx = [i for i, d in enumerate(ds.window_dates) if d in test_dates]

def evaluate(checkpoint_path):
    model = FinancialMoETransformer(
        n_companies=ds.n_companies+2, n_metrics=ds.n_metrics+2,
        d_model=128, nhead=4, num_layers=model_cfg.num_layers,
        dim_feedforward=model_cfg.dim_feedforward,
        num_experts=model_cfg.num_experts, expert_hidden=model_cfg.expert_hidden,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(Subset(ds, test_idx), batch_size=2, shuffle=False,
                        collate_fn=collate_fn, num_workers=0, pin_memory=False)
    preds, labels, gates_list = [], [], []
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
            gates = model.moe.get_expert_weights(xp)
            preds.extend(pred.numpy().flatten())
            labels.extend(batch['label'].numpy().flatten())
            gates_list.append(gates.numpy())

    preds = np.array(preds); labels = np.array(labels)
    gates = np.concatenate(gates_list, axis=0)
    ic = np.corrcoef(preds, labels)[0,1]
    rank_ic = np.corrcoef(preds.argsort().argsort(), labels.argsort().argsort())[0,1]
    top1 = gates.argmax(axis=1)
    return ic, rank_ic, top1, preds, labels

# V2 baseline (from earlier independent test eval)
ic_v2 = 0.0261; ric_v2 = 0.0168
cv_v2 = 1.230
print(f"V2 Baseline:  Test IC = +{ic_v2:.4f}  Rank IC = +{ric_v2:.4f}  CV = {cv_v2:.3f}")

print("\nEvaluating Phase 1 on Test Set...")
ic_p1, ric_p1, top1_p1, preds_p1, labels_p1 = evaluate("checkpoints/best.pt")
print(f"  Phase1 Test IC: {ic_p1:+.4f}  Rank IC: {ric_p1:+.4f}")
print(f"  N samples: {len(preds_p1)}")

print(f"\nExpert Usage (Phase 1, Test Set):")
for e in range(6):
    usage = (top1_p1 == e).mean() * 100
    print(f"  Expert {e+1}: {usage:5.1f}%")
cv = np.std([(top1_p1==e).mean() for e in range(6)]) / (1/6)
print(f"  CV: {cv:.3f} (V2 was 1.230)")

print(f"\nComparison:")
print(f"  V2 Baseline:  Test IC = +{ic_v2:.4f}")
print(f"  Phase 1:      Test IC = {ic_p1:+.4f}")
print(f"  Change:       {ic_p1-ic_v2:+.4f}")
