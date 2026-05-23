"""
Stock-level SlidingWindowDataset.
Each window = (end_date, stock_code) with a stock-specific label.
Supports Walk-Forward CV with date-based splitting.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Tuple, Optional
import json


class StockSlidingWindowDataset(Dataset):
    def __init__(self,
                 data_path: str = "data/processed/data_points.parquet",
                 labels_path: str = "data/processed/labels_stock.parquet",
                 window_days: int = 365,
                 forecast_horizon: int = 63,
                 max_seq_len: int = 8192,
                 base_year: int = 2015,
                 use_cache: bool = True,
                 cache_dir: str = "data/processed/cache_stock",
                 multi_window: bool = True,
                 min_windows_per_stock: int = 50):
        self.window_days = window_days
        self.forecast_horizon = forecast_horizon
        self.max_seq_len = max_seq_len
        self.base_year = base_year
        self.multi_window = multi_window
        self._window_options = [126, 252, 365, 504]
        self.cache_dir = Path(cache_dir) if use_cache else None
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        print(f"Loading data from {data_path} ...")
        self.data = pd.read_parquet(data_path)
        self.data["datetime"] = pd.to_datetime(self.data["datetime"])
        self.data = self.data.sort_values("datetime").reset_index(drop=True)

        # Pre-parse all variables into company + metric columns (vectorized, once)
        print("Pre-parsing variables...")
        vars_arr = self.data["variable"].astype(str)
        has_sep = vars_arr.str.contains("::", na=False)
        split = vars_arr.str.split("::", n=1)
        self.data["_company"] = np.where(has_sep, split.str[0], "")
        self.data["_metric"] = np.where(has_sep, split.str[1], vars_arr)
        self.data["_value_num"] = pd.to_numeric(self.data["value"], errors="coerce").fillna(0).astype(np.float32)
        self.data["_tsu"] = pd.to_numeric(self.data["time_since_update"], errors="coerce").fillna(0).astype(np.float32)
        src_map = {"market": 0, "macro": 1, "financial": 2, "alternative": 3, "sentiment": 4}
        self.data["_source_id"] = self.data["source"].map(src_map).fillna(0).astype(np.int64)
        self.data["_day"] = self.data["datetime"].dt.day.values.astype(np.int64)
        self.data["_month"] = self.data["datetime"].dt.month.values.astype(np.int64)
        self.data["_dow"] = self.data["datetime"].dt.dayofweek.values.astype(np.int64)
        self.data["_year_off"] = (self.data["datetime"].dt.year - self.base_year).values.astype(np.int64)
        print("  Done pre-parsing.")

        # Load stock labels
        print(f"Loading stock labels from {labels_path} ...")
        self.labels = pd.read_parquet(labels_path)
        self.labels["datetime"] = pd.to_datetime(self.labels["datetime"])
        self.labels = self.labels.sort_values(["datetime", "stock_code"]).reset_index(drop=True)

        # Filter stocks with enough data
        stock_counts = self.labels.groupby("stock_code").size()
        valid_stocks = stock_counts[stock_counts >= min_windows_per_stock].index
        self.labels = self.labels[self.labels["stock_code"].isin(valid_stocks)]
        print(f"  Stocks: {self.labels['stock_code'].nunique()} (min {min_windows_per_stock} windows)")

        # Build vocab
        self.company_to_id, self.metric_to_id = self._build_dual_vocab()
        self.n_companies = len(self.company_to_id)
        self.n_metrics = len(self.metric_to_id)
        self.var_to_id = self.company_to_id

        # Build window index: list of (end_date, stock_code) pairs
        self.window_index = self._build_window_index()
        print(f"Built {len(self.window_index):,} stock-windows")

    def _build_dual_vocab(self):
        vocab_dir = self.cache_dir or Path("data/processed/cache_stock")
        vocab_dir.mkdir(parents=True, exist_ok=True)
        c_path, m_path = vocab_dir / "company_vocab.json", vocab_dir / "metric_vocab.json"

        existing_c = json.loads(c_path.read_text()) if c_path.exists() else {}
        existing_m = json.loads(m_path.read_text()) if m_path.exists() else {}

        variables = self.data["variable"].unique()
        companies, metrics = set(), set()
        for v in variables:
            vstr = str(v)
            if "::" in vstr:
                comp, met = vstr.split("::", 1)
                companies.add(comp); metrics.add(met)
            else:
                metrics.add(vstr)

        if existing_c:
            max_id = max(existing_c.values())
            for c in sorted(companies - set(existing_c.keys())):
                max_id += 1; existing_c[c] = max_id
            company_to_id = existing_c
        else:
            company_to_id = {c: i+1 for i, c in enumerate(sorted(companies))}
        if existing_m:
            max_id = max(existing_m.values())
            for m in sorted(metrics - set(existing_m.keys())):
                max_id += 1; existing_m[m] = max_id
            metric_to_id = existing_m
        else:
            metric_to_id = {m: i+1 for i, m in enumerate(sorted(metrics))}

        company_to_id[""] = 0; metric_to_id[""] = 0
        c_path.write_text(json.dumps(company_to_id, ensure_ascii=False))
        m_path.write_text(json.dumps(metric_to_id, ensure_ascii=False))
        print(f"Dual vocab: {len(company_to_id):,}c x {len(metric_to_id):,}m")
        return company_to_id, metric_to_id

    def _build_window_index(self):
        """Build list of (end_date, stock_code) for all valid windows."""
        unique_dates = pd.to_datetime(self.labels["datetime"].unique())
        unique_dates = np.sort(unique_dates)
        data_min = self.data["datetime"].min()
        data_max = self.data["datetime"].max()

        window_index = []
        for end_date in unique_dates:
            # Check: enough lookback data
            if end_date - pd.Timedelta(days=self.window_days) < data_min:
                continue
            # Check: label date exists
            target_date = end_date + pd.Timedelta(days=self.forecast_horizon)
            if target_date > data_max:
                continue
            # Get all stocks with labels at this end_date
            day_labels = self.labels[self.labels["datetime"] == end_date]
            for _, row in day_labels.iterrows():
                window_index.append((end_date, row["stock_code"]))
        return window_index

    @staticmethod
    def _parse_variable(v: str) -> Tuple[str, str]:
        vstr = str(v)
        if "::" in vstr:
            comp, met = vstr.split("::", 1)
            return comp, met
        return "", vstr

    def _extract_calendar(self, datetimes):
        dt = datetimes.dt
        return (dt.day.values.astype(np.int64),
                dt.month.values.astype(np.int64),
                dt.dayofweek.values.astype(np.int64),
                (dt.year - self.base_year).values.astype(np.int64))

    def __len__(self):
        return len(self.window_index)

    def _get_window(self, end_date: pd.Timestamp, target_stock: str) -> Optional[Dict]:
        wdays = np.random.choice(self._window_options) if self.multi_window else self.window_days
        start_date = end_date - pd.Timedelta(days=wdays)
        mask = (self.data["datetime"] > start_date) & (self.data["datetime"] <= end_date)
        if not mask.any():
            return None
        # Get integer positions for the window slice
        idx = np.where(mask)[0]
        if len(idx) > self.max_seq_len:
            idx = idx[-self.max_seq_len:]
        L = len(idx)

        # Vectorized extraction: all columns as numpy arrays (no iterrows!)
        company_names = self.data["_company"].iloc[idx]
        metric_names = self.data["_metric"].iloc[idx]
        company_ids = company_names.map(self.company_to_id).fillna(0).astype(np.int64).values
        metric_ids = metric_names.map(self.metric_to_id).fillna(0).astype(np.int64).values
        source_ids = self.data["_source_id"].iloc[idx].values
        values = self.data["_value_num"].iloc[idx].values
        tsu = self.data["_tsu"].iloc[idx].values

        time_bins = np.zeros(L, dtype=np.int64)
        time_bins[tsu > 21] = 1; time_bins[tsu > 63] = 2; time_bins[tsu > 126] = 3

        day = self.data["_day"].iloc[idx].values
        month = self.data["_month"].iloc[idx].values
        dow = self.data["_dow"].iloc[idx].values
        year_off = self.data["_year_off"].iloc[idx].values

        # Target stock label
        target_date = end_date + pd.Timedelta(days=self.forecast_horizon)
        label_rows = self.labels[(self.labels["datetime"] == target_date) &
                                 (self.labels["stock_code"] == target_stock)]
        if len(label_rows) == 0:
            return None
        label = float(label_rows["label"].values[0])

        # Pad to max_seq_len
        pad_len = self.max_seq_len - L
        p = np.zeros(self.max_seq_len, dtype=bool); p[:L] = True
        return {
            "values": torch.tensor(np.pad(values, (0, pad_len))[:, None], dtype=torch.float32),
            "company_ids": torch.tensor(np.pad(company_ids, (0, pad_len)), dtype=torch.long),
            "metric_ids": torch.tensor(np.pad(metric_ids, (0, pad_len)), dtype=torch.long),
            "source_ids": torch.tensor(np.pad(source_ids, (0, pad_len)), dtype=torch.long),
            "time_bins": torch.tensor(np.pad(time_bins, (0, pad_len)), dtype=torch.long),
            "day": torch.tensor(np.pad(day, (0, pad_len)), dtype=torch.long),
            "month": torch.tensor(np.pad(month, (0, pad_len)), dtype=torch.long),
            "dow": torch.tensor(np.pad(dow, (0, pad_len)), dtype=torch.long),
            "year_offset": torch.tensor(np.pad(year_off, (0, pad_len)), dtype=torch.long),
            "time_since_update": torch.tensor(np.pad(tsu, (0, pad_len)), dtype=torch.float32),
            "mask": torch.tensor(p, dtype=torch.bool),
            "label": torch.tensor([label], dtype=torch.float32),
            "seq_len": L,
            "stock_code": target_stock,
        }

    def __getitem__(self, idx):
        end_date, stock_code = self.window_index[idx]
        end_date = pd.Timestamp(end_date)
        cache_file = None
        if self.cache_dir:
            cache_file = self.cache_dir / f"win_{end_date.strftime('%Y%m%d')}_{stock_code}.pt"
            if cache_file.exists():
                return torch.load(cache_file, weights_only=False)
        sample = self._get_window(end_date, stock_code)
        if self.cache_dir and sample is not None:
            torch.save(sample, cache_file)
        return sample

    def train_val_test_split_by_date(self, train_frac=0.80, val_frac=0.10):
        """Split by UNIQUE DATES (not stock-windows) to prevent leakage."""
        unique_dates = np.sort(pd.to_datetime(self.labels["datetime"].unique()))
        n = len(unique_dates)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))
        train_dates = set(unique_dates[:train_end])
        val_dates = set(unique_dates[train_end:val_end])
        test_dates = set(unique_dates[val_end:])

        train_idx = [i for i, (d, s) in enumerate(self.window_index) if d in train_dates]
        val_idx = [i for i, (d, s) in enumerate(self.window_index) if d in val_dates]
        test_idx = [i for i, (d, s) in enumerate(self.window_index) if d in test_dates]
        return train_idx, val_idx, test_idx


def collate_fn_stock(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    keys = ["values", "company_ids", "metric_ids", "source_ids", "time_bins",
            "day", "month", "dow", "year_offset", "time_since_update", "mask"]
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch])
    out["label"] = torch.stack([b["label"] for b in batch]).view(-1, 1)
    return out
