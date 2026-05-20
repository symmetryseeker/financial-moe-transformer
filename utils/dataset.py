"""
Sliding-window dataset for financial time-series tokens.

Each sample is a window of WINDOW_TRADING_DAYS calendar days of data points,
with a label from FORECAST_HORIZON trading days after the window end.

The sequence length varies per window (different number of data points arrive
each day).  We pad to MAX_SEQ_LEN and provide an attention mask.

Returns a dict with:
    values:            (L, 1)     z-scored values
    var_ids:           (L,)       variable identity indices
    day, month, dow:   (L,)       calendar features
    year_offset:       (L,)       year - base_year
    time_since_update: (L,)       days since last observation
    mask:              (L,)       bool (True = real token)
    label:             (1,)       scalar target
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Tuple, Optional
import pickle


class SlidingWindowDataset(Dataset):
    """
    Sliding window over a long-format data-points table.

    Each window is defined by an end-date.  All data points with
    datetime ∈ (end_date - window_days, end_date] are gathered,
    sorted by datetime, and returned as a padded sequence.
    """

    def __init__(self,
                 data_path: str = "D:/financial_data/processed/data_points.parquet",
                 labels_path: str = "D:/financial_data/processed/labels.parquet",
                 window_days: int = 365,
                 forecast_horizon: int = 63,
                 max_seq_len: int = 8192,
                 base_year: int = 2015,
                 use_cache: bool = True,
                 cache_dir: str = "data/processed/cache",
                 multi_window: bool = True):        # Phase 2: random window length
        self.window_days = window_days
        self.forecast_horizon = forecast_horizon
        self.max_seq_len = max_seq_len
        self.base_year = base_year
        self.multi_window = multi_window
        self._window_options = [126, 252, 365, 504]  # 6mo, 1yr, 1.5yr, 2yr
        self.cache_dir = Path(cache_dir) if use_cache else None
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        data_path = Path(data_path)
        labels_path = Path(labels_path)

        print(f"Loading data points from {data_path} ...")
        self.data = pd.read_parquet(data_path)
        self.data["datetime"] = pd.to_datetime(self.data["datetime"])
        self.data = self.data.sort_values("datetime").reset_index(drop=True)

        print(f"Loading labels from {labels_path} ...")
        self.labels = pd.read_parquet(labels_path)
        self.labels["datetime"] = pd.to_datetime(self.labels["datetime"])
        self.labels = self.labels.sort_values("datetime").reset_index(drop=True)

        # Build dual vocab: company_id + metric_id (solves 92K vocab explosion)
        self.company_to_id, self.metric_to_id = self._build_dual_vocab()
        self.n_companies = len(self.company_to_id)
        self.n_metrics = len(self.metric_to_id)
        # Legacy compat
        self.var_to_id = self.company_to_id  # for train.py vocab_size check

        # Build valid window end dates
        self.window_dates = self._build_windows()
        print(f"Built {len(self.window_dates)} windows")

        # Cache (already set above)
        self.use_cache = use_cache

    def _build_dual_vocab(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Build two small vocabularies instead of one huge one.

        Variable names like 'sh_600519::close' are split:
          company='sh_600519', metric='close'
        Variables without '::' (e.g., macro data) get company=0 (shared).

        Vocab is persisted to disk and reused across pipeline runs to ensure
        deterministic ordering — critical for checkpoint compatibility.
        """
        import json
        vocab_dir = self.cache_dir or Path("data/processed/cache")
        vocab_dir.mkdir(parents=True, exist_ok=True)
        c_path = vocab_dir / "company_vocab.json"
        m_path = vocab_dir / "metric_vocab.json"

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

        print(f"Dual vocab: {len(company_to_id):,} companies × {len(metric_to_id):,} metrics "
              f"(vs {len(company_to_id) * len(metric_to_id):,} naive)")
        return company_to_id, metric_to_id

    @staticmethod
    def _parse_variable(v: str) -> Tuple[str, str]:
        """Split 'company::metric' into (company, metric)."""
        vstr = str(v)
        if "::" in vstr:
            comp, met = vstr.split("::", 1)
            return comp, met
        return "", vstr

    def _build_windows(self) -> np.ndarray:
        """
        Generate window end dates.

        A window at end_date t includes all data points from (t - window_days, t].
        The label is the label at t + forecast_horizon.

        We need at least window_days of data before the first window, and
        forecast_horizon of labels after the last window.
        """
        all_dates = pd.to_datetime(self.data["datetime"].unique())
        all_dates = np.sort(all_dates)
        label_dates = pd.to_datetime(self.labels["datetime"].unique())

        # Trading-day-based window (252 trading days ≈ 365 calendar days)
        # For simplicity, use calendar-day windows
        min_date = all_dates[0] + pd.Timedelta(days=self.window_days)
        max_date = all_dates[-1] - pd.Timedelta(days=self.forecast_horizon)

        # Find label dates that are in range and have a valid label
        valid_labels = label_dates[(label_dates >= min_date) & (label_dates <= max_date)]

        return np.sort(valid_labels)

    def _date_features(self, datetimes: pd.Series) -> Tuple[np.ndarray, ...]:
        """Extract calendar features from a datetime Series."""
        dt = datetimes.dt
        day = dt.day.values.astype(np.int64)
        month = dt.month.values.astype(np.int64)
        dow = dt.dayofweek.values.astype(np.int64)
        year_offset = (dt.year - self.base_year).values.astype(np.int64)
        return day, month, dow, year_offset

    def _get_window(self, end_date: pd.Timestamp) -> Dict[str, torch.Tensor]:
        """Build a single window as a dict of tensors."""
        # Phase 2: random window length for data augmentation
        wdays = np.random.choice(self._window_options) if self.multi_window else self.window_days
        start_date = end_date - pd.Timedelta(days=wdays)

        # Filter data points in window
        mask = (self.data["datetime"] > start_date) & (self.data["datetime"] <= end_date)
        window = self.data.loc[mask].sort_values("datetime")

        L = len(window)
        if L == 0:
            return None

        # Truncate if too long
        if L > self.max_seq_len:
            window = window.iloc[-self.max_seq_len:]
            L = self.max_seq_len

        # Dual IDs: company + metric
        company_ids = np.zeros(L, dtype=np.int64)
        metric_ids = np.zeros(L, dtype=np.int64)
        for i, (_, row) in enumerate(window.iterrows()):
            comp, met = self._parse_variable(row["variable"])
            company_ids[i] = self.company_to_id.get(comp, 0)
            metric_ids[i] = self.metric_to_id.get(met, 0)

        # Values
        values = window["value"].fillna(0.0).values.astype(np.float32)

        # Calendar features
        dts = pd.to_datetime(window["datetime"])
        day, month, dow, year_offset = self._date_features(dts)

        # Time since update
        tsu = window["time_since_update"].fillna(0.0).values.astype(np.float32)

        # Label: find the label at end_date + forecast_horizon
        target_date = end_date + pd.Timedelta(days=self.forecast_horizon)
        label_rows = self.labels[self.labels["datetime"] == target_date]
        if len(label_rows) == 0:
            return None
        label = label_rows["label"].values[0].astype(np.float32)

        # Source IDs: 0=market, 1=macro, 2=financial, 3=alternative, 4=sentiment
        source_map = {"market": 0, "macro": 1, "financial": 2, "alternative": 3,
                      "sentiment": 4, "carbon": 3, "text": 5}
        source_ids = np.array([source_map.get(s, 0) for s in window["source"]], dtype=np.int64)

        # Time bins: 0=recent(0-21d), 1=mid(22-63d), 2=far(64-126d), 3=distant(127+)
        time_bins = np.zeros(L, dtype=np.int64)
        time_bins[tsu > 21] = 1
        time_bins[tsu > 63] = 2
        time_bins[tsu > 126] = 3

        # Padding
        pad_len = self.max_seq_len - L

        return {
            "values": torch.from_numpy(
                np.pad(values.reshape(-1, 1), ((0, pad_len), (0, 0)), constant_values=0)
            ),
            "company_ids": torch.from_numpy(
                np.pad(company_ids, (0, pad_len), constant_values=0)
            ),
            "metric_ids": torch.from_numpy(
                np.pad(metric_ids, (0, pad_len), constant_values=0)
            ),
            "source_ids": torch.from_numpy(
                np.pad(source_ids, (0, pad_len), constant_values=0)
            ),
            "time_bins": torch.from_numpy(
                np.pad(time_bins, (0, pad_len), constant_values=0)
            ),
            "day": torch.from_numpy(
                np.pad(day, (0, pad_len), constant_values=0)
            ),
            "month": torch.from_numpy(
                np.pad(month, (0, pad_len), constant_values=0)
            ),
            "dow": torch.from_numpy(
                np.pad(dow, (0, pad_len), constant_values=0)
            ),
            "year_offset": torch.from_numpy(
                np.pad(year_offset, (0, pad_len), constant_values=0)
            ),
            "time_since_update": torch.from_numpy(
                np.pad(tsu, (0, pad_len), constant_values=0)
            ),
            "mask": torch.from_numpy(
                np.concatenate([np.ones(L, dtype=bool), np.zeros(pad_len, dtype=bool)])
            ),
            "label": torch.tensor(label),
            "seq_len": L,
        }

    def __len__(self) -> int:
        return len(self.window_dates)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        end_date = pd.Timestamp(self.window_dates[idx])

        # Check cache
        if self.use_cache:
            cache_file = self.cache_dir / f"win_{end_date.strftime('%Y%m%d')}.pt"
            if cache_file.exists():
                return torch.load(cache_file, weights_only=False)

        sample = self._get_window(end_date)

        if self.use_cache and sample is not None:
            torch.save(sample, self.cache_dir / f"win_{end_date.strftime('%Y%m%d')}.pt")

        return sample

    def train_val_test_split(self,
                             train_frac: float = 0.7,
                             val_frac: float = 0.15,
                             n_folds: int = 5,
                             fold: int = -1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split window dates chronologically.

        If fold >= 0: walk-forward CV with n_folds.
          fold 0 = earliest test period, fold n_folds-1 = latest.
          Each fold gets its own contiguous train/val/test blocks.
        If fold < 0: legacy single split with train_frac / val_frac.
        """
        n = len(self.window_dates)

        if fold >= 0:
            fold_size = max(1, n // (n_folds + 2))
            test_start = n - (n_folds - fold) * fold_size
            test_end = min(n, test_start + fold_size)
            val_start = max(0, test_start - fold_size)
            train_end = val_start
            return (
                self.window_dates[:train_end],
                self.window_dates[val_start:test_start],
                self.window_dates[test_start:test_end],
            )
        else:
            train_end = int(n * train_frac)
            val_end = int(n * (train_frac + val_frac))
            return (
                self.window_dates[:train_end],
                self.window_dates[train_end:val_end],
                self.window_dates[val_end:],
            )


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """Custom collate: stack samples of varying lengths (already padded to MAX_SEQ_LEN)."""
    # Filter None samples
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}

    return {
        "values": torch.stack([b["values"] for b in batch]),
        "company_ids": torch.stack([b["company_ids"] for b in batch]),
        "metric_ids": torch.stack([b["metric_ids"] for b in batch]),
        "source_ids": torch.stack([b["source_ids"] for b in batch]),
        "time_bins": torch.stack([b["time_bins"] for b in batch]),
        "day": torch.stack([b["day"] for b in batch]),
        "month": torch.stack([b["month"] for b in batch]),
        "dow": torch.stack([b["dow"] for b in batch]),
        "year_offset": torch.stack([b["year_offset"] for b in batch]),
        "time_since_update": torch.stack([b["time_since_update"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }
