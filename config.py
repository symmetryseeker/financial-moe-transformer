"""
Global configuration for Financial MoE Transformer.
Tuned for GTX 1060 6GB VRAM / i7-8750H / 8GB RAM.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# External data store (D drive — large capacity, holds raw downloaded data)
DATA_EXTERNAL = Path("D:/financial_data")

# Use D drive for all large data (8GB RAM constraint)
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = Path("D:/financial_data/processed")  # on D drive
DATA_EMBEDDINGS = ROOT / "data" / "embeddings"
MODEL_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"

for d in [DATA_RAW, DATA_PROCESSED, DATA_EMBEDDINGS, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Ensure external data directory exists
DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)

# ─── Hardware ──────────────────────────────────────────────────────────────────
# GTX 1060 6GB: FP16 throughput is 1/64 of FP32 ― use FP32.
# Update NVIDIA driver to >=525 to get CUDA 11.8+ for PyTorch 2.1+.
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
USE_AMP = False  # no mixed-precision: GP106 has no tensor cores

# ─── Model Architecture ────────────────────────────────────────────────────────
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.2
ATTN_DROPOUT = 0.2
ACTIVATION = "gelu"

# Linear attention + chunking
ATTN_TYPE = "cosformer"
ATTN_CHUNK_SIZE = 2048
MAX_SEQ_LEN = 8192

# ─── Hierarchical Encoder (v3) ─────────────────────────────────────────────────
# Multi-scale: 4 time scales for temporal pyramid
TIME_SCALES = [21, 63, 126, 252]   # 1mo, 3mo, 6mo, 12mo trading days
SCALE_DIMS = [32, 32, 32, 32]      # dimension per scale (sum = d_model)

# Stock-level encoder: compresses each stock's time series
STOCK_ENCODER_TYPE = "conv"         # "conv" | "transformer" | "pool"
STOCK_CONV_KERNELS = [5, 3]         # kernel sizes for 1D conv stack
STOCK_ENCODER_LAYERS = 2            # conv layers per stock

# Max stocks to encode individually (top N by volume)
MAX_STOCKS = 50
# Key stock metrics to use
STOCK_METRICS = ["close", "volume", "peTTM", "pbMRQ", "turn"]

# Source grouping: which sources get per-group encoding
GROUPED_SOURCES = ["market"]        # per-stock grouping
POOLED_SOURCES = ["macro", "financial", "alternative", "sentiment"]  # direct pool

# ─── MoE ───────────────────────────────────────────────────────────────────────
NUM_EXPERTS = 6
EXPERT_HIDDEN = 64
EXPERT_LAYERS = 3                   # 128→64→32→1
TOP_K = 1                        # back to Top-1 (simpler, works better)
LOAD_BALANCE_COEF = 0.01          # keep LB gentle — higher values overpowers MSE

# ─── Embeddings ────────────────────────────────────────────────────────────────
# Dual embedding (company + metric) — solves 92K vocab explosion
N_COMPANIES = 6000
N_METRICS = 800
COMPANY_DIM = 32
METRIC_DIM = 96
VALUE_DIM = 1

TIME_BUCKETS = [0, 1, 2, 3, 4, 5, 10, 20, 60, 120]
NUM_TIME_BUCKETS = len(TIME_BUCKETS) + 1

DAY_EMBED_DIM = 8
MONTH_EMBED_DIM = 8
DOW_EMBED_DIM = 8
TIME_BUCKET_EMBED_DIM = 8

# ─── Data ──────────────────────────────────────────────────────────────────────
WINDOW_TRADING_DAYS = 252
FORECAST_HORIZON = 63  # 3-month ahead (higher SNR than 1-month)
ZSCORE_ROLLING_YEARS = 5
ZSCORE_MIN_PERIODS = 252
MIN_SEQ_LEN = 64
TOKEN_MASK_PROB = 0.05

# Walk-forward CV
N_CV_FOLDS = 5                     # number of time-series folds

# Train/val/test splits (for simple split mode)
TRAIN_FRAC = 0.6
VAL_FRAC = 0.25                   # 0.15→0.25: more stable IC estimate
TEST_FRAC = 0.15

REQUIRED_COLS = ["datetime", "source", "variable", "value"]

# ─── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
EPOCHS = 20
EARLY_STOP_PATIENCE = 5
LR = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 500
MAX_GRAD_NORM = 1.0
NUM_WORKERS = 2

LR_SCHEDULER = "cosine"
LR_MIN_FACTOR = 0.01

# ─── Loss Coefficients ─────────────────────────────────────────────────────────
LOSS_MSE_COEF = 1.0
LOSS_LOAD_BALANCE_COEF = 0.01          # Phase 2 baseline for isolation
LOSS_RANK_COEF = 0.5               # 0.3→0.5: stronger ranking focus
LOSS_DIRECTION_COEF = 0.0           # 0.1→0: disabled (labels balanced 49/51)
LOSS_Z_COEF = 0.0                  # disabled for isolation test
LOSS_CONCEPT_REG_COEF = 0.001

# ─── Text Encoding (offline) ───────────────────────────────────────────────────
TEXT_ENCODER_MODEL = "BAAI/bge-small-zh-v1.5"
TEXT_EMBED_DIM = 384             # original BGE output dim
TEXT_EMBED_REDUCED = 128         # PCA target dim

# LLM state generation
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LLM_LOAD_IN_4BIT = True
LLM_STATE_DIMS = ["risk_appetite", "liquidity", "growth_expectation", "policy_stance"]

# ─── Evaluation ────────────────────────────────────────────────────────────────
TEST_YEARS = 3                   # most recent N years as test set
BACKTEST_STEP_MONTHS = 1         # rebalance monthly

# Concept similarity (for optional attention regularisation)
CONCEPT_VARIABLES: List[str] = field(default_factory=lambda: [
    "CPI同比", "PPI同比", "PMI", "M2同比", "社会融资规模",
    "工业增加值", "固定资产投资", "社会消费品零售总额",
    "出口同比", "进口同比", "美元兑人民币", "沪深300市盈率",
    "融资余额", "北向资金", "两融余额",
])


@dataclass
class ModelConfig:
    d_model: int = D_MODEL
    nhead: int = NHEAD
    num_layers: int = NUM_LAYERS
    dim_feedforward: int = DIM_FEEDFORWARD
    dropout: float = DROPOUT
    attn_dropout: float = ATTN_DROPOUT
    activation: str = ACTIVATION
    attn_type: str = ATTN_TYPE
    max_seq_len: int = MAX_SEQ_LEN
    attn_chunk_size: int = ATTN_CHUNK_SIZE

    # Hierarchical
    time_scales: list = field(default_factory=lambda: TIME_SCALES)
    scale_dims: list = field(default_factory=lambda: SCALE_DIMS)
    stock_encoder_type: str = STOCK_ENCODER_TYPE
    max_stocks: int = MAX_STOCKS
    stock_metrics: list = field(default_factory=lambda: STOCK_METRICS)

    # MoE
    num_experts: int = NUM_EXPERTS
    expert_hidden: int = EXPERT_HIDDEN
    expert_layers: int = EXPERT_LAYERS
    top_k: int = TOP_K

    # Embeddings (dual: company + metric)
    n_companies: int = N_COMPANIES
    n_metrics: int = N_METRICS
    company_dim: int = COMPANY_DIM
    metric_dim: int = METRIC_DIM
    day_embed_dim: int = DAY_EMBED_DIM
    month_embed_dim: int = MONTH_EMBED_DIM
    dow_embed_dim: int = DOW_EMBED_DIM
    time_bucket_embed_dim: int = TIME_BUCKET_EMBED_DIM


@dataclass
class TrainConfig:
    batch_size: int = BATCH_SIZE
    gradient_accumulation: int = GRADIENT_ACCUMULATION
    epochs: int = EPOCHS
    early_stop_patience: int = EARLY_STOP_PATIENCE
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    warmup_steps: int = WARMUP_STEPS
    max_grad_norm: float = MAX_GRAD_NORM
    use_amp: bool = USE_AMP
    num_workers: int = NUM_WORKERS
    n_cv_folds: int = N_CV_FOLDS

    # Loss coefficients
    mse_coef: float = LOSS_MSE_COEF
    load_balance_coef: float = LOSS_LOAD_BALANCE_COEF
    rank_coef: float = LOSS_RANK_COEF
    direction_coef: float = LOSS_DIRECTION_COEF
    concept_reg_coef: float = LOSS_CONCEPT_REG_COEF

    # Data
    token_mask_prob: float = TOKEN_MASK_PROB


# Singleton convenience
model_cfg = ModelConfig()
train_cfg = TrainConfig()
