"""
Build concept-similarity matrix for optional attention regularisation.

Encodes human-readable descriptions of financial variables (e.g. "CPI同比",
"PMI") with BGE-small, computes cosine similarity, and saves the matrix.
During training, the model can be regularised to produce attention patterns
that respect these pairwise similarities.

Usage:
    python utils/concept_similarity.py --output data/embeddings/concept_sim.npy
"""

import argparse
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Default concept list (Chinese financial variables)
DEFAULT_CONCEPTS = [
    # Macro
    "CPI同比增速",
    "PPI生产者价格指数",
    "PMI采购经理指数",
    "M2广义货币供应量同比",
    "社会融资规模增量",
    "工业增加值同比",
    "固定资产投资完成额同比",
    "社会消费品零售总额同比",
    "出口金额同比",
    "进口金额同比",
    "CPI食品项同比",
    "CPI非食品项同比",
    "GDP同比增速",
    "失业率",

    # Monetary / credit
    "7天逆回购利率",
    "1年期LPR贷款市场报价利率",
    "5年期LPR",
    "10年期国债收益率",
    "信用利差AAA级",
    "美元兑人民币汇率",

    # Market / trading
    "沪深300指数",
    "沪深300市盈率",
    "沪深300波动率",
    "融资余额",
    "北向资金净流入",
    "两融余额占流通市值比",
    "换手率",
    "市场成交额",

    # Sentiment
    "新闻舆情正面向得分",
    "分析师一致预期EPS",
    "分析师评级上调比例",
    "研报预测准确度",

    # International
    "美联储联邦基金利率",
    "美国10年期国债收益率",
    "VIX恐慌指数",
    "MSCI新兴市场指数",
]


def compute_similarity(concepts: list[str],
                       model_name: str = "BAAI/bge-small-zh-v1.5",
                       device: str = "cpu") -> np.ndarray:
    """Compute cosine similarity matrix for a list of concept descriptions."""
    print(f"Encoding {len(concepts)} concepts with {model_name} ...")
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(concepts, normalize_embeddings=True, show_progress_bar=False)
    sim = embeddings @ embeddings.T  # cosine sim since normalised
    return sim


def main():
    parser = argparse.ArgumentParser(description="Concept similarity matrix")
    parser.add_argument("--output", type=str,
                        default="data/embeddings/concept_sim.npy")
    parser.add_argument("--model-name", type=str,
                        default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--concepts", type=str, nargs="*",
                        help="Override default concept list")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    concepts = args.concepts if args.concepts else DEFAULT_CONCEPTS
    sim = compute_similarity(concepts, args.model_name, args.device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, sim)

    # Also save concept labels
    labels_path = output_path.with_suffix(".labels.txt")
    labels_path.write_text("\n".join(concepts), encoding="utf-8")

    print(f"Saved {sim.shape} similarity matrix to {args.output}")
    print(f"Saved concept labels to {labels_path}")


if __name__ == "__main__":
    main()
