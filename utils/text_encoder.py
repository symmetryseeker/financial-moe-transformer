"""
Offline text encoding with BAAI/bge-small-zh-v1.5.

Encodes unstructured text columns (news headlines, report summaries, etc.)
into 384-dim embeddings, then PCA-reduces to 128 dims for the model.

Usage:
    python utils/text_encoder.py --input data/raw/news.csv \
        --text-col headline --date-col date \
        --output data/embeddings/news_embeds.parquet
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import torch


def encode_texts(model: SentenceTransformer,
                 texts: list[str],
                 batch_size: int = 64,
                 device: str = "cpu") -> np.ndarray:
    """Encode a list of texts → (N, 384) embeddings."""
    # sentence-transformers handles batching internally when passed a list
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        device=device,
        normalize_embeddings=False,  # PCA works better on un-normalised vectors
    )
    return embeddings  # (N, 384)


def fit_pca(embeddings: np.ndarray, n_components: int = 128) -> PCA:
    """Fit PCA on embeddings."""
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(embeddings)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA {n_components} dims explains {explained:.2%} variance")
    return pca


def main():
    parser = argparse.ArgumentParser(description="Offline text encoding")
    parser.add_argument("--input", type=str, required=True, help="CSV/Parquet with text")
    parser.add_argument("--text-col", type=str, required=True, help="Column with text")
    parser.add_argument("--date-col", type=str, default="date", help="Date column")
    parser.add_argument("--output", type=str, required=True, help="Output parquet path")
    parser.add_argument("--model-name", type=str,
                        default="BAAI/bge-small-zh-v1.5",
                        help="Sentence-transformer model")
    parser.add_argument("--fit-pca-on", type=str, default=None,
                        help="Optional: fit PCA on this file, apply to --input")
    parser.add_argument("--pca-path", type=str,
                        default="data/embeddings/pca_128.npy",
                        help="Where to save/load PCA components")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Load data
    input_path = Path(args.input)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    texts = df[args.text_col].fillna("").astype(str).tolist()
    print(f"Loaded {len(texts)} texts from {args.input}")

    # Load model
    print(f"Loading {args.model_name} ...")
    model = SentenceTransformer(args.model_name, device=args.device)

    # Encode
    embeddings = encode_texts(model, texts, args.batch_size, args.device)
    print(f"Embeddings shape: {embeddings.shape}")

    # PCA
    pca_path = Path(args.pca_path)
    if args.fit_pca_on:
        # Fit PCA on a different (larger) dataset
        fit_path = Path(args.fit_pca_on)
        fit_df = pd.read_csv(fit_path) if fit_path.suffix == ".csv" else pd.read_parquet(fit_path)
        fit_texts = fit_df[args.text_col].fillna("").astype(str).tolist()
        fit_embeddings = encode_texts(model, fit_texts, args.batch_size, args.device)
        pca = fit_pca(fit_embeddings, n_components=128)
        pca_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(pca_path, pca.components_)
        # Also save mean
        np.save(str(pca_path).replace(".npy", "_mean.npy"), pca.mean_)
    elif pca_path.exists():
        # Load existing PCA
        components = np.load(pca_path)
        mean_path = str(pca_path).replace(".npy", "_mean.npy")
        mean = np.load(mean_path) if Path(mean_path).exists() else np.zeros(384)
        pca = PCA(n_components=128)
        pca.components_ = components
        pca.mean_ = mean
        print(f"Loaded PCA from {pca_path}")
    else:
        # Fit PCA on current data
        pca = fit_pca(embeddings, n_components=128)
        pca_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(pca_path, pca.components_)
        np.save(str(pca_path).replace(".npy", "_mean.npy"), pca.mean_)

    # Transform
    reduced = pca.transform(embeddings)  # (N, 128)
    print(f"Reduced shape: {reduced.shape}")

    # Build output DataFrame: one row per text, columns = embed_0..embed_127 + date
    embed_cols = [f"text_embed_{i}" for i in range(reduced.shape[1])]
    out = pd.DataFrame(reduced, columns=embed_cols)
    out[args.date_col] = df[args.date_col].values

    # Melt to long format for the data-points table
    long = out.melt(id_vars=[args.date_col], var_name="variable", value_name="value")
    long["source"] = "text_embed"
    long = long.rename(columns={args.date_col: "datetime"})
    long["datetime"] = pd.to_datetime(long["datetime"])

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(output_path, index=False)
    print(f"Saved {len(long)} rows to {args.output}")


if __name__ == "__main__":
    main()
