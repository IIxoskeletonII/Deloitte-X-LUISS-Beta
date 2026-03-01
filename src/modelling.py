"""
modelling.py
------------
Role: Semantic Representation and Product Profiling. Builds product-level
embeddings by fusing metadata and review signals into a single vector per
product, then indexes them with FAISS for fast similarity search.

Pipeline:
  1. Encode product metadata (title, description, features, store) per product.
  2. Encode each review, then compute a weighted mean per product (weight =
     helpful_vote + 1) to give more influence to reviews the community found
     useful.
  3. Fuse metadata and review embeddings (0.3 metadata + 0.7 review) and
     L2-normalize so inner-product equals cosine similarity.
  4. Build a FAISS IndexFlatIP index and save all artifacts.

Outputs (saved to src/io/output/):
  - product_embeddings.npy   (N_products, 768)
  - product_index.faiss      FAISS index
  - product_ids.npy          parent_asin array mapping FAISS row -> product
  - product_metadata.parquet  deduplicated product catalog

Usage: Run from root via module mode: python -m src.modelling
"""

import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm
import sys


def clean_text_field(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return ""
        return " ".join(str(item) for item in value if item)
    if pd.isna(value):
        return ""
    return str(value)


def build_metadata_texts(df_products):
    texts = []
    for _, row in df_products.iterrows():
        parts = []
        for col in ["product_title", "description", "features", "store"]:
            val = clean_text_field(row.get(col, ""))
            if val:
                parts.append(val)
        texts.append(" | ".join(parts))
    return texts


def encode_in_batches(model, texts, batch_size=128, device="cpu"):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, batch_size=batch_size, show_progress_bar=False, device=device)
        all_embeddings.append(embs)
    return np.vstack(all_embeddings)


def main():
    SRC_DIR = Path(__file__).resolve().parent
    INPUT_FILE = SRC_DIR / "io" / "output" / "processed_data.parquet"
    OUTPUT_DIR = SRC_DIR / "io" / "output"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[*] Starting Product-Level Semantic Embedding Pipeline...")

    if not INPUT_FILE.exists():
        print(f"[!] Error: Could not find {INPUT_FILE}")
        print("    Did you run preprocessing.py first?")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    -> Compute Device: {device.upper()}")

    print("    -> Loading preprocessed dataset...")
    df = pd.read_parquet(INPUT_FILE)

    model_name = "all-mpnet-base-v2"
    print(f"    -> Initializing Model: {model_name}...")
    model = SentenceTransformer(model_name, device=device)
    embed_dim = model.get_sentence_embedding_dimension()
    print(f"       Embedding dimensions: {embed_dim}")

    meta_cols = ["product_title", "description", "features", "store",
                 "average_rating", "rating_number", "main_category"]
    agg_dict = {}
    for col in meta_cols:
        if col in df.columns:
            agg_dict[col] = "first"

    print("    -> Building deduplicated product catalog...")
    df_products = df.groupby("parent_asin").agg(agg_dict).reset_index()
    product_asins = df_products["parent_asin"].values
    n_products = len(df_products)
    print(f"       {n_products} unique products found.")

    print("    -> Stage 1: Encoding product metadata...")
    metadata_texts = build_metadata_texts(df_products)
    metadata_embeddings = encode_in_batches(model, metadata_texts, batch_size=128, device=device)

    print("    -> Stage 2: Encoding reviews and aggregating per product...")
    for col in ["review_title", "text"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text_field)
        else:
            df[col] = ""

    review_title_col = "review_title" if "review_title" in df.columns else ""
    text_col = "text" if "text" in df.columns else ""

    if review_title_col and text_col:
        df["_review_text"] = df[review_title_col] + " " + df[text_col]
    elif text_col:
        df["_review_text"] = df[text_col]
    else:
        df["_review_text"] = ""

    helpful_col = "helpful_vote" if "helpful_vote" in df.columns else None
    if helpful_col:
        df["_weight"] = df[helpful_col].fillna(0).astype(float) + 1.0
    else:
        df["_weight"] = 1.0

    asin_to_idx = {asin: i for i, asin in enumerate(product_asins)}
    review_embeddings_accum = np.zeros((n_products, embed_dim), dtype=np.float64)
    weight_accum = np.zeros(n_products, dtype=np.float64)

    batch_size = 128
    review_texts = df["_review_text"].tolist()
    review_asins = df["parent_asin"].tolist()
    review_weights = df["_weight"].tolist()

    for i in tqdm(range(0, len(review_texts), batch_size), desc="Review batches"):
        batch_texts = review_texts[i : i + batch_size]
        batch_asins = review_asins[i : i + batch_size]
        batch_weights = review_weights[i : i + batch_size]

        batch_embs = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False, device=device)

        for j, (asin, w) in enumerate(zip(batch_asins, batch_weights)):
            idx = asin_to_idx.get(asin)
            if idx is not None:
                review_embeddings_accum[idx] += batch_embs[j].astype(np.float64) * w
                weight_accum[idx] += w

        if device == "cuda" and i % (batch_size * 50) == 0:
            torch.cuda.empty_cache()

    mask = weight_accum > 0
    review_embeddings = np.zeros((n_products, embed_dim), dtype=np.float32)
    review_embeddings[mask] = (review_embeddings_accum[mask] / weight_accum[mask, np.newaxis]).astype(np.float32)

    del review_embeddings_accum, weight_accum, review_texts, review_asins, review_weights
    df.drop(columns=["_review_text", "_weight"], inplace=True)
    if device == "cuda":
        torch.cuda.empty_cache()

    print("    -> Stage 3: Fusing embeddings (0.3 metadata + 0.7 review)...")
    product_embeddings = 0.3 * metadata_embeddings + 0.7 * review_embeddings
    norms = np.linalg.norm(product_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    product_embeddings = (product_embeddings / norms).astype(np.float32)

    print("    -> Stage 4: Building FAISS index (IndexFlatIP)...")
    index = faiss.IndexFlatIP(embed_dim)
    index.add(product_embeddings)

    print("    -> Saving artifacts...")
    np.save(OUTPUT_DIR / "product_embeddings.npy", product_embeddings)
    np.save(OUTPUT_DIR / "product_ids.npy", product_asins)
    faiss.write_index(index, str(OUTPUT_DIR / "product_index.faiss"))
    df_products.to_parquet(OUTPUT_DIR / "product_metadata.parquet", index=False)

    print(f"[*] Product-Level Embedding Complete.")
    print(f"    Products indexed: {n_products}")
    print(f"    Embedding dim: {product_embeddings.shape[1]}")
    print(f"    FAISS index size: {index.ntotal}")


if __name__ == "__main__":
    main()
