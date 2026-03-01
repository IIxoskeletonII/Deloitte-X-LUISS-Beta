"""
search_engine.py
----------------
Role: Semantic Search and Recommendation Engine. Loads all pre-computed
artifacts (FAISS index, product embeddings, metadata, summaries, entities)
and provides methods for:
  - Semantic search: encode a natural-language query and find the most
    relevant products by cosine similarity, with optional rating/category
    filters.
  - Recommendations: given a product, find similar products via FAISS
    nearest-neighbor lookup.
  - Product details: retrieve full product info including summary and
    extracted entities.

The engine over-fetches 3x candidates before filtering because FAISS does
not support metadata filtering natively.

Usage: Import and instantiate SearchEngine, or run standalone for a quick
       demo: python -m src.search_engine
"""

import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path


class SearchEngine:
    def __init__(self, artifacts_dir=None):
        if artifacts_dir is None:
            artifacts_dir = Path(__file__).resolve().parent / "io" / "output"
        else:
            artifacts_dir = Path(artifacts_dir)

        self.model = SentenceTransformer("all-mpnet-base-v2")

        self.index = faiss.read_index(str(artifacts_dir / "product_index.faiss"))
        self.product_ids = np.load(artifacts_dir / "product_ids.npy", allow_pickle=True)
        self.product_embeddings = np.load(artifacts_dir / "product_embeddings.npy")

        self.metadata = pd.read_parquet(artifacts_dir / "product_metadata.parquet")
        self.metadata_lookup = {
            row["parent_asin"]: row.to_dict()
            for _, row in self.metadata.iterrows()
        }

        summaries_path = artifacts_dir / "product_summaries.parquet"
        if summaries_path.exists():
            df_sum = pd.read_parquet(summaries_path)
            self.summaries = {
                row["parent_asin"]: row["summary"]
                for _, row in df_sum.iterrows()
            }
        else:
            self.summaries = {}

        entities_path = artifacts_dir / "product_entities.parquet"
        if entities_path.exists():
            df_ent = pd.read_parquet(entities_path)
            self.entities = {}
            for _, row in df_ent.iterrows():
                try:
                    self.entities[row["parent_asin"]] = json.loads(row["entities"])
                except (json.JSONDecodeError, TypeError):
                    self.entities[row["parent_asin"]] = {}
        else:
            self.entities = {}

        self.asin_to_idx = {
            asin: i for i, asin in enumerate(self.product_ids)
        }

    def search(self, query, top_k=10, min_rating=None, category=None):
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding = query_embedding.astype(np.float32)

        fetch_k = top_k * 3
        scores, indices = self.index.search(query_embedding, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.product_ids):
                continue
            asin = self.product_ids[idx]
            meta = self.metadata_lookup.get(asin, {})

            if min_rating is not None:
                avg_rating = meta.get("average_rating", 0)
                try:
                    if float(avg_rating) < min_rating:
                        continue
                except (ValueError, TypeError):
                    continue

            if category is not None and category != "All":
                prod_category = meta.get("main_category", "")
                if isinstance(prod_category, str) and category.lower() not in prod_category.lower():
                    continue

            result = {
                "parent_asin": asin,
                "score": float(score),
                "product_title": meta.get("product_title", ""),
                "average_rating": meta.get("average_rating", ""),
                "rating_number": meta.get("rating_number", ""),
                "store": meta.get("store", ""),
                "main_category": meta.get("main_category", ""),
                "summary": self.summaries.get(asin, ""),
                "entities": self.entities.get(asin, {}),
            }
            results.append(result)

            if len(results) >= top_k:
                break

        return results

    def recommend(self, parent_asin, top_k=5):
        idx = self.asin_to_idx.get(parent_asin)
        if idx is None:
            return []

        product_emb = self.product_embeddings[idx : idx + 1].astype(np.float32)
        scores, indices = self.index.search(product_emb, top_k + 1)

        results = []
        for score, found_idx in zip(scores[0], indices[0]):
            if found_idx < 0 or found_idx >= len(self.product_ids):
                continue
            asin = self.product_ids[found_idx]
            if asin == parent_asin:
                continue
            meta = self.metadata_lookup.get(asin, {})
            results.append({
                "parent_asin": asin,
                "score": float(score),
                "product_title": meta.get("product_title", ""),
                "average_rating": meta.get("average_rating", ""),
                "store": meta.get("store", ""),
                "summary": self.summaries.get(asin, ""),
            })
            if len(results) >= top_k:
                break

        return results

    def get_product_details(self, parent_asin):
        meta = self.metadata_lookup.get(parent_asin)
        if meta is None:
            return None

        return {
            **meta,
            "summary": self.summaries.get(parent_asin, ""),
            "entities": self.entities.get(parent_asin, {}),
        }

    def get_categories(self):
        if "main_category" in self.metadata.columns:
            cats = self.metadata["main_category"].dropna().unique().tolist()
            return sorted(set(str(c) for c in cats if c))
        return []


def main():
    print("[*] Loading Search Engine...")
    engine = SearchEngine()
    print(f"    -> {engine.index.ntotal} products indexed.")

    test_queries = [
        "low-priced skincare product",
        "organic shampoo for sensitive scalp",
        "vitamins for energy and focus",
    ]

    for query in test_queries:
        print(f"\n--- Query: '{query}' ---")
        results = engine.search(query, top_k=5)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] {r['product_title']}")
            if r["summary"]:
                print(f"     Summary: {r['summary'][:100]}...")
            if r["entities"]:
                ents = list(r["entities"].keys())[:5]
                print(f"     Entities: {', '.join(ents)}")


if __name__ == "__main__":
    main()
