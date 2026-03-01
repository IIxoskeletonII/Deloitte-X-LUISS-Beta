"""
preprocessing.py
----------------
Role: Core Data Pipeline. Loads extracted JSONL files, cleans metadata,
filters low-engagement reviews, merges datasets, renames columns for
clarity, drops low-value fields, and outputs an optimized Parquet file
for downstream NLP tasks (embedding, summarization, entity recognition).

Usage: Run from root via module mode: python -m src.preprocessing
"""

import pandas as pd
from pathlib import Path
import sys


def main():
    SRC_DIR = Path(__file__).resolve().parent
    EXTRACTED_DIR = SRC_DIR / "io" / "input" / "extracted"
    OUTPUT_DIR = SRC_DIR / "io" / "output"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    REVIEWS_PATH = EXTRACTED_DIR / "Health_and_Personal_Care.jsonl"
    META_PATH = EXTRACTED_DIR / "meta_Health_and_Personal_Care.jsonl"
    OUTPUT_FILE = OUTPUT_DIR / "processed_data.parquet"

    print("[*] Starting Preprocessing Pipeline...")

    if not REVIEWS_PATH.exists() or not META_PATH.exists():
        print(f"[!] Error: Missing input files in {EXTRACTED_DIR}")
        print("    Did you run data_extraction.py first?")
        sys.exit(1)

    print("    -> Loading datasets into memory...")
    df_reviews = pd.read_json(REVIEWS_PATH, lines=True)
    df_meta = pd.read_json(META_PATH, lines=True)

    print("    -> Cleaning Metadata...")
    df_meta.dropna(subset=["title", "store"], inplace=True)

    cols_to_drop_meta = ["price", "bought_together"]
    df_meta.drop(
        columns=[c for c in cols_to_drop_meta if c in df_meta.columns],
        inplace=True,
    )

    print("    -> Filtering Reviews (Requires >= 5 reviews per product)...")
    review_counts = df_reviews["parent_asin"].value_counts()
    valid_asins = review_counts[review_counts >= 5].index
    df_reviews_filtered = df_reviews[df_reviews["parent_asin"].isin(valid_asins)]

    print("    -> Merging Datasets on 'parent_asin'...")
    df_merged = pd.merge(
        df_reviews_filtered,
        df_meta,
        on="parent_asin",
        how="inner",
        suffixes=("_rev", "_meta"),
    )

    print("    -> Renaming and cleaning columns...")
    rename_map = {}
    if "title_meta" in df_merged.columns:
        rename_map["title_meta"] = "product_title"
    if "title_rev" in df_merged.columns:
        rename_map["title_rev"] = "review_title"
    if rename_map:
        df_merged.rename(columns=rename_map, inplace=True)

    cols_to_drop = ["images_rev", "images_meta", "videos", "details"]
    df_merged.drop(
        columns=[c for c in cols_to_drop if c in df_merged.columns],
        inplace=True,
    )

    print(f"    -> Saving optimized data to {OUTPUT_FILE.name}...")
    df_merged.to_parquet(OUTPUT_FILE, index=False)

    print(f"[*] Preprocessing Complete. Final shape: {df_merged.shape}")
    print(f"    Columns: {list(df_merged.columns)}")


if __name__ == "__main__":
    main()
