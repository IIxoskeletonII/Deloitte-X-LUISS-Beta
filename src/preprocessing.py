"""
preprocessing.py
----------------
Role: Core Data Pipeline. Loads extracted JSONL files, cleans metadata, 
filters low-engagement reviews, merges datasets, and outputs an optimized 
Parquet file for the modelling phase.

Usage: Run from root via module mode: python -m src.preprocessing
"""

import pandas as pd
from pathlib import Path
import sys

def main():
    # 1. Setup Paths (Adhering to Project Rules - no hardcoded paths)
    SRC_DIR = Path(__file__).resolve().parent
    EXTRACTED_DIR = SRC_DIR / "io" / "input" / "extracted"
    OUTPUT_DIR = SRC_DIR / "io" / "output"
    
    # Ensure output directory exists (tracked via .gitkeep, but contents ignored)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    REVIEWS_PATH = EXTRACTED_DIR / "Health_and_Personal_Care.jsonl"
    META_PATH = EXTRACTED_DIR / "meta_Health_and_Personal_Care.jsonl"
    OUTPUT_FILE = OUTPUT_DIR / "processed_data.parquet"

    print("[*] Starting Preprocessing Pipeline...")

    # 2. Ingestion
    if not REVIEWS_PATH.exists() or not META_PATH.exists():
        print(f"[!] Error: Missing input files in {EXTRACTED_DIR}")
        print("    Did you run data_extraction.py first?")
        sys.exit(1)

    print("    -> Loading datasets into memory...")
    # Using lines=True for JSON Lines format
    df_reviews = pd.read_json(REVIEWS_PATH, lines=True)
    df_meta = pd.read_json(META_PATH, lines=True)

    # 3. Metadata Cleaning
    print("    -> Cleaning Metadata...")
    # Rule: Clean missing values in Title and Store (Brand)
    df_meta.dropna(subset=['title', 'store'], inplace=True)
    
    # Rule: Cannot use Price or Bought_Together heavily (Drop them)
    cols_to_drop = ['price', 'bought_together']
    df_meta.drop(columns=[c for c in cols_to_drop if c in df_meta.columns], inplace=True)

    # 4. Review Filtering
    print("    -> Filtering Reviews (Requires >= 5 reviews per product)...")
    # Rule: Filter out products with low engagement (<5 reviews)
    review_counts = df_reviews['parent_asin'].value_counts()
    valid_asins = review_counts[review_counts >= 5].index
    df_reviews_filtered = df_reviews[df_reviews['parent_asin'].isin(valid_asins)]

    # 5. Dataset Merging
    print("    -> Merging Datasets on 'parent_asin'...")
    # Inner join ensures we only keep products that exist in BOTH cleaned datasets
    df_merged = pd.merge(df_reviews_filtered, df_meta, on='parent_asin', how='inner', suffixes=('_rev', '_meta'))

    # 6. Optimized Storage
    print(f"    -> Saving optimized data to {OUTPUT_FILE.name}...")
    df_merged.to_parquet(OUTPUT_FILE, index=False)
    
    print(f"[*] Preprocessing Complete. Final shape: {df_merged.shape}")

if __name__ == "__main__":
    main()