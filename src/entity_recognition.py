"""
entity_recognition.py
---------------------
Role: Named Entity Recognition for product profiling. Extracts entities
(brands, organizations, products, locations, persons) from the most
helpful reviews for each product. Entities are aggregated per product
and filtered to keep only those mentioned at least twice, reducing noise.

Model: dslim/bert-base-NER (token classification).
Processes the top 20 most helpful reviews per product.

Output: src/io/output/product_entities.parquet
  Columns: parent_asin, entities (JSON string of entity -> count mapping)

Usage: Run from root via module mode: python -m src.entity_recognition
"""

import json
import pandas as pd
import torch
from transformers import pipeline
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import sys


TOP_REVIEWS_PER_PRODUCT = 20
MIN_ENTITY_COUNT = 2


def main():
    SRC_DIR = Path(__file__).resolve().parent
    INPUT_FILE = SRC_DIR / "io" / "output" / "processed_data.parquet"
    OUTPUT_DIR = SRC_DIR / "io" / "output"
    OUTPUT_FILE = OUTPUT_DIR / "product_entities.parquet"

    print("[*] Starting Named Entity Recognition Pipeline...")

    if not INPUT_FILE.exists():
        print(f"[!] Error: Could not find {INPUT_FILE}")
        print("    Did you run preprocessing.py first?")
        sys.exit(1)

    device_id = 0 if torch.cuda.is_available() else -1
    device_name = "CUDA" if device_id == 0 else "CPU"
    print(f"    -> Compute Device: {device_name}")

    print("    -> Loading preprocessed dataset...")
    df = pd.read_parquet(INPUT_FILE)

    helpful_col = "helpful_vote" if "helpful_vote" in df.columns else None
    if helpful_col:
        df[helpful_col] = df[helpful_col].fillna(0)

    print("    -> Initializing dslim/bert-base-NER...")
    ner = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        device=device_id,
        aggregation_strategy="simple",
    )

    product_asins = df["parent_asin"].unique()
    print(f"    -> Processing {len(product_asins)} products...")

    results = []
    for asin in tqdm(product_asins, desc="NER extraction"):
        product_reviews = df[df["parent_asin"] == asin]
        if helpful_col:
            product_reviews = product_reviews.nlargest(TOP_REVIEWS_PER_PRODUCT, helpful_col)
        else:
            product_reviews = product_reviews.head(TOP_REVIEWS_PER_PRODUCT)

        entity_counter = Counter()

        for _, row in product_reviews.iterrows():
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            text = text[:512]

            try:
                entities = ner(text)
                for ent in entities:
                    word = ent["word"].strip()
                    label = ent["entity_group"]
                    if word and len(word) > 1:
                        entity_counter[(word, label)] += 1
            except Exception:
                continue

        filtered = {
            f"{word} ({label})": count
            for (word, label), count in entity_counter.items()
            if count >= MIN_ENTITY_COUNT
        }

        results.append({
            "parent_asin": asin,
            "entities": json.dumps(filtered, ensure_ascii=False),
        })

    df_results = pd.DataFrame(results)
    df_results.to_parquet(OUTPUT_FILE, index=False)

    non_empty = sum(1 for r in results if r["entities"] != "{}")
    print(f"[*] Entity Recognition Complete.")
    print(f"    Products processed: {len(results)}")
    print(f"    Products with entities: {non_empty}")
    print(f"    Output: {OUTPUT_FILE.name}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
