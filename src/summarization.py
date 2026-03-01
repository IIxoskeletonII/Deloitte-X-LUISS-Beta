"""
summarization.py
----------------
Role: Review-Driven Product Insights. For each product, collects the most
helpful reviews, concatenates them up to the model's token limit, and
generates a human-readable summary capturing the overall user consensus
on strengths and weaknesses.

Model: facebook/bart-large-cnn (abstractive summarization).
Processes products in batches with checkpoint/resume support so that
interrupted runs can continue without re-processing.

Output: src/io/output/product_summaries.parquet
  Columns: parent_asin, summary, num_reviews_used

Usage: Run from root via module mode: python -m src.summarization
"""

import pandas as pd
import torch
from transformers import pipeline
from pathlib import Path
from tqdm import tqdm
import sys


CHECKPOINT_INTERVAL = 1000
MAX_INPUT_CHARS = 3500
BATCH_SIZE = 8
MIN_LENGTH = 50
MAX_LENGTH = 150


def main():
    SRC_DIR = Path(__file__).resolve().parent
    INPUT_FILE = SRC_DIR / "io" / "output" / "processed_data.parquet"
    OUTPUT_DIR = SRC_DIR / "io" / "output"
    OUTPUT_FILE = OUTPUT_DIR / "product_summaries.parquet"
    CHECKPOINT_FILE = OUTPUT_DIR / "_summarization_checkpoint.parquet"

    print("[*] Starting Review Summarization Pipeline...")

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

    product_asins = df["parent_asin"].unique()
    total_products = len(product_asins)
    print(f"    -> {total_products} products to summarize.")

    completed = {}
    if CHECKPOINT_FILE.exists():
        df_ckpt = pd.read_parquet(CHECKPOINT_FILE)
        for _, row in df_ckpt.iterrows():
            completed[row["parent_asin"]] = {
                "summary": row["summary"],
                "num_reviews_used": row["num_reviews_used"],
            }
        print(f"    -> Resuming from checkpoint: {len(completed)} products already done.")

    remaining_asins = [a for a in product_asins if a not in completed]
    if not remaining_asins:
        print("[*] All products already summarized. Saving final output...")
        results = [{"parent_asin": k, **v} for k, v in completed.items()]
        pd.DataFrame(results).to_parquet(OUTPUT_FILE, index=False)
        print(f"[*] Summarization Complete. Output: {OUTPUT_FILE.name}")
        return

    print(f"    -> Initializing facebook/bart-large-cnn...")
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=device_id,
        dtype=torch.float16 if device_id == 0 else torch.float32,
    )

    product_texts = []
    product_ids = []
    reviews_used_counts = []

    print("    -> Preparing review texts per product...")
    for asin in tqdm(remaining_asins, desc="Collecting reviews"):
        product_reviews = df[df["parent_asin"] == asin]
        if helpful_col:
            product_reviews = product_reviews.sort_values(helpful_col, ascending=False)

        combined = []
        char_count = 0
        num_used = 0
        for _, row in product_reviews.iterrows():
            review_text = str(row.get("text", "")).strip()
            if not review_text:
                continue
            if char_count + len(review_text) > MAX_INPUT_CHARS:
                break
            combined.append(review_text)
            char_count += len(review_text)
            num_used += 1

        if combined:
            product_texts.append(" ".join(combined))
            product_ids.append(asin)
            reviews_used_counts.append(num_used)

    print(f"    -> Generating summaries for {len(product_texts)} products...")
    processed = 0
    for i in tqdm(range(0, len(product_texts), BATCH_SIZE), desc="Summarizing"):
        batch_texts = product_texts[i : i + BATCH_SIZE]
        batch_ids = product_ids[i : i + BATCH_SIZE]
        batch_counts = reviews_used_counts[i : i + BATCH_SIZE]

        try:
            summaries = summarizer(
                batch_texts,
                max_length=MAX_LENGTH,
                min_length=MIN_LENGTH,
                do_sample=False,
                truncation=True,
            )
        except Exception:
            for j, (asin, count) in enumerate(zip(batch_ids, batch_counts)):
                try:
                    result = summarizer(
                        batch_texts[j],
                        max_length=MAX_LENGTH,
                        min_length=MIN_LENGTH,
                        do_sample=False,
                        truncation=True,
                    )
                    completed[asin] = {
                        "summary": result[0]["summary_text"],
                        "num_reviews_used": count,
                    }
                except Exception:
                    completed[asin] = {
                        "summary": "",
                        "num_reviews_used": 0,
                    }
            processed += len(batch_ids)
            continue

        for summary, asin, count in zip(summaries, batch_ids, batch_counts):
            completed[asin] = {
                "summary": summary["summary_text"],
                "num_reviews_used": count,
            }

        processed += len(batch_ids)

        if processed % CHECKPOINT_INTERVAL < BATCH_SIZE:
            ckpt_data = [{"parent_asin": k, **v} for k, v in completed.items()]
            pd.DataFrame(ckpt_data).to_parquet(CHECKPOINT_FILE, index=False)

    results = [{"parent_asin": k, **v} for k, v in completed.items()]
    df_results = pd.DataFrame(results)
    df_results.to_parquet(OUTPUT_FILE, index=False)

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    print(f"[*] Summarization Complete. {len(df_results)} products summarized.")
    print(f"    Output: {OUTPUT_FILE.name}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
