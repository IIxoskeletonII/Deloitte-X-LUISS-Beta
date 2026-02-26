"""
verify_embeddings.py
--------------------
Role: A temporary diagnostic script to mathematically verify the integrity 
of the generated embeddings before we commit to the next phase.
"""

import pandas as pd
from pathlib import Path
import numpy as np

def main():
    SRC_DIR = Path(__file__).resolve().parent
    FILE_PATH = SRC_DIR / "io" / "output" / "embedded_data.parquet"

    print(f"[*] Loading {FILE_PATH.name} for verification...")
    df = pd.read_parquet(FILE_PATH)

    print("\n--- EMBEDDING VERIFICATION REPORT ---")
    
    # 1. Check if column exists
    if 'embedding' not in df.columns:
        print("[X] FAILED: 'embedding' column not found!")
        return
    print("[OK] 'embedding' column exists.")

    # 2. Check dimensions
    sample_embedding = df['embedding'].iloc[0]
    dim_size = len(sample_embedding)
    print(f"[*] Detected Embedding Dimensions: {dim_size}")
    
    if dim_size == 384:
        print("[OK] Dimensionality matches all-MiniLM-L6-v2 spec (384).")
    else:
        print(f"[X] FAILED: Expected 384 dimensions, found {dim_size}.")

    # 3. Check for nulls/failures
    null_count = df['embedding'].isna().sum()
    if null_count == 0:
        print(f"[OK] No missing embeddings detected across {len(df)} rows.")
    else:
        print(f"[X] FAILED: Found {null_count} missing embeddings.")

    # 4. Show a sample
    print("\n[*] Sample Vector (First 5 values):")
    print(np.round(sample_embedding[:5], 4))
    print("-------------------------------------")

if __name__ == "__main__":
    main()