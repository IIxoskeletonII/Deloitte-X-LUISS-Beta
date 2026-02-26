"""
modelling.py
------------
Role: NLP Task A - Semantic Representation. 
Loads the cleaned parquet file, combines product metadata (Title, Description, 
Features) with Review Text into a single rich text feature, and encodes it 
into a shared semantic space using a Hugging Face Transformer.
Utilizes GPU acceleration if available.

Usage: Run from root via module mode: python -m src.modelling
"""

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys

def clean_text_field(col_data):
    """Safely converts lists/arrays to strings and handles NaN values."""
    if isinstance(col_data, (list, tuple, np.ndarray)):
        if len(col_data) == 0:
            return ""
        return " ".join([str(item) for item in col_data if item])
    
    # 2. Now it is safe to check for scalar NaNs (like None or float('nan'))
    if pd.isna(col_data):
        return ""
    
    # 3. Convert standard scalar values (strings, ints) to string
    return str(col_data)

def main():
    # 1. Setup Paths
    SRC_DIR = Path(__file__).resolve().parent
    INPUT_FILE = SRC_DIR / "io" / "output" / "processed_data.parquet"
    OUTPUT_FILE = SRC_DIR / "io" / "output" / "embedded_data.parquet"

    print("[*] Starting NLP Task A: Semantic Representation...")

    if not INPUT_FILE.exists():
        print(f"[!] Error: Could not find {INPUT_FILE}")
        print("    Did you run preprocessing.py first?")
        sys.exit(1)

    # 2. Hardware Detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    -> Compute Device Detected: {device.upper()}")
    if device == "cpu":
        print("    [!] WARNING: Running on CPU. This will take significantly longer.")

    # 3. Load Data
    print("    -> Loading preprocessed dataset...")
    df = pd.read_parquet(INPUT_FILE)
    
    # 4. Feature Engineering: Create the "Shared Semantic Space" text
    # The brief explicitly mandates combining Title + Description + Features + Review Text
    print("    -> Constructing combined semantic text...")
    
    # Apply safe cleaning to handle potential lists and NaNs
    for col in ['title', 'description', 'features', 'text']:
        if col in df.columns:
            df[col] = df[col].apply(clean_text_field)
        else:
            df[col] = "" # Fallback if a column is completely missing

    # Concatenate with clear separators to help the model understand context
    df['semantic_text'] = (
        "Product: " + df['title'] + " | " +
        "Description: " + df['description'] + " | " +
        "Features: " + df['features'] + " | " +
        "Review: " + df['text']
    )

    # 5. Model Initialization
    model_name = 'all-MiniLM-L6-v2'
    print(f"    -> Initializing Hugging Face Model: {model_name}...")
    model = SentenceTransformer(model_name, device=device)

    # 6. Embedding Generation
    print(f"    -> Encoding {len(df)} rows. This will take some time...")
    print("    -> Progress bar active:")
    
    # Generate embeddings. show_progress_bar gives us ETA. 
    # batch_size=256 is highly optimized for an RTX 3060 6GB.
    embeddings = model.encode(
        df['semantic_text'].tolist(),
        batch_size=256, 
        show_progress_bar=True,
        device=device
    )

    # 7. Save Results
    print("    -> Attaching embeddings to dataset...")
    # Convert numpy array of embeddings to a list of arrays to store in a pandas DataFrame
    df['embedding'] = list(embeddings)

    print(f"    -> Saving embedded data to {OUTPUT_FILE.name}...")
    # Parquet natively supports storing lists/arrays, which is perfect for embeddings
    df.drop(columns=['semantic_text'], inplace=True) # Drop to save disk space
    df.to_parquet(OUTPUT_FILE, index=False)

    print("[*] Semantic Representation Complete.")

if __name__ == "__main__":
    main()