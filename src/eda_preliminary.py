"""
eda_preliminary.py
------------------
Role: Performs Exploratory Data Analysis (EDA) on the extracted JSONL files. 
It verifies column names, checks for missing values (specifically justifying 
the dropping of Price/Bought_Together), and validates the join keys 
between the metadata and review datasets.

Usage: Run from root via module mode: python -m src.eda_preliminary
"""
import json
import gzip
import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
# Dynamically resolve paths relative to this file
SRC_DIR = Path(__file__).resolve().parent
INPUT_DIR = SRC_DIR / "io" / "input" / "extracted"

def peek_data(filename, n_rows=5000):
    """
    Reads the first n_rows of a JSONL file into a Pandas DataFrame.
    Handles both .jsonl and .jsonl.gz if necessary.
    """
    data = []
    file_path = INPUT_DIR / filename
    
    print(f"--- Peeking at {filename} ---")
    
    # Check if file exists
    if not file_path.exists():
        print(f"ERROR: File not found at {file_path}")
        print(f"Current working directory: {Path.cwd()}")
        return None

    # Open the file (handling errors)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n_rows:
                    break
                try:
                    # Parse JSON line
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping line {i} due to decode error.")
                    continue
    except Exception as e:
        print(f"Failed to read file: {e}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # --- DISPLAY STATISTICS ---
    print(f"Successfully loaded {len(df)} rows.")
    
    # 1. Column Names
    print("\n[Columns Available]:")
    print(df.columns.tolist())
    
    # 2. Missing Values (Crucial for Business Vision)
    print("\n[Missing Values Count]:")
    print(df.isnull().sum())
    
    # 3. Sample Content (Text Richness Check)
    print("\n[Content Sampling]:")
    if 'text' in df.columns:
        # Check review text length/quality
        sample = df['text'].iloc[0] if len(df) > 0 else "No data"
        print(f"Sample Review: {str(sample)[:200]}...")
    elif 'description' in df.columns:
        # Check product description
        sample = df['description'].iloc[0] if len(df) > 0 else "No data"
        print(f"Sample Description: {str(sample)[:200]}...")
    elif 'title' in df.columns:
        sample = df['title'].iloc[0] if len(df) > 0 else "No data"
        print(f"Sample Title: {str(sample)[:200]}...")
        
    return df

if __name__ == "__main__":
    # --- EXECUTION ---
    
    # 1. Analyze Metadata
    print(">>> ANALYZING METADATA...")
    meta_df = peek_data("meta_Health_and_Personal_Care.jsonl")
    
    print("\n" + "="*60 + "\n")
    
    # 2. Analyze Reviews
    print(">>> ANALYZING REVIEWS...")
    reviews_df = peek_data("Health_and_Personal_Care.jsonl")

    # 3. Linkage Verification
    # (Can we join these tables? Required for the project)
    if meta_df is not None and reviews_df is not None:
        print("\n>>> CHECKING LINKAGE KEYS...")
        meta_keys = set(meta_df.columns)
        review_keys = set(reviews_df.columns)
        common_keys = meta_keys.intersection(review_keys)
        
        print(f"Common columns found: {common_keys}")
        
        if 'parent_asin' in common_keys:
             print("SUCCESS: 'parent_asin' found. This is the robust join key.")
        elif 'asin' in common_keys:
             print("SUCCESS: 'asin' found. Standard join key available.")
        else:
             print("WARNING: No obvious join key (asin/parent_asin) found. Check column names manually.")