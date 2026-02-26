"""
data_extraction.py
------------------
Role: This script handles the raw data ingestion. It identifies compressed 
.jsonl.gz files in the raw input directory, decompresses them, and stores 
the result in the extracted folder for further processing.

Usage: Run from root via module mode: python -m src.data_extraction
"""

import gzip
import shutil
import sys
from pathlib import Path

def main():
    # 1. Setup Paths (Relative to Project Root)
    # We resolve from this file's location: DUMMY-VARIABLES/src/data_extraction.py
    SRC_DIR = Path(__file__).resolve().parent
    ROOT_DIR = SRC_DIR.parent
    
    # Define Input/Output according to standard structure
    # Input: src/io/input/raw
    INPUT_DIR = SRC_DIR / "io" / "input" / "raw"
    # Output: src/io/input/extracted
    OUTPUT_DIR = SRC_DIR / "io" / "input" / "extracted"

    print(f"[*] Starting Data Extraction...")
    print(f"    Source: {INPUT_DIR}")
    print(f"    Target: {OUTPUT_DIR}")

    # 2. Validation
    if not INPUT_DIR.exists():
        print(f"[!] Error: Input directory not found at {INPUT_DIR}")
        print("    Please ensure you have placed the .jsonl.gz files in src/io/input/raw/")
        sys.exit(1)

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Extraction Loop
    found_files = list(INPUT_DIR.glob("*.jsonl.gz"))
    
    if not found_files:
        print("[!] Warning: No .jsonl.gz files found to extract.")
        return

    for file_path in found_files:
        # Determine output filename (remove .gz extension)
        # file.stem on "data.jsonl.gz" returns "data.jsonl"
        output_file_path = OUTPUT_DIR / file_path.stem
        
        print(f"    -> Extracting: {file_path.name}...")
        
        try:
            with gzip.open(file_path, "rb") as f_in:
                with open(output_file_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"       [OK] Saved to: {output_file_path.name}")
        except Exception as e:
            print(f"       [X] Failed to extract {file_path.name}: {e}")

    print("[*] Extraction Complete.")

if __name__ == "__main__":
    main()