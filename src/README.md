# DiscoverAI - Source Code

## Overview

This directory contains the full pipeline for the DiscoverAI semantic search system.
The system processes the Health & Personal Care dataset from Amazon Reviews 2023,
builds product-level semantic embeddings fusing metadata and review signals, generates
review summaries and entity tags, and exposes an interactive search interface.

## Execution Order

Run each module in sequence from the repository root:

| Step | File                   | Description                                              |
|------|------------------------|----------------------------------------------------------|
| 1    | `data_extraction.py`   | Decompress raw `.jsonl.gz` files                         |
| 2    | `eda_preliminary.py`   | Exploratory data analysis and validation                 |
| 3    | `preprocessing.py`     | Clean, filter, merge datasets into Parquet               |
| 4    | `modelling.py`         | Product-level embeddings + FAISS index                   |
| 5    | `summarization.py`     | Generate review summaries per product (BART)             |
| 6    | `entity_recognition.py`| Extract named entities from reviews (BERT-NER)           |
| 7    | `search_engine.py`     | Load artifacts and run sample queries (standalone test)  |
| 8    | `demo.py`              | Launch interactive Gradio demo                           |

Each file is run as a module:
```
python -m src.data_extraction
python -m src.eda_preliminary
python -m src.preprocessing
python -m src.modelling
python -m src.summarization
python -m src.entity_recognition
python -m src.search_engine
python -m src.demo
```

Alternatively, use the `main.ipynb` notebook at the repository root to run the
full pipeline in Google Colab with GPU acceleration.

## Models Used

| Task                 | Model                        | Dimensions | Source         |
|----------------------|------------------------------|------------|----------------|
| Text Embedding       | `all-mpnet-base-v2`          | 768        | Sentence-BERT  |
| Summarization        | `facebook/bart-large-cnn`    | -          | Hugging Face   |
| Entity Recognition   | `dslim/bert-base-NER`        | -          | Hugging Face   |

## Directory Structure

```
src/
  io/
    input/
      raw/         <- Place .jsonl.gz files here
      extracted/   <- Decompressed JSONL files
    output/        <- All pipeline outputs (Parquet, FAISS, NumPy)
```

## Prerequisites

- Python 3.10+
- See `requirements.txt` in the repository root for all dependencies
- GPU recommended (Google Colab with T4/A100)
- For GPU FAISS: `pip install faiss-gpu-cu12` (code falls back to `faiss-cpu`)

## Output Artifacts

| File                       | Description                                    |
|----------------------------|------------------------------------------------|
| `processed_data.parquet`   | Cleaned and merged dataset                     |
| `product_metadata.parquet` | Deduplicated product catalog                   |
| `product_embeddings.npy`   | Product embedding vectors (N x 768)            |
| `product_ids.npy`          | ASIN array mapping FAISS index rows to products|
| `product_index.faiss`      | Serialized FAISS index for similarity search   |
| `product_summaries.parquet`| Review summaries per product                   |
| `product_entities.parquet` | Named entities extracted per product            |
