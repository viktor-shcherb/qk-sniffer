#!/usr/bin/env python
"""
Prepare LongBench-v2 long-context subset and upload to Hugging Face.

Steps:
1. Load env vars from .env (expects HF_TOKEN at least).
2. Download zai-org/LongBench-v2.
3. Keep only examples with length == "long".
4. Convert to dataset with columns: id, text (text = context).
5. Push to viktoroo/longbench2-128k-plus.

Requirements:
    pip install datasets huggingface_hub python-dotenv
"""

import os

from dotenv import load_dotenv
from datasets import load_dataset

SOURCE_DATASET_ID = "zai-org/LongBench-v2"
TARGET_REPO_ID = "viktoroo/longbench2-128k-plus"


def main():
    # 1. Load env vars from .env
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Put your token in a .env file as HF_TOKEN=..."
        )

    # 2. Download dataset
    # The repo has a single split: train (503 rows). :contentReference[oaicite:0]{index=0}
    print(f"Loading dataset: {SOURCE_DATASET_ID}")
    ds = load_dataset(SOURCE_DATASET_ID, split="train", token=hf_token)

    print(f"Total rows before filtering: {len(ds)}")

    # 3. Keep only 'long' examples (length field is 'short'/'medium'/'long'). :contentReference[oaicite:1]{index=1}
    ds_long = ds.filter(lambda ex: ex["length"] == "long")
    print(f"Rows with length == 'long': {len(ds_long)}")

    # 4. Map to {id, text}, where:
    #    id   = original _id
    #    text = original context (the long document). :contentReference[oaicite:2]{index=2}
    def to_id_text(ex):
        return {
            "id": ex["_id"],
            "text": ex["context"],
        }

    ds_id_text = ds_long.map(
        to_id_text,
        remove_columns=ds_long.column_names,  # keep only id + text
    )

    # 5. Push to Hugging Face Hub
    print(f"Pushing to Hub: {TARGET_REPO_ID}")
    ds_id_text.push_to_hub(
        repo_id=TARGET_REPO_ID,
        token=hf_token,
    )

    print("Done.")


if __name__ == "__main__":
    main()
