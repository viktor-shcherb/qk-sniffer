#!/usr/bin/env python
"""Build viktoroo/fineweb-edu-long-context-sample from the FineWeb-Edu sample."""

from __future__ import annotations

import os
from typing import Iterable

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm.auto import tqdm

HF_TOKEN_ENV = "HF_TOKEN"

TARGET_DATASET_ID = os.environ.get("TARGET_DATASET_ID", "viktoroo/fineweb-edu-long-context-sample")
SOURCE_DATASET_ID = os.environ.get("SOURCE_DATASET_ID", "HuggingFaceFW/fineweb-edu")
SOURCE_CONFIG = os.environ.get("SOURCE_DATASET_CONFIG", "sample-10BT")
MIN_TOKEN_COUNT = int(os.environ.get("MIN_TOKEN_COUNT", 128 * 1024))
TARGET_EXAMPLES = int(os.environ.get("TARGET_EXAMPLES", 1024))
PRIVATE = os.environ.get("PRIVATE", "true").lower() == "true"


def iter_long_examples(*, limit: int = None) -> Iterable[dict]:
    stream = load_dataset(SOURCE_DATASET_ID, SOURCE_CONFIG, split="train", streaming=True)
    total = 0
    for example in stream:
        if example.get("token_count", 0) >= MIN_TOKEN_COUNT:
            yield example
            total += 1

        if total >= limit:
            break


def main() -> None:
    load_dotenv()
    hf_token = os.environ.get(HF_TOKEN_ENV)
    if not hf_token:
        raise RuntimeError(f"Set {HF_TOKEN_ENV} in your .env before running this script.")
    examples = []
    for example in tqdm(
            iter_long_examples(limit=TARGET_EXAMPLES),
            total=TARGET_EXAMPLES,
            desc="Collecting long contexts",
            unit="example"
    ):
        examples.append(example)

    if len(examples) < TARGET_EXAMPLES:
        raise RuntimeError(
            f"Only gathered {len(examples)} examples with token_count >= {MIN_TOKEN_COUNT}. "
            "Consider lowering MIN_TOKEN_COUNT or decreasing the sample size."
        )
    dataset = Dataset.from_list(examples)
    dataset.push_to_hub(TARGET_DATASET_ID, token=hf_token, private=PRIVATE)
    print(f"Uploaded {len(dataset)} rows to {TARGET_DATASET_ID} from {SOURCE_DATASET_ID}:{SOURCE_CONFIG}")


if __name__ == "__main__":
    main()
