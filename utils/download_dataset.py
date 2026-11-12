"""Download a small random subset of the arXiv summarization dataset."""
from pathlib import Path

import pandas as pd
from datasets import load_dataset

NUM_PAPERS = 1500
OUT_PATH = Path("dataset/raw/arxiv_subset.parquet")


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # streaming=True reads papers one by one instead of downloading everything
    stream = load_dataset(
        "ccdv/arxiv-summarization", "document", split="train", streaming=True
    )
    # shuffle inside a small buffer so we don't only get the first papers
    stream = stream.shuffle(seed=42, buffer_size=1000)

    rows = []
    for i, example in enumerate(stream):
        if i == 0:
            # Check the REAL schema before trusting documentation
            print("Columns found in the dataset:", list(example.keys()))
        if i >= NUM_PAPERS:
            break
        rows.append(
            {
                # use the dataset's id if present, otherwise create our own
                "id": example.get("id", f"paper_{i:04d}"),
                "article": example["article"],
                "abstract": example["abstract"],
            }
        )
        if (i + 1) % 250 == 0:
            print(f"Downloaded {i + 1} papers...")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df)} papers to {OUT_PATH}")


if __name__ == "__main__":
    main()