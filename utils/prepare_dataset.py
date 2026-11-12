"""Clean the raw subset, apply length limits, and save train/val/test splits."""
from pathlib import Path

import pandas as pd

from utils import config as cfg
from utils.preprocessing import clean_text, truncate_words

RAW = Path("dataset/raw/arxiv_subset.parquet")
OUT_DIR = Path("dataset/processed")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RAW)
    print("Raw papers:", len(df))

    # 1. Clean text
    df["article_clean"] = df["article"].apply(clean_text)
    df["abstract_clean"] = df["abstract"].apply(clean_text)

    # 2. Filter bad rows (word counts measured AFTER cleaning)
    df["article_words"] = df["article_clean"].str.split().str.len()
    df["abstract_words"] = df["abstract_clean"].str.split().str.len()
    keep = (
        (df["article_words"] >= cfg.MIN_ARTICLE_WORDS)
        & (df["abstract_words"] >= cfg.MIN_ABSTRACT_WORDS)
        & (df["abstract_words"] <= cfg.MAX_ABSTRACT_WORDS)
    )
    print("Removed by filters:", (~keep).sum())
    df = df[keep].copy()

    # 3. Create training pairs with the length strategy
    df["input_text"] = df["article_clean"].apply(
        lambda t: truncate_words(t, cfg.MAX_ARTICLE_WORDS)
    )
    df["summary"] = df["abstract_clean"].apply(
        lambda t: truncate_words(t, cfg.MAX_SUMMARY_WORDS)
    )

    # Keep the full cleaned abstract too, for fair evaluation later
    out = df[["id", "input_text", "summary", "abstract_clean"]].rename(
        columns={"abstract_clean": "abstract_full"}
    )

    # 4. Shuffle and split 80/10/10
    out = out.sample(frac=1, random_state=cfg.RANDOM_SEED).reset_index(drop=True)
    n = len(out)
    n_train = int(n * cfg.TRAIN_FRAC)
    n_val = int(n * cfg.VAL_FRAC)
    splits = {
        "train": out.iloc[:n_train],
        "val": out.iloc[n_train : n_train + n_val],
        "test": out.iloc[n_train + n_val :],
    }

    # 5. Save
    for name, part in splits.items():
        part.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
        print(f"{name}: {len(part)} papers")

    print("Saved to", OUT_DIR)


if __name__ == "__main__":
    main()