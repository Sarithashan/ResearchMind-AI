"""Build the vocabulary from the TRAINING split only, then report statistics."""
import pandas as pd

from utils import config as cfg
from utils.tokenizer import UNK_ID, Tokenizer

TOKENIZER_PATH = "models/tokenizer.json"


def unk_rate(tokenizer, texts):
    """Fraction of words that map to <unk>."""
    total = unk = 0
    for text in texts:
        for w in text.split():
            total += 1
            unk += tokenizer.word2idx.get(w, UNK_ID) == UNK_ID
    return unk / total


def main():
    train = pd.read_parquet("dataset/processed/train.parquet")
    val = pd.read_parquet("dataset/processed/val.parquet")

    # IMPORTANT: build from training data only. Using val/test would leak information.
    texts = list(train["input_text"]) + list(train["summary"])
    tok = Tokenizer.build(texts, max_vocab=cfg.MAX_VOCAB, min_freq=cfg.MIN_FREQ)
    tok.save(TOKENIZER_PATH)

    print("Vocabulary size:", tok.vocab_size)
    print("Unknown-word rate, train inputs:", f"{unk_rate(tok, train['input_text']):.1%}")
    print("Unknown-word rate, val inputs:  ", f"{unk_rate(tok, val['input_text']):.1%}")
    print("Unknown-word rate, val summaries:", f"{unk_rate(tok, val['summary']):.1%}")

    # Round-trip test on one summary
    sample = train["summary"].iloc[0]
    ids = tok.encode(sample, cfg.SUMMARY_LEN, add_sos=True, add_eos=True)
    print("\nOriginal :", sample)
    print("IDs      :", ids[:15], "...")
    print("Decoded  :", tok.decode(ids))
    print("\nSaved to", TOKENIZER_PATH)


if __name__ == "__main__":
    main()