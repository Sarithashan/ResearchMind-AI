"""Convert the processed text splits into integer arrays for the models."""
import numpy as np
import pandas as pd

from utils import config as cfg
from utils.tokenizer import Tokenizer


def build_split(df, tok):
    enc = np.array(
        [tok.encode(t, cfg.ARTICLE_LEN) for t in df["input_text"]], dtype="int32"
    )
    full = np.array(
        [tok.encode(t, cfg.SUMMARY_LEN, add_sos=True, add_eos=True) for t in df["summary"]],
        dtype="int32",
    )
    decoder_input = full[:, :-1]    # <sos> w1 ... w80
    decoder_target = full[:, 1:]    # w1 ... w80 <eos>
    return enc, decoder_input, decoder_target


def main():
    tok = Tokenizer.load("models/tokenizer.json")
    for name in ["train", "val", "test"]:
        df = pd.read_parquet(f"dataset/processed/{name}.parquet")
        enc, dec_in, dec_tgt = build_split(df, tok)

        # Sanity check: input and target must be the same sequence shifted by one
        assert (dec_in[:, 1:] == dec_tgt[:, :-1]).all()

        np.savez_compressed(
            f"dataset/processed/{name}_arrays.npz",
            encoder_input=enc,
            decoder_input=dec_in,
            decoder_target=dec_tgt,
        )
        print(f"{name}: encoder {enc.shape}, decoder_in {dec_in.shape}, decoder_target {dec_tgt.shape}")

    print("\nFirst training example (first 8 ids):")
    print("decoder_input :", dec_in[0][:8]) if False else None
    train = np.load("dataset/processed/train_arrays.npz")
    print("decoder_input :", train["decoder_input"][0][:8])
    print("decoder_target:", train["decoder_target"][0][:8])


if __name__ == "__main__":
    main()