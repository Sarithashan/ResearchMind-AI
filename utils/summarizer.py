"""Inference for the proposed attention model (used by the Streamlit app)."""
import re
from pathlib import Path

import numpy as np

from utils import config as cfg
from utils.tokenizer import EOS_ID, PAD_ID, SOS_ID, UNK_ID, Tokenizer

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "saved" / "attention_infer.weights.h5"


def load_summarizer():
    """Build the custom Seq2Seq attention model and load the trained weights."""
    from models.seq2seq import Seq2SeqAttention   # imports TensorFlow/Keras

    tok = Tokenizer.load(ROOT / "models" / "tokenizer.json")
    model = Seq2SeqAttention(tok.vocab_size, cfg.EMBED_DIM, cfg.HIDDEN_DIM)
    model([
        np.zeros((1, cfg.ARTICLE_LEN), dtype="int32"),
        np.zeros((1, cfg.SUMMARY_LEN - 1), dtype="int32"),
    ])                                              # builds the weights
    model.load_weights(str(WEIGHTS))
    return tok, model


def generate_ids(tok, fast, input_text, max_words=cfg.MAX_SUMMARY_WORDS,
                 ban_unk=True, no_repeat_ngram=3):
    """Greedy decoding for ONE paper (same constrained decoding as in evaluation).

    Returns (encoder ids, decoder ids, number of generated words)."""
    enc = np.array([tok.encode(input_text, cfg.ARTICLE_LEN)], dtype="int32")
    dec_len = cfg.SUMMARY_LEN - 1
    dec = np.zeros((1, dec_len), dtype="int32")
    dec[0, 0] = SOS_ID
    k = no_repeat_ngram
    n_words = 0

    for t in range(min(max_words, dec_len - 1)):
        probs = fast([enc, dec]).numpy()[0, t, :].copy()
        probs[[PAD_ID, SOS_ID]] = 0
        if ban_unk:
            probs[UNK_ID] = 0
        if k:
            seq = dec[0, 1:t + 1].tolist()
            if len(seq) >= k - 1:
                prefix = tuple(seq[-(k - 1):])
                for j in range(len(seq) - k + 1):
                    if tuple(seq[j:j + k - 1]) == prefix:
                        probs[seq[j + k - 1]] = 0
        nxt = int(probs.argmax())
        dec[0, t + 1] = nxt
        if nxt == EOS_ID:
            break
        n_words += 1
    return enc, dec, n_words


def summarize(tok, fast, input_text, max_words=cfg.MAX_SUMMARY_WORDS,
              ban_unk=True, no_repeat_ngram=3):
    _, dec, _ = generate_ids(tok, fast, input_text, max_words, ban_unk, no_repeat_ngram)
    return tok.decode(dec[0])


def attention_map(fast, enc, dec, n_words):
    """Attention weights, shape (n_words, 500): row i = where the model looked for word i."""
    _, attn = fast.model([enc, dec], return_attention=True, training=False)
    return np.asarray(attn)[0, :n_words, :]


def tidy(text):
    """Make tokenized text a little easier to read."""
    text = text.replace("xmath", "[math]")
    text = re.sub(r"\s+([.,;:)])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+-\s+", "-", text)
    return text[:1].upper() + text[1:]


if __name__ == "__main__":
    import time

    import pandas as pd

    from utils.inference import FastModel

    tok, model = load_summarizer()
    fast = FastModel(model)
    df = pd.read_parquet(ROOT / "dataset" / "processed" / "test.parquet")
    t0 = time.time()
    print(tidy(summarize(tok, fast, df["input_text"].iloc[0], 60)))
    print("seconds:", round(time.time() - t0, 1))