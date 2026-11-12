"""A simple word-level tokenizer built from scratch."""
import json
from collections import Counter
from pathlib import Path

PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"
SPECIAL_TOKENS = [PAD, UNK, SOS, EOS]
PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3


class Tokenizer:
    def __init__(self, word2idx: dict):
        self.word2idx = word2idx
        self.idx2word = {i: w for w, i in word2idx.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    @classmethod
    def build(cls, texts, max_vocab: int, min_freq: int = 2):
        """Create a vocabulary from the most frequent words in `texts`."""
        counter = Counter(word for text in texts for word in text.split())
        word2idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        slots = max_vocab - len(SPECIAL_TOKENS)
        for word, count in counter.most_common(slots):
            if count >= min_freq:
                word2idx[word] = len(word2idx)
        return cls(word2idx)

    def encode(self, text: str, max_len: int, add_sos=False, add_eos=False) -> list:
        """Words -> ids, truncated and padded to exactly max_len."""
        ids = [self.word2idx.get(w, UNK_ID) for w in text.split()]
        budget = max_len - int(add_sos) - int(add_eos)
        ids = ids[:budget]
        if add_sos:
            ids = [SOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids + [PAD_ID] * (max_len - len(ids))

    def decode(self, ids) -> str:
        """ids -> words. Stops at <eos>, skips <pad> and <sos>."""
        words = []
        for i in ids:
            i = int(i)
            if i == EOS_ID:
                break
            if i in (PAD_ID, SOS_ID):
                continue
            words.append(self.idx2word.get(i, UNK))
        return " ".join(words)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))