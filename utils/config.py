"""Shared settings for preprocessing, training and the web app."""

# Text-length strategy
MAX_ARTICLE_WORDS = 500   # model input: first 500 words of the paper
MAX_SUMMARY_WORDS = 80    # model target: first 80 words of the abstract

# Quality filters (removes empty or malformed rows)
MIN_ARTICLE_WORDS = 200
MIN_ABSTRACT_WORDS = 30
MAX_ABSTRACT_WORDS = 400

# Split
RANDOM_SEED = 42
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1   # the remaining 10% is the test set

# Tokenizer / sequence lengths
MAX_VOCAB = 10000          # keep the 10,000 most frequent words (incl. special tokens)
MIN_FREQ = 2               # ignore words seen fewer than 2 times in training data
ARTICLE_LEN = MAX_ARTICLE_WORDS          # encoder input length
SUMMARY_LEN = MAX_SUMMARY_WORDS + 2      # decoder length: words + <sos> + <eos>