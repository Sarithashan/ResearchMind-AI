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