"""Text cleaning used for BOTH the training dataset and uploaded PDFs."""
import re

_CITATION = re.compile(r"@xcite")
_MATH = re.compile(r"@xmath\d+")


def clean_text(text: str) -> str:
    """Normalize text so dataset text and PDF text look identical to the model."""
    text = text.lower()
    text = _CITATION.sub(" ", text)          # drop citation placeholders
    text = _MATH.sub(" xmath ", text)        # keep one token marking "a math symbol was here"
    text = text.replace("_", " ")            # drop italics markers
    text = re.sub(r"([,;:()?!])", r" \1 ", text)        # space around punctuation
    text = re.sub(r"\.(?=\s|$)", " . ", text)           # sentence-ending periods only
    text = re.sub(r"(?<=\w)-(?=\w)", " - ", text)       # hyphens inside words
    text = re.sub(r"\s+", " ", text).strip()            # collapse whitespace and newlines
    return text


def truncate_words(text: str, max_words: int) -> str:
    """Keep only the first max_words words."""
    return " ".join(text.split()[:max_words])