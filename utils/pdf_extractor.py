"""Extract and prepare text from research paper PDFs (PyMuPDF)."""
import re
import sys
from pathlib import Path

import pymupdf

from utils import config as cfg
from utils.preprocessing import clean_text, truncate_words

SECTION_NAMES = {
    "abstract", "introduction", "related work", "background",
    "methodology", "methods", "method", "materials and methods",
    "experiments", "results", "discussion", "conclusion", "conclusions",
    "limitations", "future work", "references",
    "acknowledgements", "acknowledgments",
}
# sections that must NOT go into the model input
SKIP_SECTIONS = {"front", "abstract", "references", "acknowledgements", "acknowledgments"}

_NUM_PREFIX = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[ivx]+\.)\s+", re.IGNORECASE)


def extract_raw_text(pdf_source) -> str:
    """pdf_source: a file path, or raw bytes (what Streamlit's uploader gives)."""
    if isinstance(pdf_source, (bytes, bytearray)):
        doc = pymupdf.open(stream=bytes(pdf_source), filetype="pdf")
    else:
        doc = pymupdf.open(str(pdf_source))
    pages = [page.get_text("text") for page in doc]
    doc.close()
    text = "\n".join(pages)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words split across lines
    return text


def _heading_name(line: str):
    s = line.strip().lower().rstrip(":.")
    s = _NUM_PREFIX.sub("", s).strip()
    return s if s in SECTION_NAMES else None


def split_sections(raw_text: str) -> dict:
    """Split text into sections by known headings. Text before the first heading is 'front'."""
    sections = {"front": []}
    current = "front"
    for line in raw_text.splitlines():
        name = _heading_name(line)
        if name:
            current = name
            sections.setdefault(current, [])
        else:
            sections[current].append(line)
    return {k: " ".join(v) for k, v in sections.items()}


def prepare_for_model(pdf_source) -> dict:
    """Full pipeline: PDF -> sections -> cleaned, truncated model input."""
    raw = extract_raw_text(pdf_source)
    sections = split_sections(raw)

    body_parts = [txt for name, txt in sections.items() if name not in SKIP_SECTIONS]
    used_fallback = not any(body_parts) or "introduction" not in sections
    body = " ".join(body_parts) if not used_fallback else raw

    cleaned = clean_text(body)
    return {
        "raw_text": raw,
        "sections": sections,
        "sections_found": [k for k in sections if k != "front"],
        "used_fallback": used_fallback,  # True = headings not detected, whole text used
        "input_text": truncate_words(cleaned, cfg.MAX_ARTICLE_WORDS),
    }


if __name__ == "__main__":
    path = Path(sys.argv[1])
    result = prepare_for_model(path)
    print("Raw characters:", len(result["raw_text"]))
    print("Sections found:", result["sections_found"])
    print("Used fallback (no headings detected):", result["used_fallback"])
    print("Model input words:", len(result["input_text"].split()))
    print("\nMODEL INPUT (first 600 chars):\n", result["input_text"][:600])