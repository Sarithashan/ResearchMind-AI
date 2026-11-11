# ResearchMind AI

An attention-based deep learning system for research paper summarization and analysis.

**Model:** Custom Seq2Seq with a Bidirectional LSTM encoder, Bahdanau attention and an LSTM decoder.
**Stack:** Python, TensorFlow/Keras, PyMuPDF, NLTK, Streamlit.

## Status
Phase 1 complete: project setup and Streamlit skeleton.

## Run locally
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    streamlit run frontend\app.py