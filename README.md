# ResearchMind AI

An attention-based deep learning system for research paper summarization and analysis. Upload a paper as a PDF, extract its text, and generate a short abstractive summary with a custom Seq2Seq model: a **Bidirectional LSTM encoder, Bahdanau attention and an LSTM decoder**, implemented from scratch in TensorFlow/Keras. The web app also shows measured model comparisons, an attention heatmap, grounded research insights and document Q&A.

> **Research prototype.** The model was trained from scratch on about 1,100 arXiv papers. Its summaries imitate the style of an abstract but are generic and often not specific to the paper. See [Results](#results) and [Limitations](#limitations).

## Features

| Page | What it does | Method |
|---|---|---|
| Upload Paper | Extracts text from a PDF, detects sections, cleans it | PyMuPDF + rules |
| Generate Summary | Abstractive summary with adjustable length | **Custom BiLSTM + Bahdanau attention model** |
| Research Insights | Objective, methodology, findings, conclusion, limitations | Cue-phrase sentence selection (text copied from the paper) |
| Model Comparison | ROUGE table, loss curves, inference time | Measured results from this project |
| Attention Visualization | Heatmap of where the model looked for each generated word | Attention weights of the custom model |
| Document Q&A | Returns the best-matching passages of the paper with their sections | TF-IDF passage retrieval (no generated answers) |

## Architecture

```
PDF -> text extraction -> cleaning -> section split -> tokenizer (10,000 words)
    -> embedding -> BiLSTM encoder -> Bahdanau attention -> LSTM decoder
    -> dense + softmax -> summary
```

Custom implementations (written for this project): word tokenizer, BiLSTM encoder, Bahdanau attention layer, attention decoder, Seq2Seq model, RNN/LSTM/GRU baselines, masked loss, training loop, greedy decoding with constraints.
Libraries: TensorFlow/Keras, PyMuPDF, pandas, NumPy, Streamlit, Plotly, rouge-score.

## Experiment setup

- **Data:** random subset of 1,500 papers from the arXiv summarization dataset (`ccdv/arxiv-summarization`, `document` configuration); after filtering, 1,116 train / 139 validation / 140 test.
- **Input:** first 500 words of the paper body (abstract and references removed). **Target:** first 80 words of the abstract.
- **Shared settings for all four models:** embedding 128, hidden size 256, vocabulary 10,000, batch size 32, Adam (learning rate 1e-3, gradient clipping 1.0), masked cross-entropy, teacher forcing, early stopping (patience 8, up to 60 epochs), seed 42.
- Baselines (RNN, LSTM, GRU) have no attention: the encoder's final state initializes the decoder.
- Training was run on a Google Colab T4 GPU.

## Results

Validation loss (lower is better; one run per model):

| Model | Best val loss | Best epoch / epochs run |
|---|---|---|
| Word-frequency-only floor | 6.3796 | n/a |
| RNN | 5.5711 | 22 / 30 |
| LSTM | 5.6069 | 31 / 39 |
| GRU | 5.6973 | 60 / 60 (not converged) |
| Attention (proposed) | 5.6416 | 25 / 33 |

Test ROUGE F1 (140 papers, stemming, against the 80-word reference). "Constrained" decoding forbids `<unk>` and repeated 3-grams.

| Model | Decoding | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|
| lead-80 (first 80 words of the paper) | extractive | 0.3281 | 0.0858 | 0.1976 |
| RNN | plain greedy | 0.1642 | 0.0116 | 0.1588 |
| RNN | constrained | 0.1561 | 0.0260 | 0.1339 |
| LSTM | plain greedy | 0.1950 | 0.0226 | 0.1605 |
| LSTM | constrained | 0.2767 | 0.0417 | 0.1979 |
| GRU | plain greedy | 0.1828 | 0.0282 | 0.1403 |
| GRU | constrained | 0.2715 | 0.0469 | 0.1902 |
| Attention (proposed) | plain greedy | 0.1475 | 0.0181 | 0.1280 |
| Attention (proposed) | constrained | 0.2708 | 0.0429 | 0.2032 |

**What the measurements show**
- All models beat the word-frequency-only baseline on validation loss.
- Copying the first 80 words of the paper scored higher on ROUGE-1 and ROUGE-2 than every from-scratch model.
- The attention model had the highest ROUGE-L among the neural models, but LSTM and GRU were close. These are single runs on a small test set, so the differences between LSTM, GRU and attention should not be treated as significant.
- The plain RNN was the weakest on generated text.
- Constrained decoding raised ROUGE for every model except the RNN's ROUGE-1. Part of the gain is because `<unk>` can never match a reference word.

## Limitations

- Small training set (about 1,100 papers) and a from-scratch model: summaries are generic and not paper-specific, and may contain wrong topics.
- A 10,000-word vocabulary turns many technical words into `<unk>`; math and citation placeholders reduce the usable text.
- Only the first 500 words of the body are read, so methods and results later in the paper are not seen.
- Results are from single runs; no confidence intervals or repeated seeds.
- Greedy decoding only; summaries are not verified against the paper.
- PDF extraction is rule-based: scanned PDFs are not supported, math may be garbled, and unusual section headings fall back to the whole text.
- Research Insights and Document Q&A are classic retrieval methods, not neural models.

## Future work

More training data, subword tokenization, pretrained embeddings, beam search with a coverage mechanism, chunked reading of long papers, repeated runs with confidence intervals, and a pretrained summarizer as an optional comparison baseline.

## Project structure

```
ResearchMind-AI/
|-- frontend/app.py          Streamlit app
|-- models/                  encoder.py, attention.py, decoder.py, seq2seq.py,
|                            baseline_models.py, tokenizer.json, saved/ (trained weights)
|-- utils/                   config, tokenizer, preprocessing, pdf_extractor, training,
|                            inference, summarizer, insights, dataset scripts
|-- results/                 measured ROUGE tables, loss histories, test predictions
|-- requirements.txt
```

## Run locally (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run frontend\app.py
```

TensorFlow runs on the CPU on Windows; the first summary takes longer because the model compiles once.

## Reproduce the data and training

```powershell
python utils\download_dataset.py
python -m utils.prepare_dataset
python -m utils.build_tokenizer
python -m utils.make_arrays
```

The arrays were then uploaded to Google Colab, where the models were trained with `models/` and `utils/training.py`. The dataset itself is not stored in this repository.