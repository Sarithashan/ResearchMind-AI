import html
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import config as cfg
from utils.insights import build_passages, extract_insights, retrieve
from utils.pdf_extractor import prepare_for_model
from utils.preprocessing import clean_text, truncate_words
from utils.summarizer import attention_map, generate_ids, summarize, tidy

RESULTS = ROOT / "results"

st.set_page_config(page_title="ResearchMind AI", page_icon="📚", layout="wide")

# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1100px; }
    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        padding: 3rem 2.5rem; border-radius: 20px; color: white;
        margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(79,70,229,0.3);
    }
    .hero h1 { font-size: 3rem; margin: 0 0 0.5rem 0; color: white; }
    .hero p  { font-size: 1.15rem; opacity: 0.95; margin: 0; }
    .card {
        background: #ffffff; border: 1px solid #e5e7eb; border-radius: 16px;
        padding: 1.5rem; height: 100%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }
    .card h3 { margin-top: 0; color: #4f46e5; }
    .card p  { color: #374151; margin-bottom: 0; }
    .badge {
        display: inline-block; background: #ede9fe; color: #5b21b6;
        padding: 0.3rem 0.9rem; border-radius: 999px;
        font-size: 0.85rem; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained model (first time only)...")
def get_model():
    from utils.inference import FastModel
    from utils.summarizer import load_summarizer

    tok, model = load_summarizer()
    return tok, FastModel(model)


# ---------- Sidebar ----------
st.sidebar.title("📚 ResearchMind AI")
page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Upload Paper",
        "Generate Summary",
        "Research Insights",
        "Model Comparison",
        "Attention Visualization",
        "Document Q&A",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Deep Learning + NLP project")


# ---------- Pages ----------
if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <h1>ResearchMind AI</h1>
            <p>An attention-based deep learning system that reads research papers
            and writes short abstractive summaries.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        '<div class="card"><h3>📄 Upload</h3><p>Upload a research paper PDF and the text is extracted automatically.</p></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        '<div class="card"><h3>🧠 Summarize</h3><p>A custom BiLSTM encoder with Bahdanau attention and an LSTM decoder writes the summary.</p></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        '<div class="card"><h3>📊 Compare</h3><p>See measured ROUGE scores and loss curves for RNN, LSTM, GRU and the attention model.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="badge">Research prototype</span>', unsafe_allow_html=True)
    st.markdown("### How to use")
    st.markdown(
        "1. Open **Upload Paper** and upload a PDF.\n"
        "2. Go to **Generate Summary** and click generate.\n"
        "3. Explore **Research Insights**, **Document Q&A** and **Attention Visualization**.\n"
        "4. Open **Model Comparison** to see the measured results."
    )
    st.info(
        "The model was trained from scratch on about 1,100 arXiv papers, so its summaries "
        "are generic and may not reflect the paper's actual content. See Model Comparison for details."
    )

elif page == "Upload Paper":
    st.title("Upload Paper")
    uploaded = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

    if uploaded is not None:
        with st.spinner("Extracting text..."):
            result = prepare_for_model(uploaded.getvalue())

        if len(result["raw_text"].strip()) < 200:
            st.error(
                "Very little text was found. This PDF may be a scanned image, "
                "which is not supported yet."
            )
            st.stop()

        st.session_state["paper"] = result
        st.session_state["paper_name"] = uploaded.name
        st.session_state.pop("last_summary", None)
        st.session_state.pop("attn_result", None)
        st.success(f"Uploaded: {uploaded.name}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Characters extracted", f"{len(result['raw_text']):,}")
        c2.metric("Sections detected", len(result["sections_found"]))
        c3.metric("Model input words", len(result["input_text"].split()))

        if result["used_fallback"]:
            st.warning("Section headings were not detected, so the whole text was used.")
        else:
            st.markdown("**Sections found:** " + ", ".join(result["sections_found"]))

        with st.expander("Preview of extracted text"):
            st.text(result["raw_text"][:3000])
        with st.expander("Cleaned text that will be given to the model"):
            st.text(result["input_text"])

elif page == "Generate Summary":
    st.title("Generate Summary")
    paper = st.session_state.get("paper")

    if paper:
        st.success(f"Using uploaded paper: {st.session_state.get('paper_name', '')}")
        input_text = paper["input_text"]
    else:
        st.info("No paper uploaded yet. Upload one on the Upload Paper page, or paste text below.")
        pasted = st.text_area("Paste the paper body (without the abstract)", height=200)
        input_text = (
            truncate_words(clean_text(pasted), cfg.MAX_ARTICLE_WORDS) if pasted.strip() else ""
        )

    max_words = st.slider(
        "Maximum summary length (words)", 20, cfg.MAX_SUMMARY_WORDS, 60, step=10
    )

    if st.button("Generate summary", type="primary", disabled=not input_text):
        tok, fast = get_model()
        t0 = time.time()
        with st.spinner("Generating..."):
            raw = summarize(tok, fast, input_text, max_words=max_words)
        st.session_state["last_summary"] = (tidy(raw), time.time() - t0)

    if "last_summary" in st.session_state:
        summary, elapsed = st.session_state["last_summary"]
        st.subheader("Generated summary (model output)")
        st.markdown(
            f'<div class="card"><p>{html.escape(summary)}</p></div>', unsafe_allow_html=True
        )
        st.caption(f"Generated in {elapsed:.1f} s on this computer's CPU.")
        st.warning(
            "This text was produced by a small model trained from scratch. It imitates the style "
            "of an abstract and is **not verified against the paper**. Do not rely on it for facts."
        )

    if input_text:
        with st.expander("Text taken directly from the paper (first 80 words of the cleaned body)"):
            st.text(" ".join(input_text.split()[:80]))

elif page == "Research Insights":
    st.title("Research Insights")
    paper = st.session_state.get("paper")
    if not paper:
        st.info("Upload a paper first on the Upload Paper page.")
        st.stop()
    st.success(f"Paper: {st.session_state.get('paper_name', '')}")
    st.caption(
        "Every sentence below is copied from the paper. It was picked by matching cue phrases "
        "(for example 'we propose' or 'results show'). It is not generated by the neural network, "
        "and this simple matching can miss or mis-pick sentences."
    )
    for category, items in extract_insights(paper["sections"]).items():
        st.subheader(category)
        if not items:
            st.write("Nothing identifiable in the text for this category.")
        for sent, sec in items:
            st.markdown(
                f'<div class="card"><p>{html.escape(sent)}</p></div>', unsafe_allow_html=True
            )
            st.caption(f"Source: '{sec}' section of the paper")

elif page == "Model Comparison":
    st.title("Model Comparison")
    v2 = RESULTS / "test_results_v2.csv"
    if not v2.exists():
        st.error("results/test_results_v2.csv was not found.")
        st.stop()

    df = pd.read_csv(v2)
    st.subheader("ROUGE on the test set (140 papers)")
    st.caption(
        "Scores are F1 with stemming, measured against the first 80 words of each paper's "
        "abstract. 'lead-80' simply copies the first 80 words of the paper (extractive baseline)."
    )
    st.dataframe(df.round(4), hide_index=True)

    chart_df = df[df["decoding"] != "plain greedy"].melt(
        id_vars=["model", "decoding"],
        value_vars=["ROUGE-1", "ROUGE-2", "ROUGE-L"],
        var_name="metric",
        value_name="score",
    )
    fig = px.bar(
        chart_df, x="metric", y="score", color="model", barmode="group",
        title="ROUGE scores (constrained decoding)",
    )
    st.plotly_chart(fig)

    st.subheader("Training and validation loss")
    names = {"rnn": "RNN", "lstm": "LSTM", "gru": "GRU", "attention": "Attention (proposed)"}
    fig2 = go.Figure()
    rows = []
    for key, label in names.items():
        p = RESULTS / f"{key}_history.json"
        if not p.exists():
            continue
        h = json.loads(p.read_text())
        epochs = list(range(1, len(h["val_loss"]) + 1))
        fig2.add_trace(go.Scatter(x=epochs, y=h["val_loss"], mode="lines", name=f"{label} (val)"))
        fig2.add_trace(
            go.Scatter(x=epochs, y=h["loss"], mode="lines", name=f"{label} (train)",
                       line=dict(dash="dot"))
        )
        best = min(h["val_loss"])
        rows.append({
            "model": label,
            "best val loss": round(best, 4),
            "best epoch": h["val_loss"].index(best) + 1,
            "epochs run": len(epochs),
        })
    fig2.add_hline(y=6.3796, line_dash="dash",
                   annotation_text="word-frequency-only floor (6.3796)")
    fig2.update_layout(xaxis_title="Epoch", yaxis_title="Loss")
    st.plotly_chart(fig2)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    t = RESULTS / "test_results.csv"
    if t.exists():
        st.subheader("Inference time")
        tdf = pd.read_csv(t)[["model", "sec/summary (batched)"]]
        tdf = tdf[tdf["model"] != "lead-80 (extractive)"]
        st.dataframe(tdf.round(4), hide_index=True)
        st.caption("Batched decoding of 140 papers on a Colab T4 GPU, divided by 140. "
                   "This is not single-document latency.")

    st.subheader("What the measurements show")
    st.markdown(
        """
- All four neural models beat a word-frequency-only baseline on validation loss.
- On ROUGE-1 and ROUGE-2, simply copying the first 80 words of the paper scored higher than every from-scratch model.
- The attention model had the highest ROUGE-L of the neural models, but LSTM and GRU were close. These are single runs on 140 papers, so the differences are small.
- The plain RNN was clearly the weakest on generated text.
- Banning unknown words and repeated phrases at generation time raised ROUGE for every model. Part of that gain is because unknown tokens can never match a reference word.
- Generated summaries are generic and not paper-specific. Likely reasons: a small training set (about 1,100 papers), a 10,000-word vocabulary, and a 500-word input window.
        """
    )

elif page == "Attention Visualization":
    st.title("Attention Visualization")
    paper = st.session_state.get("paper")
    if paper:
        st.success(f"Using uploaded paper: {st.session_state.get('paper_name', '')}")
        input_text = paper["input_text"]
    else:
        st.info("No paper uploaded yet. Upload one on the Upload Paper page, or paste text below.")
        pasted = st.text_area("Paste the paper body (without the abstract)", height=200)
        input_text = (
            truncate_words(clean_text(pasted), cfg.MAX_ARTICLE_WORDS) if pasted.strip() else ""
        )

    max_words = st.slider("Summary length to analyse (words)", 10, 60, 30, step=10)
    if st.button("Compute attention", type="primary", disabled=not input_text):
        tok, fast = get_model()
        with st.spinner("Generating the summary and collecting attention weights..."):
            enc, dec, n = generate_ids(tok, fast, input_text, max_words=max_words)
            attn = attention_map(fast, enc, dec, n) if n else None
        st.session_state["attn_result"] = {
            "words": [tok.idx2word.get(int(i), "?") for i in dec[0, 1:1 + n]],
            "attn": attn,
            "input_words": input_text.split()[:cfg.ARTICLE_LEN],
        }

    res = st.session_state.get("attn_result")
    if res and res["attn"] is not None:
        words, attn, iw = res["words"], res["attn"], res["input_words"]
        L = len(iw)
        size = 20
        nb = -(-L // size)
        binned = np.stack(
            [attn[:, b * size:(b + 1) * size].sum(axis=1) for b in range(nb)], axis=1
        )
        fig = go.Figure(go.Heatmap(
            z=binned,
            x=[f"{b * size + 1}-{min((b + 1) * size, L)}" for b in range(nb)],
            y=[f"{i + 1}. {w}" for i, w in enumerate(words)],
            colorscale="Viridis",
        ))
        fig.update_layout(
            xaxis_title="Input word positions (20-word blocks)",
            yaxis_title="Generated word",
            yaxis=dict(autorange="reversed"),
            height=max(400, 18 * len(words) + 150),
        )
        st.plotly_chart(fig)

        c1, c2 = st.columns(2)
        c1.metric("Average peak attention weight", f"{float(attn[:, :L].max(axis=1).mean()):.3f}")
        c2.metric("Value if attention were spread evenly", f"{1 / L:.3f}")

        idx = st.selectbox(
            "Pick a generated word to inspect", range(len(words)),
            format_func=lambda i: f"{i + 1}. {words[i]}",
        )
        row = attn[idx, :L]
        top = np.argsort(row)[::-1][:8]
        st.dataframe(
            pd.DataFrame({
                "input position": top + 1,
                "input word": [iw[j] for j in top],
                "attention weight": np.round(row[top], 4),
            }),
            hide_index=True,
        )
        st.caption(
            "Attention weights show where the model looked while choosing each word. They do not "
            "prove that those words caused the output. If the peak weight is close to the evenly "
            "spread value, the model's attention is diffuse."
        )
    elif res:
        st.warning("The model produced no words for this input.")

elif page == "Document Q&A":
    st.title("Document Q&A")
    paper = st.session_state.get("paper")
    if not paper:
        st.info("Upload a paper first on the Upload Paper page.")
        st.stop()
    st.success(f"Paper: {st.session_state.get('paper_name', '')}")
    st.caption(
        "This finds the passages of your paper that best match the question. It does not write "
        "an answer: every result is text copied from the paper, shown with its section."
    )
    examples = [
        "",
        "What is the objective of this research?",
        "Which methodology is used?",
        "What are the main findings?",
        "What limitations are mentioned?",
    ]
    example = st.selectbox("Example questions", examples)
    question = st.text_input("Ask a question about the uploaded paper", value=example)
    if question.strip():
        hits = retrieve(question, build_passages(paper["sections"]), k=3)
        if not hits:
            st.warning("No passage in the paper matches this question.")
        for score, p in hits:
            st.markdown(
                f'<div class="card"><p>{html.escape(p["text"])}</p></div>', unsafe_allow_html=True
            )
            st.caption(f"Section: {p['section']} | passage {p['id']} | match score {score:.2f}")