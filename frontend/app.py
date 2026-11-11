import streamlit as st

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
    .soon {
        text-align: center; padding: 3rem 1rem; border: 2px dashed #c4b5fd;
        border-radius: 16px; background: #faf5ff; color: #4c1d95;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
COMING_SOON = {
    "Upload Paper": ("Upload a research paper PDF and preview the extracted text.", "Phase 3"),
    "Generate Summary": ("Generate an abstractive summary with the custom Seq2Seq model.", "Phase 10"),
    "Research Insights": ("Objectives, methodology, findings and conclusions.", "Phase 9"),
    "Model Comparison": ("Compare RNN, LSTM, GRU and the attention model.", "Phase 8"),
    "Attention Visualization": ("See which words influenced the summary.", "Phase 9"),
    "Document Q&A": ("Ask questions answered from your uploaded paper.", "Phase 9"),
}

if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <h1>ResearchMind AI</h1>
            <p>An attention-based deep learning system that reads research papers
            and writes clear, concise summaries.</p>
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
        '<div class="card"><h3>📊 Analyze</h3><p>Compare RNN, LSTM, GRU and attention models using ROUGE scores.</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="badge">Prototype in development</span>', unsafe_allow_html=True)
    st.markdown("### How to use")
    st.markdown(
        "1. Open **Upload Paper** and upload a PDF.\n"
        "2. Go to **Generate Summary** to create a summary.\n"
        "3. Explore **Research Insights**, **Model Comparison** and **Attention Visualization**."
    )
else:
    desc, phase = COMING_SOON[page]
    st.title(page)
    st.markdown(
        f"""
        <div class="soon">
            <h2>🚧 Coming soon</h2>
            <p>{desc}</p>
            <span class="badge">Planned for {phase}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )