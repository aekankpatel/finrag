import streamlit as st
from pathlib import Path
from pydantic import Field
from datetime import datetime
import gdown
from llama_index.core import load_index_from_storage, StorageContext, Settings
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

INDEX_DIR = Path("finrag/index")
PROCESSED_DIR = Path("finrag/data/processed")
FOLDER_ID = "1NiAUNTdyA6OX5V1ZVxoT1KrgOKk2kEQs"

# ---- Auto-download index from Google Drive if missing ----
def ensure_index():
    if not INDEX_DIR.exists() or not (INDEX_DIR / "docstore.json").exists():
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(
            id=FOLDER_ID,
            output=str(INDEX_DIR),
            quiet=False,
        )

ensure_index()

ALL_DOCS = sorted([
    f.name for f in PROCESSED_DIR.glob("*.txt")
    if f.name != "manifest.csv"
]) if PROCESSED_DIR.exists() else []

COMPANY_MAP = {
    "meta": "meta_10K_2025.txt",
    "facebook": "meta_10K_2025.txt",
    "instagram": "meta_10K_2025.txt",
    "amazon": "Amazon_10K_2025.txt",
    "amzn": "Amazon_10K_2025.txt",
    "aws": "Amazon_10K_2025.txt",
    "nvidia": "NVIDIA_10Q_2025Q3.txt",
    "nvda": "NVIDIA_10Q_2025Q3.txt",
    "bank of america": "BAC+2024+Annual+Report.txt",
    "bofa": "BAC+2024+Annual+Report.txt",
    "bac": "BAC+2024+Annual+Report.txt",
    "goldman": "GoldmanSachsBDC_10Q._2025Q2pdf.txt",
    "apple": "aaple_10Q_2025Q1.txt",
    "aapl": "aaple_10Q_2025Q1.txt",
    "microsoft": "Microsoft_10Q_2025Q3.txt",
    "msft": "Microsoft_10Q_2025Q3.txt",
    "tesla": "tesla_10K_2025.txt",
    "tsla": "tesla_10K_2025.txt",
    "walmart": "Walmart (WMT) Q4 2026 Earnings Call Transcript | The Motley Fool.txt",
    "wmt": "Walmart (WMT) Q4 2026 Earnings Call Transcript | The Motley Fool.txt",
    "jpmorgan": "jpm-4q25-earnings-call-transcript.txt",
    "jpm": "jpm-4q25-earnings-call-transcript.txt",
    "capital markets": "capital-markets-forecast-2026.txt",
    "global economy": "Global Economic Prospects, January 2026.txt",
    "macro": "Global Economic Prospects, January 2026.txt",
    "banking outlook": "ey-global-banking-outlook-2025-en.txt",
    "m&a": "goldman-sachs-2026-global-ma-outlook.txt",
}

def detect_source(question: str):
    q = question.lower()
    for keyword, filename in COMPANY_MAP.items():
        if keyword in q:
            return filename
    return None

def confidence_color(score: float) -> str:
    if score >= 0.75:
        return "#00c896"
    elif score >= 0.55:
        return "#f5a623"
    else:
        return "#e05c5c"

def format_export(question, answer, sources, source_filter):
    lines = [
        "FinRAG — Financial Intelligence Export",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Source filter: {source_filter or 'All documents'}",
        "=" * 60,
        f"QUESTION:\n{question}",
        "=" * 60,
        f"ANSWER:\n{answer}",
        "=" * 60,
        "SOURCES:",
    ]
    for i, node in enumerate(sources, 1):
        src = node.metadata.get("source", "unknown")
        score = round(node.score, 4) if node.score else "N/A"
        lines.append(f"\n[{i}] {src} | Score: {score}")
        lines.append(node.text[:300].strip())
    return "\n".join(lines)

st.set_page_config(page_title="FinRAG", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0f1117; }
    .source-card {
        background-color: #1e2130;
        border-left: 3px solid #00c896;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.85em;
        color: #cdd6f4;
    }
    .answer-box {
        background-color: #1e2130;
        border-left: 4px solid #7c93f7;
        padding: 16px 20px;
        border-radius: 8px;
        color: #ffffff;
        font-size: 1em;
        line-height: 1.6;
    }
    .detected-badge {
        background-color: #1e2130;
        border: 1px solid #00c896;
        color: #00c896;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8em;
        display: inline-block;
        margin-bottom: 12px;
    }
    .history-question {
        background-color: #2a2d3e;
        border-radius: 8px;
        padding: 10px 14px;
        color: #a0aec0;
        font-size: 0.9em;
        margin-bottom: 4px;
    }
    .history-answer {
        background-color: #1e2130;
        border-left: 3px solid #7c93f7;
        border-radius: 6px;
        padding: 10px 14px;
        color: #cdd6f4;
        font-size: 0.9em;
        margin-bottom: 16px;
    }
    .compare-col {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 16px;
        color: #ffffff;
        line-height: 1.6;
        height: 100%;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading financial index...")
def load_index():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    llm = Groq(model="llama3-8b-8192", api_key=st.secrets["GROQ_API_KEY"])
    Settings.embed_model = embed_model
    Settings.llm = llm
    storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    return load_index_from_storage(storage_context)


def run_query(index, question, source_filter=None, top_k=8):
    if source_filter:
        filters = MetadataFilters(filters=[
            MetadataFilter(key="source", value=source_filter)
        ])
        engine = index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact",
            filters=filters,
        )
    else:
        engine = index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact",
        )
    return engine.query(f"{question} Answer only using information from the provided source documents. If the documents do not contain relevant information, say so explicitly rather than generalizing.")


# ---- Session state ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "question" not in st.session_state:
    st.session_state.question = ""
if "mode" not in st.session_state:
    st.session_state.mode = "Single"

index = load_index()

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.mode = st.radio("Mode", ["Single", "Compare"], horizontal=True)

    manual_filter = st.selectbox(
        "Filter by document (optional override)",
        ["Auto-detect", "All documents"] + ALL_DOCS
    )
    top_k = st.slider("Number of source chunks", min_value=3, max_value=15, value=8)
    st.divider()

    st.markdown("**Suggested questions:**")
    suggestions = [
        "What are Meta's key regulatory risks?",
        "How did NVIDIA describe AI demand?",
        "What is Amazon's revenue outlook?",
        "What risks does Apple highlight in their 10-Q?",
        "How does Bank of America describe interest rate risk?",
        "What is Tesla's outlook for 2025?",
        "What does the global banking outlook say about AI?",
        "How did JPMorgan describe the macro environment?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True):
            st.session_state.question = s

    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ---- Header ----
st.title("📊 FinRAG")
st.caption("Retrieval-Augmented Intelligence for Financial Documents")

# ---- Chat history ----
if st.session_state.chat_history:
    st.subheader("💬 Chat History")
    for entry in st.session_state.chat_history:
        st.markdown(f'<div class="history-question">🧑 {entry["question"]} <span style="float:right;font-size:0.75em;color:#555">{entry["time"]} · {entry["source"]}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="history-answer">🤖 {entry["answer"]}</div>', unsafe_allow_html=True)
    st.divider()

# ---- SINGLE MODE ----
if st.session_state.mode == "Single":
    question = st.text_input(
        "Ask a question about your financial documents",
        value=st.session_state.question,
        placeholder="e.g. What are Meta's key regulatory risks?",
    )

    if question:
        if manual_filter == "Auto-detect":
            source_filter = detect_source(question)
        elif manual_filter == "All documents":
            source_filter = None
        else:
            source_filter = manual_filter

        with st.spinner("Thinking..."):
            response = run_query(index, question, source_filter, top_k)

        label = source_filter or "All documents"
        st.markdown(f'<div class="detected-badge">🔍 Searching: {label}</div>', unsafe_allow_html=True)

        if response.source_nodes:
            top_score = response.source_nodes[0].score or 0
            color = confidence_color(top_score)
            st.markdown(f"""
                <div style="margin-bottom:12px;">
                    <span style="font-size:0.8em;color:#888;">Retrieval confidence</span><br>
                    <div style="background:#2a2d3e;border-radius:10px;height:8px;width:100%;margin-top:4px;">
                        <div style="background:{color};width:{min(int(top_score*100),100)}%;height:8px;border-radius:10px;"></div>
                    </div>
                    <span style="font-size:0.75em;color:{color};">{round(top_score*100,1)}%</span>
                </div>
            """, unsafe_allow_html=True)

        st.subheader("🧠 Answer")
        st.markdown(f'<div class="answer-box">{response.response}</div>', unsafe_allow_html=True)

        st.download_button(
            label="⬇️ Export answer as .txt",
            data=format_export(question, response.response, response.source_nodes, source_filter),
            file_name=f"finrag_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )

        st.subheader("📄 Sources")
        for i, node in enumerate(response.source_nodes, 1):
            source = node.metadata.get("source", "unknown")
            score = round(node.score, 4) if node.score else "N/A"
            preview = node.text[:400].strip().replace("\n", " ")
            st.markdown(f'''
                <div class="source-card">
                    <strong>[{i}] {source}</strong> &nbsp;|&nbsp; Score: <code>{score}</code><br><br>
                    {preview}...
                </div>
            ''', unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "question": question,
            "answer": response.response[:300] + "..." if len(response.response) > 300 else response.response,
            "source": label,
            "time": datetime.now().strftime("%H:%M"),
        })
        st.session_state.question = ""

# ---- COMPARE MODE ----
else:
    st.subheader("⚖️ Compare Mode")
    st.caption("Ask the same question across two documents side by side")

    compare_question = st.text_input(
        "Question to compare",
        placeholder="e.g. What are the key risk factors?",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        doc_a = st.selectbox("Company A", ALL_DOCS, index=0)
    with col_b:
        doc_b = st.selectbox("Company B", ALL_DOCS, index=1)

    if compare_question and st.button("⚖️ Compare", use_container_width=True):
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"### 🏢 {doc_a}")
            with st.spinner(f"Querying {doc_a}..."):
                resp_a = run_query(index, compare_question, doc_a, top_k)
            score_a = resp_a.source_nodes[0].score if resp_a.source_nodes else 0
            color_a = confidence_color(score_a)
            st.markdown(f'<div style="font-size:0.75em;color:{color_a};margin-bottom:8px;">Confidence: {round(score_a*100,1)}%</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="compare-col">{resp_a.response}</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown(f"### 🏢 {doc_b}")
            with st.spinner(f"Querying {doc_b}..."):
                resp_b = run_query(index, compare_question, doc_b, top_k)
            score_b = resp_b.source_nodes[0].score if resp_b.source_nodes else 0
            color_b = confidence_color(score_b)
            st.markdown(f'<div style="font-size:0.75em;color:{color_b};margin-bottom:8px;">Confidence: {round(score_b*100,1)}%</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="compare-col">{resp_b.response}</div>', unsafe_allow_html=True)

        export_text = f"FinRAG — Comparison Export\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\nQUESTION: {compare_question}\n\n{'='*60}\n{doc_a}:\n{resp_a.response}\n\n{'='*60}\n{doc_b}:\n{resp_b.response}"
        st.download_button(
            label="⬇️ Export comparison as .txt",
            data=export_text,
            file_name=f"finrag_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )
