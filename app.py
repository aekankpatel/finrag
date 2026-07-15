import streamlit as st
from pathlib import Path
from datetime import datetime
import requests
from llama_index.core import load_index_from_storage, StorageContext, Settings
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

INDEX_DIR = Path("finrag/index")
RAW_BASE = "https://raw.githubusercontent.com/aekankpatel/finrag-index/main"
LFS_BASE = "https://media.githubusercontent.com/media/aekankpatel/finrag-index/main"
LFS_FILES = {"default__vector_store.json"}
INDEX_FILES = ["docstore.json", "index_store.json", "default__vector_store.json", "graph_store.json"]

def ensure_index():
    if not INDEX_DIR.exists() or not (INDEX_DIR / "docstore.json").exists():
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        for fname in INDEX_FILES:
            base = LFS_BASE if fname in LFS_FILES else RAW_BASE
            url = f"{base}/{fname}"
            r = requests.get(url, stream=True, timeout=300)
            r.raise_for_status()
            with open(INDEX_DIR / fname, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

ensure_index()

ALL_DOCS = [
    "aaple_10K_2025.txt","aaple_10Q_2025Q1.txt","aaple_10Q_2025Q4.txt",
    "Amazon (AMZN) Q4 2025 Earnings Call Transcript | The Motley Fool.txt",
    "Amazon_10K_2025.txt","Amazon_10Q_2025Q3.txt","BAC+2024+Annual+Report.txt",
    "BofA (BAC) Q4 2025 Earnings Call Transcript | The Motley Fool.txt",
    "capital-markets-forecast-2026.txt","ey-global-banking-outlook-2025-en.txt",
    "Global Economic Prospects, January 2026.txt",
    "Goldman Sachs (GS) Q4 2025 Earnings Transcript | The Motley Fool.txt",
    "goldman-sachs-2026-global-ma-outlook.txt","GoldmanSachsBDC_10Q._2025Q2pdf.txt",
    "jpm-4q25-earnings-call-transcript.txt","meta_10K_2025.txt",
    "Microsoft_10Q_2025Q3.txt","microsoft_EC_2025Q2.txt",
    "Nvidia (NVDA) Q4 2025 Earnings Call Transcript | The Motley Fool.txt",
    "NVIDIA_10Q_2025Q3.txt","tesla_10K_2025.txt","tesla_10Q_2025Q3.txt",
    "tesla.txt","The BEAT.txt",
    "Walmart (WMT) Q4 2026 Earnings Call Transcript | The Motley Fool.txt",
]

COMPANY_MAP = {
    "meta": "meta_10K_2025.txt","facebook": "meta_10K_2025.txt","instagram": "meta_10K_2025.txt",
    "amazon": "Amazon_10K_2025.txt","amzn": "Amazon_10K_2025.txt","aws": "Amazon_10K_2025.txt",
    "nvidia": "NVIDIA_10Q_2025Q3.txt","nvda": "NVIDIA_10Q_2025Q3.txt",
    "bank of america": "BAC+2024+Annual+Report.txt","bofa": "BAC+2024+Annual+Report.txt","bac": "BAC+2024+Annual+Report.txt",
    "goldman": "GoldmanSachsBDC_10Q._2025Q2pdf.txt",
    "apple": "aaple_10Q_2025Q1.txt","aapl": "aaple_10Q_2025Q1.txt",
    "microsoft": "Microsoft_10Q_2025Q3.txt","msft": "Microsoft_10Q_2025Q3.txt",
    "tesla": "tesla_10K_2025.txt","tsla": "tesla_10K_2025.txt",
    "walmart": "Walmart (WMT) Q4 2026 Earnings Call Transcript | The Motley Fool.txt",
    "jpmorgan": "jpm-4q25-earnings-call-transcript.txt","jpm": "jpm-4q25-earnings-call-transcript.txt",
    "capital markets": "capital-markets-forecast-2026.txt",
    "global economy": "Global Economic Prospects, January 2026.txt",
    "macro": "Global Economic Prospects, January 2026.txt",
    "banking outlook": "ey-global-banking-outlook-2025-en.txt",
    "m&a": "goldman-sachs-2026-global-ma-outlook.txt",
}

def detect_source(question):
    q = question.lower()
    for keyword, filename in COMPANY_MAP.items():
        if keyword in q:
            return filename
    return None

def confidence_color(score):
    if score >= 0.75: return "#166534"
    elif score >= 0.55: return "#92400E"
    else: return "#991B1B"

def format_export(question, answer, sources, source_filter):
    lines = ["FinRAG — Financial Intelligence Export",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Source filter: {source_filter or 'All documents'}","=" * 60,
        f"QUESTION:\n{question}","=" * 60,f"ANSWER:\n{answer}","=" * 60,"SOURCES:"]
    for i, node in enumerate(sources, 1):
        src = node.metadata.get("source", "unknown")
        score = round(node.score, 4) if node.score else "N/A"
        lines.append(f"\n[{i}] {src} | Score: {score}")
        lines.append(node.text[:300].strip())
    return "\n".join(lines)

st.set_page_config(page_title="FinRAG", page_icon="📰", layout="wide")
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Lora:ital,wght@0,400;0,500;1,400&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #F6F1EB !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #EDE8E1 !important;
    border-right: 1px solid #D6CFC6;
}
[data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif; }
h1 { font-family: 'Playfair Display', serif !important; font-weight: 600 !important; color: #1C1410 !important; letter-spacing: -0.5px; }
h2, h3 { font-family: 'Playfair Display', serif !important; font-weight: 400 !important; color: #2D2219 !important; }
.finrag-header { border-bottom: 2px solid #1C1410; padding-bottom: 10px; margin-bottom: 4px; }
.finrag-kicker { font-family: 'DM Sans', sans-serif; font-size: 0.7em; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; color: #92400E; margin-bottom: 6px; }
.answer-box {
    background: #FFFFFF;
    border: 1px solid #D6CFC6;
    border-top: 3px solid #1C1410;
    padding: 20px 24px;
    border-radius: 2px;
    color: #1C1410;
    font-family: 'Lora', serif;
    font-size: 1em;
    line-height: 1.8;
    margin: 8px 0 16px 0;
}
.source-card {
    background: #FFFFFF;
    border: 1px solid #D6CFC6;
    border-left: 3px solid #92400E;
    padding: 12px 16px;
    border-radius: 2px;
    margin-bottom: 8px;
    font-size: 0.83em;
    color: #3D3530;
    font-family: 'DM Sans', sans-serif;
    line-height: 1.6;
}
.source-card strong { color: #1C1410; font-weight: 500; }
.source-card code { background: #F0EBE4; color: #92400E; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }
.detected-badge {
    background: #FEF3C7;
    border: 1px solid #D97706;
    color: #92400E;
    padding: 3px 10px;
    border-radius: 3px;
    font-size: 0.75em;
    font-weight: 500;
    letter-spacing: 0.05em;
    display: inline-block;
    margin-bottom: 12px;
    font-family: 'DM Sans', sans-serif;
    text-transform: uppercase;
}
.history-question {
    background: #EDE8E1;
    border-radius: 3px;
    padding: 9px 14px;
    color: #5C4F44;
    font-size: 0.88em;
    margin-bottom: 3px;
    font-family: 'DM Sans', sans-serif;
}
.history-answer {
    background: #FFFFFF;
    border-left: 2px solid #92400E;
    padding: 9px 14px;
    color: #3D3530;
    font-family: 'Lora', serif;
    font-size: 0.88em;
    line-height: 1.6;
    margin-bottom: 14px;
}
.compare-col {
    background: #FFFFFF;
    border: 1px solid #D6CFC6;
    border-top: 3px solid #1C1410;
    padding: 16px;
    border-radius: 2px;
    color: #1C1410;
    font-family: 'Lora', serif;
    line-height: 1.75;
}
.confidence-bar-bg { background: #E8E1D8; border-radius: 2px; height: 6px; width: 100%; margin-top: 5px; }
.stTextInput > div > div > input {
    border: 1px solid #C4BCB3 !important;
    border-radius: 2px !important;
    background: #FFFFFF !important;
    color: #1C1410 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: #92400E !important;
    box-shadow: 0 0 0 1px #92400E !important;
}
.stButton > button {
    border: 1px solid #C4BCB3 !important;
    border-radius: 2px !important;
    background: transparent !important;
    color: #3D3530 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85em !important;
}
.stButton > button:hover { background: #EDE8E1 !important; border-color: #92400E !important; color: #92400E !important; }
.stDownloadButton > button { background: #1C1410 !important; color: #F6F1EB !important; border: none !important; border-radius: 2px !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.85em !important; }
.stDownloadButton > button:hover { background: #2D2219 !important; }
[data-testid="stMarkdownContainer"] p { color: #3D3530; font-family: 'DM Sans', sans-serif; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading financial index...")
def load_index():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    Settings.llm = None
    storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    return load_index_from_storage(storage_context)

def run_query(index, question, source_filter=None, top_k=8):
    from groq import Groq as GroqClient
    if source_filter:
        filters = MetadataFilters(filters=[MetadataFilter(key="source", value=source_filter)])
        retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
    else:
        retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(question)
    context = "\n\n".join([n.text[:500] for n in nodes[:5]])
    client = GroqClient(api_key=st.secrets["GROQ_API_KEY"])
    chat_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a financial analyst assistant. Answer questions using only the provided context from financial documents. Be specific and cite relevant details."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer only using the context above."}
        ],
        temperature=0.1, max_tokens=1024,
    )
    class SimpleResponse:
        def __init__(self, text, source_nodes):
            self.response = text
            self.source_nodes = source_nodes
    return SimpleResponse(chat_response.choices[0].message.content, nodes)

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "question" not in st.session_state: st.session_state.question = ""
if "mode" not in st.session_state: st.session_state.mode = "Single"

index = load_index()

with st.sidebar:
    st.markdown('<p class="finrag-kicker">Settings</p>', unsafe_allow_html=True)
    st.session_state.mode = st.radio("Mode", ["Single", "Compare"], horizontal=True)
    if st.session_state.mode == "Single":
        manual_filter = st.selectbox("Filter by document", ["Auto-detect", "All documents"] + ALL_DOCS)
    else:
        manual_filter = "All documents"
    top_k = st.slider("Source passages", min_value=3, max_value=15, value=8)
    st.divider()
    st.markdown('<p class="finrag-kicker">Suggested questions</p>', unsafe_allow_html=True)
    suggestions = [
        "What are Meta's key regulatory risks?","How did NVIDIA describe AI demand?",
        "What is Amazon's revenue outlook?","What risks does Apple highlight in their 10-Q?",
        "How does Bank of America describe interest rate risk?","What is Tesla's outlook for 2025?",
        "What does the global banking outlook say about AI?","How did JPMorgan describe the macro environment?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True): st.session_state.question = s
    st.divider()
    if st.button("Clear history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

st.markdown('<p class="finrag-kicker">Financial intelligence</p>', unsafe_allow_html=True)
st.markdown('<div class="finrag-header"><h1>FinRAG</h1></div>', unsafe_allow_html=True)
st.markdown('<p style="color:#78716C;font-family:\'DM Sans\',sans-serif;font-size:0.9em;margin-top:4px;">Ask questions across earnings calls, 10-Ks, and market reports.</p>', unsafe_allow_html=True)

if st.session_state.chat_history:
    st.markdown('<p class="finrag-kicker" style="margin-top:24px;">Previous questions</p>', unsafe_allow_html=True)
    for entry in st.session_state.chat_history:
        st.markdown(f'<div class="history-question">{entry["question"]} <span style="float:right;font-size:0.75em;color:#A8998C">{entry["time"]} · {entry["source"]}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="history-answer">{entry["answer"]}</div>', unsafe_allow_html=True)
    st.divider()

if st.session_state.mode == "Single":
    question = st.text_input("", value=st.session_state.question, placeholder="e.g. What are Meta's key regulatory risks?", label_visibility="collapsed")
    if question:
        if manual_filter == "Auto-detect": source_filter = detect_source(question)
        elif manual_filter == "All documents": source_filter = None
        else: source_filter = manual_filter
        with st.spinner("Searching documents..."):
            response = run_query(index, question, source_filter, top_k)
        label = source_filter or "All documents"
        st.markdown(f'<div class="detected-badge">Source — {label}</div>', unsafe_allow_html=True)
        if response.source_nodes:
            top_score = response.source_nodes[0].score or 0
            color = confidence_color(top_score)
            st.markdown(f"""<div style="margin-bottom:16px;">
                <span style="font-size:0.75em;color:#78716C;font-family:'DM Sans',sans-serif;text-transform:uppercase;letter-spacing:0.08em;">Retrieval confidence</span>
                <div class="confidence-bar-bg"><div style="background:{color};width:{min(int(top_score*100),100)}%;height:6px;border-radius:2px;"></div></div>
                <span style="font-size:0.75em;color:{color};font-family:'DM Sans',sans-serif;font-weight:500;">{round(top_score*100,1)}%</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('<p class="finrag-kicker">Answer</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">{response.response}</div>', unsafe_allow_html=True)
        st.download_button(label="Export as .txt",
            data=format_export(question, response.response, response.source_nodes, source_filter),
            file_name=f"finrag_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", mime="text/plain")
        st.markdown('<p class="finrag-kicker" style="margin-top:20px;">Source passages</p>', unsafe_allow_html=True)
        for i, node in enumerate(response.source_nodes, 1):
            source = node.metadata.get("source", "unknown")
            score = round(node.score, 4) if node.score else "N/A"
            preview = node.text[:400].strip().replace("\n", " ")
            st.markdown(f'<div class="source-card"><strong>[{i}] {source}</strong> &nbsp;&middot;&nbsp; <code>{score}</code><br><br>{preview}…</div>', unsafe_allow_html=True)
        st.session_state.chat_history.append({"question": question,
            "answer": response.response[:300] + "..." if len(response.response) > 300 else response.response,
            "source": label, "time": datetime.now().strftime("%H:%M")})
        st.session_state.question = ""
else:
    st.markdown('<p class="finrag-kicker" style="margin-top:16px;">Side-by-side comparison</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#78716C;font-family:\'DM Sans\',sans-serif;font-size:0.88em;margin-bottom:12px;">Ask the same question across two documents.</p>', unsafe_allow_html=True)
    compare_question = st.text_input("", placeholder="e.g. What are the key risk factors?", label_visibility="collapsed")
    col_a, col_b = st.columns(2)
    with col_a: doc_a = st.selectbox("Document A", ALL_DOCS, index=0)
    with col_b: doc_b = st.selectbox("Document B", ALL_DOCS, index=1)
    if compare_question and st.button("Compare documents", use_container_width=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f'<p class="finrag-kicker">{doc_a}</p>', unsafe_allow_html=True)
            with st.spinner(f"Searching {doc_a}..."): resp_a = run_query(index, compare_question, doc_a, top_k)
            score_a = resp_a.source_nodes[0].score if resp_a.source_nodes else 0
            color_a = confidence_color(score_a)
            st.markdown(f'<div style="font-size:0.75em;color:{color_a};font-family:\'DM Sans\',sans-serif;font-weight:500;margin-bottom:8px;">{round(score_a*100,1)}% confidence</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="compare-col">{resp_a.response}</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown(f'<p class="finrag-kicker">{doc_b}</p>', unsafe_allow_html=True)
            with st.spinner(f"Searching {doc_b}..."): resp_b = run_query(index, compare_question, doc_b, top_k)
            score_b = resp_b.source_nodes[0].score if resp_b.source_nodes else 0
            color_b = confidence_color(score_b)
            st.markdown(f'<div style="font-size:0.75em;color:{color_b};font-family:\'DM Sans\',sans-serif;font-weight:500;margin-bottom:8px;">{round(score_b*100,1)}% confidence</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="compare-col">{resp_b.response}</div>', unsafe_allow_html=True)
        export_text = f"FinRAG — Comparison Export\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\nQUESTION: {compare_question}\n\n{'='*60}\n{doc_a}:\n{resp_a.response}\n\n{'='*60}\n{doc_b}:\n{resp_b.response}"
        st.download_button(label="Export comparison as .txt", data=export_text,
            file_name=f"finrag_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", mime="text/plain")
