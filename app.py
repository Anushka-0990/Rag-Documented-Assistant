
import streamlit as st
import tempfile
import os

from ingest import load_and_chunk
from vectorstore import build_vectorstore
from qa_chain import build_qa_chain
from agent import build_agent

st.set_page_config(
    page_title="Document Intelligence Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        font-size: 17px;
    }

    /* Hide default Streamlit chrome for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Page background */
    .stApp {
        background: linear-gradient(180deg, #f5f3ff 0%, #fdf4ff 45%, #fff 100%);
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes pulseDot {
        0% { box-shadow: 0 0 0 0 rgba(16,185,129,0.55); }
        70% { box-shadow: 0 0 0 8px rgba(16,185,129,0); }
        100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
    }

    /* Hero header */
    .hero {
        padding: 2.2rem 2.4rem;
        border-radius: 20px;
        background: linear-gradient(120deg, #6366f1, #8b5cf6, #d946ef, #8b5cf6, #6366f1);
        background-size: 300% 300%;
        animation: gradientShift 10s ease infinite, fadeInUp 0.6s ease;
        margin-bottom: 1.8rem;
        color: white;
        box-shadow: 0 10px 30px rgba(139, 92, 246, 0.25);
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        color: white;
    }
    .hero p {
        font-size: 1.1rem;
        margin: 0;
        opacity: 0.95;
        line-height: 1.5;
    }
    .hero .badge {
        display: inline-block;
        background: rgba(255,255,255,0.22);
        padding: 6px 16px;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.9rem;
        backdrop-filter: blur(4px);
        transition: transform 0.2s ease, background 0.2s ease;
    }
    .hero .badge:hover {
        transform: translateY(-2px) scale(1.04);
        background: rgba(255,255,255,0.32);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f3f0ff 0%, #fbf5ff 100%);
        border-right: 1px solid #e5e3fa;
        font-size: 1.02rem;
    }
    section[data-testid="stSidebar"] h2 {
        font-weight: 800;
        font-size: 1.5rem;
        color: #4c1d95;
    }

    /* Card container */
    .info-card {
        background: white;
        border: 1px solid #ece9f9;
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(80, 60, 180, 0.07);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        animation: fadeInUp 0.5s ease;
    }
    .info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(124, 58, 237, 0.14);
    }
    .info-card h4 {
        margin: 0 0 0.6rem 0;
        font-size: 1.1rem;
        font-weight: 700;
        color: #3730a3;
    }
    .info-card p {
        margin: 0;
        font-size: 0.98rem;
        color: #6b7280;
        line-height: 1.7;
    }

    /* Step list in sidebar */
    .step-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 4px;
        font-size: 0.98rem;
        color: #4b5563;
        border-radius: 8px;
        transition: background 0.2s ease, transform 0.2s ease;
    }
    .step-item:hover {
        background: #ede9fe;
        transform: translateX(3px);
    }
    .step-num {
        display: flex;
        align-items: center;
        justify-content: center;
        min-width: 26px;
        height: 26px;
        border-radius: 50%;
        background: #ede9fe;
        color: #6d28d9;
        font-size: 0.82rem;
        font-weight: 700;
    }

    /* Chat bubbles */
    .stChatMessage {
        border-radius: 16px !important;
        font-size: 1.05rem !important;
        transition: transform 0.2s ease;
        animation: fadeInUp 0.35s ease;
    }
    .stChatMessage:hover {
        transform: translateX(2px);
    }

    /* Status pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #ecfdf5;
        color: #047857;
        padding: 9px 18px;
        border-radius: 999px;
        font-size: 0.95rem;
        font-weight: 700;
        border: 1px solid #a7f3d0;
        animation: fadeInUp 0.4s ease;
    }
    .status-pill::before {
        content: "";
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #10b981;
        animation: pulseDot 1.6s infinite;
    }

    /* Source chunk box */
    .source-chunk {
        background: #f9fafb;
        border-left: 4px solid #8b5cf6;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 10px;
        font-size: 0.95rem;
        color: #374151;
        transition: background 0.2s ease, border-left-width 0.2s ease;
    }
    .source-chunk:hover {
        background: #f3f0ff;
        border-left-width: 6px;
    }
    .source-chunk .chunk-label {
        font-weight: 700;
        color: #6d28d9;
        font-size: 0.85rem;
        margin-bottom: 4px;
        display: block;
    }

    /* Upload box override */
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 16px !important;
        border: 2px dashed #a78bfa !important;
        background: #faf9ff !important;
        transition: border-color 0.25s ease, background 0.25s ease, transform 0.2s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #7c3aed !important;
        background: #f3effe !important;
        transform: scale(1.005);
    }

    /* Headings and body text throughout */
    h1, h2, h3 { font-weight: 800 !important; }
    .stMarkdown p, .stMarkdown li { font-size: 1.02rem; }

    /* Buttons */
    .stButton button, button[kind] {
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stButton button:hover, button[kind]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(124, 58, 237, 0.25) !important;
    }

    /* Toggle label */
    [data-testid="stWidgetLabel"] p {
        font-size: 1.02rem !important;
        font-weight: 600 !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        font-size: 1.05rem !important;
        border-radius: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  HERO HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>📄 Document Intelligence Assistant</h1>
    <p>Ask questions about any document in plain English — grounded answers, real citations, zero hallucination guesswork.</p>
    <span class="badge">⚡ RAG-powered</span>
    <span class="badge" style="margin-left:6px;">🤖 Agentic tool routing</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## Document Intelligence")
    st.caption("RAG + Agentic GenAI Assistant")
    st.markdown("---")

    use_agent = st.toggle(
        "🤖 Enable Agent Mode",
        value=False,
        help="Lets the assistant decide between searching the document "
             "or using a calculator tool, based on your question."
    )

    st.markdown("---")

    st.markdown("""
    <div class="info-card">
        <h4>🧠 How it works</h4>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        "Upload a PDF",
        "It's split into chunks & embedded",
        "Ask a question",
        "Relevant chunks are retrieved",
        "You get a grounded, cited answer"
    ]
    for i, step in enumerate(steps, 1):
        st.markdown(f"""
        <div class="step-item">
            <div class="step-num">{i}</div>
            <div>{step}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="info-card">
        <h4>⚙️ Tech stack</h4>
        <p>LangChain · ChromaDB · Sentence-Transformers · Groq (Llama 3.1) · Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
#  MAIN AREA
# ============================================================
uploaded_file = st.file_uploader("📎 Upload a PDF", type="pdf", label_visibility="visible")

if uploaded_file:
    if st.session_state.get("current_file") != uploaded_file.name:
        with st.spinner("🔍 Reading and indexing your document... (first run takes ~30s)"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            chunks = load_and_chunk(tmp_path)
            vectorstore = build_vectorstore(chunks, persist_dir=tempfile.mkdtemp())
            qa_chain = build_qa_chain(vectorstore)

            st.session_state.qa_chain = qa_chain
            st.session_state.agent = build_agent(qa_chain)
            st.session_state.current_file = uploaded_file.name
            st.session_state.chunk_count = len(chunks)
            st.session_state.chat_history = []

            os.unlink(tmp_path)

        st.markdown(
            f'<div class="status-pill">Indexed into {st.session_state.chunk_count} chunks — ready to answer questions</div>',
            unsafe_allow_html=True
        )
        st.write("")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ---------- Render chat history ----------
    for entry in st.session_state.chat_history:
        with st.chat_message("user", avatar="🧑"):
            st.write(entry["question"])
        with st.chat_message("assistant", avatar="✨"):
            st.write(entry["answer"])
            if entry.get("sources"):
                with st.expander("📚 View sources"):
                    for i, src in enumerate(entry["sources"]):
                        st.markdown(f"""
                        <div class="source-chunk">
                            <span class="chunk-label">Chunk {i+1} · page {src['page']}</span>
                            {src['content']}
                        </div>
                        """, unsafe_allow_html=True)

    # ---------- Question input ----------
    question = st.chat_input("💬 Ask a question about your document...")

    if question:
        with st.chat_message("user", avatar="🧑"):
            st.write(question)

        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("Thinking..."):
                if use_agent:
                    response = st.session_state.agent.run(question)
                    answer_text = response
                    sources = []
                else:
                    result = st.session_state.qa_chain({"query": question})
                    answer_text = result["result"]
                    sources = [
                        {
                            "page": doc.metadata.get("page", "N/A"),
                            "content": doc.page_content[:300] + "..."
                        }
                        for doc in result["source_documents"]
                    ]

            st.write(answer_text)
            if sources:
                with st.expander("📚 View sources"):
                    for i, src in enumerate(sources):
                        st.markdown(f"""
                        <div class="source-chunk">
                            <span class="chunk-label">Chunk {i+1} · page {src['page']}</span>
                            {src['content']}
                        </div>
                        """, unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer_text,
            "sources": sources
        })

else:
    st.markdown("""
    <div class="info-card" style="text-align:center; padding: 2.5rem 1.5rem;">
        <h4 style="font-size: 1.1rem;">📤 Upload a PDF to get started</h4>
        <p>Try a policy document, research paper, or product manual — then ask anything about it.</p>
    </div>
    """, unsafe_allow_html=True)