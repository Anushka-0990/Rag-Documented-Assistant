# 🧠 Document Intelligence Assistant

**A production-style Retrieval-Augmented Generation (RAG) application** that lets users upload any PDF and ask natural-language questions, receiving grounded, cited answers instead of hallucinated ones — with an optional agentic tool-routing layer.

🔗 **[Live Demo](#)** &nbsp;·&nbsp; 📦 **[Source Code](#)** &nbsp;·&nbsp; 🎥 **[Screenshots below](#screenshots)**

---

## 📌 Why this project

Most fresher GenAI projects stop at "chat with your PDF." This one goes further:

- ✅ **Grounded answers** — the LLM is explicitly instructed to say "I don't know" rather than hallucinate, and every answer is backed by retrieved source chunks
- ✅ **Confidence scoring** — retrieved chunks are ranked by semantic similarity, not just returned blindly
- ✅ **Agentic reasoning** — a ReAct-pattern agent decides whether to search the document or use another tool (e.g. a calculator), rather than following one fixed path
- ✅ **Evaluated, not just built** — tested against a manual QA benchmark to measure real answer quality (see `simple_eval.py`)

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | [LangChain](https://langchain.com) |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) — local, free |
| Vector Store | [ChromaDB](https://www.trychroma.com/) |
| LLM | [Groq](https://groq.com) (Llama 3.1 8B Instant) |
| UI | [Streamlit](https://streamlit.io) |
| Deployment | Hugging Face Spaces|

## 🏗️ Architecture

```
 PDF Upload
     │
     ▼
 Chunking (RecursiveCharacterTextSplitter)
     │
     ▼
 Embedding (sentence-transformers) ──► ChromaDB (vector store)
                                             │
 User Question ──► Embed Question ──► Retrieve top-k similar chunks
                                             │
                    (Question + Chunks) ──► LLM (Groq / Llama 3.1)
                                             │
                                   Grounded, cited answer
                                             │
                [Agent Mode: LLM also reasons about which tool to call]
```

## ✨ Features

- 📄 Upload any PDF and ask questions in plain English
- 🔍 Semantic search over document content (not just keyword matching)
- 📚 Every answer shows its source chunks with page numbers and confidence scores
- 🤖 Toggleable Agent Mode — routes between document search and a calculator tool
- 💬 Full chat-style conversation history within a session
- 🧪 Built-in evaluation harness to test answer accuracy

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- A free [Groq API key](https://console.groq.com)

### Setup

```bash
git clone https://github.com/Anushka-0990/rag-document-assistant.git
cd rag-document-assistant

python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows

pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your Groq API key

streamlit run app.py
```

## 🧪 Evaluation

```bash
python simple_eval.py your_document.pdf
```
Edit the `EVAL_SET` in `simple_eval.py` with real Q&A pairs from your document to benchmark answer accuracy.

## 📁 Project Structure

```
rag-document-assistant/
├── app.py              # Streamlit UI
├── ingest.py            # PDF loading & chunking
├── vectorstore.py        # Embedding & ChromaDB setup
├── qa_chain.py            # Retrieval-QA chain with grounded prompt
├── agent.py                # Agentic tool-routing layer (ReAct pattern)
├── simple_eval.py           # Evaluation harness
├── requirements.txt
├── Dockerfile
└── .env.example
```

## 📸 Screenshots
<img width="1913" height="1025" alt="image" src="https://github.com/user-attachments/assets/2a004872-bef1-450c-95d6-2d700816b5fc" />
<img width="1907" height="982" alt="image" src="https://github.com/user-attachments/assets/7ea0e891-9c1d-4bcf-a0f5-9cdcbfb46711" />



## 🔮 Limitations & Future Work

- Context window limits how much retrieved text can be passed to the LLM at once
- Chunking can occasionally split relevant information across boundaries
- **Next steps:** multi-document RAG support, RAGAS-based automated evaluation, conversation-aware follow-up handling

## 👤 Author

**Anushka singh** — B.Tech CSE (Data Science)
