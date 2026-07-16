"""
vectorstore.py
Builds a Chroma vector store from document chunks using a local
sentence-transformer embedding model (free, no API cost).
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def build_vectorstore(chunks, persist_dir="chroma_db"):
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vectorstore


def load_vectorstore(persist_dir="chroma_db"):
    """Reload a previously persisted vector store from disk."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)