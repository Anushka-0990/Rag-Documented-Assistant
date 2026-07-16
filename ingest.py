"""
ingest.py
Loads a PDF and splits it into overlapping text chunks for embedding.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_chunk(file_path, chunk_size=800, chunk_overlap=100):
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    # Quick standalone test: python ingest.py sample.pdf
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf>")
        sys.exit(1)

    chunks = load_and_chunk(sys.argv[1])
    print(f"Loaded {len(chunks)} chunks.")
    print("---- First chunk preview ----")
    print(chunks[0].page_content[:300])