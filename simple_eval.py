
import sys
from ingest import load_and_chunk
from vectorstore import build_vectorstore
from qa_chain import build_qa_chain

# EDIT THIS: replace with 10-15 real Q&A pairs from your own document
EVAL_SET = [
    {"question": "What is this document about?", "expected": "<fill in expected answer>"},
    {"question": "What is the main topic discussed?", "expected": "<fill in expected answer>"},
    # add more pairs specific to your document...
]


def run_eval(pdf_path):
    print(f"Loading and indexing: {pdf_path}")
    chunks = load_and_chunk(pdf_path)
    vectorstore = build_vectorstore(chunks, persist_dir="eval_chroma_db")
    qa_chain = build_qa_chain(vectorstore)

    results = []
    for item in EVAL_SET:
        result = qa_chain({"query": item["question"]})
        answer = result["result"]
        results.append({
            "question": item["question"],
            "expected": item["expected"],
            "got": answer
        })
        print("\n" + "=" * 60)
        print(f"Q: {item['question']}")
        print(f"Expected: {item['expected']}")
        print(f"Got:      {answer}")

    print("\n" + "=" * 60)
    print(f"Ran {len(results)} evaluation questions.")
    print("Manually score each as Correct / Partial / Wrong / Hallucinated")
    print("and summarize the accuracy % in your README for the interview.")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_eval.py <path_to_pdf>")
        sys.exit(1)
    run_eval(sys.argv[1])