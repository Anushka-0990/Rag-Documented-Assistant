

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

PROMPT_TEMPLATE = """You are a helpful assistant answering questions based ONLY on the provided context.
If the answer isn't in the context, say "I don't have enough information to answer that."
Be concise and cite which part of the document supports your answer.

Context:
{context}

Question: {question}

Answer:"""


def build_qa_chain(vectorstore, k=4):
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add it to your .env file. "
            "Get a free key at https://console.groq.com"
        )

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain