
import os
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_groq import ChatGroq

load_dotenv()


def build_agent(qa_chain):
    

    def doc_search(query: str) -> str:
        result = qa_chain({"query": query})
        return result["result"]

    def calculator(expression: str) -> str:
        try:
            # NOTE: eval() is used here for a portfolio demo only.
            # In production, use a safe math parser (e.g. `numexpr` or `asteval`)
            # instead of eval() to avoid arbitrary code execution risk.
            allowed = set("0123456789+-*/(). ")
            if not set(expression) <= allowed:
                return "Invalid expression: only numbers and + - * / ( ) allowed"
            return str(eval(expression))
        except Exception as e:
            return f"Could not calculate: {e}"

    tools = [
        Tool(
            name="DocumentSearch",
            func=doc_search,
            description=(
                "Use this to answer any question about the content of the "
                "uploaded document. Input should be a natural language question."
            )
        ),
        Tool(
            name="Calculator",
            func=calculator,
            description=(
                "Use this ONLY for pure arithmetic calculations, "
                "e.g. '1500 * 0.18'. Input should be a math expression."
            )
        )
    ]

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent