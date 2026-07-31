

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from report_agent import generate_report
from retriever_agent import vector_search

TOP_K = 10


# ---  state  the agents --------
class State(TypedDict):
    query: str
    snippets: list[dict]
    report: str


# --- node 1: Data Retriever ------
def retriever_node(state: State) -> dict:
    snippets = vector_search(state["query"], k=TOP_K)
    print(f"[Data Retriever] {len(snippets)} snippets: "
          + ", ".join(f"{s['programming_name']} ({s['similarity_score']})" for s in snippets))
    return {"snippets": snippets}


# --- node 2: Report Generator --------
def reporter_node(state: State) -> dict:
    print("[Report Generator] synthesizing ...")
    return {"report": generate_report(state["query"], state["snippets"])}


# --- build the graph ----------------------------------------------------
builder = StateGraph(State)
builder.add_node("retriever", retriever_node)
builder.add_node("reporter", reporter_node)
builder.add_edge(START, "retriever")
builder.add_edge("retriever", "reporter")
builder.add_edge("reporter", END)

graph = builder.compile()


# --- run agents------------------------------------
print("\nAsk a question about programming languages \n")
while True:
    try:
        question = input("? ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not question:
        continue

    result = graph.invoke({"query": question})
    print("\n" + result["report"] + "\n")
