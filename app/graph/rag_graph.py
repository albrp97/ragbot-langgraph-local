from typing import TypedDict, List, Dict, Any, Tuple
import re
from pathlib import Path
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from app.config.settings import settings
from app.rag.retriever import get_retriever
from app.llm.chat import generate
from app.llm.memory import compress_if_needed
from app.tools.plan_and_run import run_python_for_question

# ---- State ----
class RAGState(TypedDict, total=False):
    query: str
    history: List[Tuple[str, str]]  
    allowed_sources: List[str]
    memory_summary: str 
    context_docs: List[Document]
    python: Dict[str, Any]
    route: str
    answer: str
    sources: List[Dict[str, Any]]

# regex patterns for detecting simple math expressions and lists
_NUMEX = re.compile(r"\b\d+(\s*[\+\-\*\/\^]\s*\d+)+\b")
_LISTEX = re.compile(r"\[\s*\d+(?:\s*,\s*\d+){1,}\s*\]")

# --- Simple Router ---
def _needs_python(q: str) -> bool:
    ql = q.lower()
    keywords = (
        "calculate", "mean", "median", "mode", "std", "sum", "min", "max",
        "list", "array", "random", "number", "integer", "float", "math", "statistics",
        "python", "code", "script", "evaluate", "compute", "compute the following",
        "add", "subtract", "multiply", "divide", "power", "exponent", "average",
        "variance", "standard deviation", "sort", "sorted", "reverse", "shuffle",
        "randomly", "randomize", "pick", "select", "choose",
    )
    if any(k in ql for k in keywords):
        return True
    if _NUMEX.search(q) or _LISTEX.search(q):
        return True
    return False

def route(state: RAGState) -> RAGState:
    return {"route": "python" if _needs_python(state["query"]) else "rag"}


# ---- Helpers ----
def _format_context(docs: List[Document]) -> str:
    lines = []
    for d in docs:
        m = d.metadata or {}
        src = m.get("source") or m.get("file_path") or "(unknown)"
        page = m.get("page", "-")
        text = " ".join(d.page_content.split())
        text = (text[:900] + "…") if len(text) > 900 else text
        lines.append(f"[{Path(src).name} p.{page}] {text}")
    return "\n\n".join(lines)

def _collect_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    out, seen = [], set()
    for d in docs:
        m = d.metadata or {}
        src = m.get("source") or m.get("file_path") or "(unknown)"
        page = m.get("page", "-")
        key = (src, page)
        if key in seen: 
            continue
        out.append({"source": Path(src).name, "page": page})
        seen.add(key)
    return out

SYSTEM_PROMPT = (
    "You are a helpful assistant for question answering over provided documents. "
    "Use ONLY the given context and, if present, the conversation memory. "
    "If the answer isn't in the context, say you don't know."
)

# Build the full prompt for LLM
def _build_prompt(docs: List[Document], question: str, memory: str, python: Dict[str, Any] | None = None) -> str:
    ctx = _format_context(docs)
    mem = f"\n# Conversation memory\n{memory}\n" if memory else ""
    py = ""
    if python:
        if python.get("ok", False):
            py = (
                "\n# Python result\n"
                f"code:\n{python.get('code','')}\n"
                f"stdout:\n{(python.get('stdout','') or '').strip()}\n"
                f"result: {python.get('result')}\n"
            )
        else:
            py = (
                "\n# Python result (error)\n"
                f"{python.get('error','')}\n"
                f"code:\n{python.get('code','')}\n"
            )
    return (
        f"{SYSTEM_PROMPT}\n"
        f"{mem}"
        f"{py}"
        f"# Context\n{ctx}\n\n"
        f"# Question\n{question}\n\n"
        f"# Answer (concise):"
    )


# ---- Nodes ----
def prepare_memory(state: RAGState) -> RAGState:
    history = state.get("history", [])
    mem = compress_if_needed(history)
    return {"memory_summary": mem}

def retrieve(state: RAGState) -> RAGState:
    query = state["query"]
    allowed = state.get("allowed_sources")
    retriever = get_retriever(k=settings.retriever_k, allowed_sources=allowed)
    docs = retriever.invoke(query)
    return {"context_docs": docs}

def generate_answer(state: RAGState) -> RAGState:
    docs = state.get("context_docs", [])
    question = state["query"]
    memory = state.get("memory_summary", "")
    py = state.get("python")  

    # if there are no docs and no python result, we can't answer
    if not docs and not py:
        return {"answer": "I don't know based on the provided inputs.", "sources": []}

    prompt = _build_prompt(docs, question, memory, python=py) 
    text = generate(prompt, thinking=False)
    sources = _collect_sources(docs)

    if py:  # add Python tool as a source if it was used
        sources.append({"source": "Python tool", "page": "-"})

    return {"answer": text.strip(), "sources": sources}


def run_python(state: RAGState) -> RAGState:
    res = run_python_for_question(state["query"], timeout_s=3.0)
    return {"python": res}

# ---- Build ----
def build_rag_graph():
    g = StateGraph(state_schema=RAGState)
    g.add_node("prepare_memory", prepare_memory)
    g.add_node("route", route)          
    g.add_node("retrieve", retrieve)
    g.add_node("run_python", run_python)
    g.add_node("generate", generate_answer)

    g.set_entry_point("prepare_memory")
    g.add_edge("prepare_memory", "route")  
    g.add_conditional_edges("route", lambda s: s["route"], {"rag": "retrieve", "python": "run_python"})
    g.add_edge("retrieve", "generate")
    g.add_edge("run_python", "generate")
    g.add_edge("generate", END)
    return g.compile()
