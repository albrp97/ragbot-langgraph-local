from typing import TypedDict, List, Dict, Any, Tuple
from pathlib import Path
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from app.config.settings import settings
from app.rag.retriever import get_retriever
from app.llm.chat import generate
from app.llm.memory import compress_if_needed

# ---- State ----
class RAGState(TypedDict, total=False):
    query: str
    history: List[Tuple[str, str]]  
    allowed_sources: List[str]
    memory_summary: str 
    context_docs: List[Document]
    answer: str
    sources: List[Dict[str, Any]]

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

def _build_prompt(docs: List[Document], question: str, memory: str) -> str:
    ctx = _format_context(docs)
    mem = f"\n# Conversation memory\n{memory}\n" if memory else ""
    return (
        f"{SYSTEM_PROMPT}\n"
        f"{mem}\n"
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

    if not docs:
        return {"answer": "I don't know based on the provided documents.", "sources": []}

    prompt = _build_prompt(docs, question, memory)
    text = generate(prompt, thinking=False)
    sources = _collect_sources(docs)
    return {"answer": text.strip(), "sources": sources}

# ---- Build ----
def build_rag_graph():
    g = StateGraph(state_schema=RAGState)
    g.add_node("prepare_memory", prepare_memory)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate_answer)
    g.set_entry_point("prepare_memory")
    g.add_edge("prepare_memory", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()
