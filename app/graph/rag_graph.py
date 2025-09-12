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
from app.extract.structured import extract_to_json
from app.extract.cv_detect import detect_cv

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
    extract_results: List[Dict[str, Any]]

# --- Router ---
# --- Router: explicit commands + smart python detection ---
_CMD_PY = re.compile(r"^\s*python\b", re.I)
_CMD_EXTRACT = re.compile(r"^\s*extract\b", re.I)

# Obvious math/compute patterns
_NUMEX = re.compile(r"(?<![A-Za-z0-9_])\d+(?:\s*[\+\-\*\/\^%]\s*\d+)+")  
_LISTEX = re.compile(r"\[\s*\d+(?:\s*,\s*\d+){1,}\s*\]") 
_RAND_BETWEEN = re.compile(r"\brandom(?: number)?\s+(?:from|between)\s*\d+\s*(?:and|to)\s*\d+\b", re.I)
_DICE = re.compile(r"\b(roll|simulate)\b.*\b(die|dice)\b", re.I)
_MATH_WORDS = re.compile(
    r"\b(mean|median|mode|std|standard deviation|variance|sum|product|multiply|divide|power|exponent|sqrt|factorial|"
    r"permutation|combination|average|sort|sorted|reverse)\b",
    re.I,
)
_PY_LIST_REQ = re.compile(r"\b(make|create|build|return)\b.*\b(python )?(list|array)\b", re.I)
_NUMERIC_TASK = re.compile(r"\b(first|next|initial)\s+\d+\s+(square|cube|prime|fibonacci|even|odd)\b", re.I)

def _strip_leading_command(q: str) -> str:
    """Remove a leading 'python' or 'extract' command word, if present."""
    if _CMD_PY.match(q):
        return _CMD_PY.sub("", q, count=1).strip()
    if _CMD_EXTRACT.match(q):
        return _CMD_EXTRACT.sub("", q, count=1).strip()
    return q.strip()

def _needs_python(q: str) -> bool:
    """
    Heuristic for auto-routing to python when the query is clearly computational,
    even without the 'python' prefix (covers your listed examples & similar).
    """
    if _NUMEX.search(q):        # arithmetic expression present
        return True
    if _LISTEX.search(q):       # numeric list present (often implies compute)
        # Only push to python if the phrasing suggests a computation on the list
        if _MATH_WORDS.search(q) or "summary" in q.lower():
            return True
    if _RAND_BETWEEN.search(q): # "random number between 0 and 100"
        return True
    if _DICE.search(q):         # "simulate 5 dice rolls"
        return True
    if _MATH_WORDS.search(q):   # generic math/stat keywords
        return True
    if _PY_LIST_REQ.search(q):  # "make a python list of the first 10 square numbers"
        return True
    if _NUMERIC_TASK.search(q): # "first 10 square/prime/fibonacci..."
        return True
    return False

def route(state: RAGState) -> RAGState:
    """
    Routing rules:
      - 'extract ...' at start -> extract branch (CV JSON)
      - 'python ...'  at start -> python tool
      - otherwise, if _needs_python(query) -> python tool
      - else -> RAG
    The command word is stripped from the query before downstream nodes.
    """
    q = state.get("query", "") or ""
    if _CMD_EXTRACT.match(q):
        return {"route": "extract", "query": _strip_leading_command(q)}
    if _CMD_PY.match(q):
        return {"route": "python", "query": _strip_leading_command(q)}
    if _needs_python(q):
        return {"route": "python", "query": q.strip()}
    return {"route": "rag", "query": q.strip()}

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

# NEW
def structured_extract(state: RAGState) -> RAGState:
    """Run CV detection + structured extraction over selected PDFs."""
    allowed = state.get("allowed_sources") or []
    raw_dir = Path("data/raw_pdfs")
    out_dir = Path("data/structured/cv")
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    hits = 0
    summaries: List[Dict[str, Any]] = []
    srcs: List[Dict[str, Any]] = []

    for fname in allowed:
        pdf = raw_dir / fname
        if not pdf.exists():
            continue
        processed += 1
        det = detect_cv(pdf)
        if not det.get("is_cv", False):
            continue
        # extract & read parsed JSON back to summarize a few fields
        json_path = extract_to_json(pdf, schema_name="cv_standard", out_dir=out_dir, debug=False)
        try:
            import json
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}

        name = (data or {}).get("name")
        email = (data or {}).get("email")
        edu_count = len((data or {}).get("education") or [])  # might be None
        summaries.append({"file": fname, "name": name, "email": email, "education_count": edu_count})
        srcs.append({"source": fname, "page": "-"})
        hits += 1

    if hits == 0:
        human = (
            f"Structured extraction finished.\n\n"
            f"Processed: {processed} files\n"
            f"Extracted CVs: 0\n"
            f"Output folder: data/structured/cv\n"
            f"(No CVs detected among the selected files.)"
        )
    else:
        lines = [
            f"- **{s['file']}** — {s.get('name') or 'N/A'} — {s.get('email') or 'N/A'} — education entries: {s['education_count']}"
            for s in summaries
        ]
        human = (
            f"Structured extraction finished.\n\n"
            f"Processed: {processed} files\n"
            f"Extracted CVs: {hits}\n"
            f"Output folder: data/structured/cv\n\n"
            + "\n".join(lines)
        )

    return {"answer": human, "sources": srcs, "extract_results": summaries}

# ---- Build ----
def build_rag_graph():
    g = StateGraph(state_schema=RAGState)
    g.add_node("prepare_memory", prepare_memory)
    g.add_node("route", route)
    g.add_node("retrieve", retrieve)
    g.add_node("run_python", run_python)
    g.add_node("structured_extract", structured_extract)
    g.add_node("generate", generate_answer)

    g.set_entry_point("prepare_memory")
    g.add_edge("prepare_memory", "route")
    g.add_conditional_edges(
        "route",
        lambda s: s["route"],
        {
            "rag": "retrieve",
            "python": "run_python",
            "extract": "structured_extract",
        },
    )
    g.add_edge("retrieve", "generate")
    g.add_edge("run_python", "generate")
    # structured_extract already returns a final answer
    g.add_edge("structured_extract", END)
    g.add_edge("generate", END)
    return g.compile()
