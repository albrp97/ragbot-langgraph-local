from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph.rag_graph import build_rag_graph

if __name__ == "__main__":
    question = "Enumerate the languages in most danger of extinction?"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])

    app = build_rag_graph()
    out = app.invoke({"query": question})

    print("🔎 Query:", question, "\n")
    print("💡 Answer:\n", out.get("answer", "").strip(), "\n")
    print("📚 Sources:")
    for s in out.get("sources", []):
        print(f" - {s['source']} p.{s['page']}")
