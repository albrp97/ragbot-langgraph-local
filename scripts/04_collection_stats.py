from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from langchain_chroma import Chroma
from app.config.settings import settings
from collections import Counter

if __name__ == "__main__":
    vs = Chroma(collection_name="docs_text", persist_directory=settings.chroma_path)
    # Pull a big sample by IDs to count sources
    items = vs._collection.get(include=["metadatas"], limit=100000)
    metas = items.get("metadatas", [])
    sources = [ (m.get("source") or m.get("file_path") or "(unknown)") for m in metas ]
    cnt = Counter(Path(s).name for s in sources)
    print("Documents in collection:", sum(cnt.values()))
    print("By file:")
    for name, c in cnt.most_common():
        print(f" - {name}: {c} chunks")
