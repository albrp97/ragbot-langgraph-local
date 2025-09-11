from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.ingest import ingest_pdfs_langchain

if __name__ == "__main__":
    ingest_pdfs_langchain(ROOT / "data" / "raw_pdfs")
