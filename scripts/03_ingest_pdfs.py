from __future__ import annotations
from pathlib import Path
import sys

# Add repo root to sys.path so `app.*` imports work when running from /scripts
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.ingest import check_and_ingest_stream  # uses the streaming updates

def _pct(i: int, n: int) -> int:
    try:
        return int(round(100 * (i / max(n, 1))))
    except Exception:
        return 0

def main():
    # Allow optional CLI arg: scripts/ingest_pdfs.py [raw_dir]
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (ROOT / "data" / "raw_pdfs")
    raw_dir = raw_dir.resolve()
    print(f"📂 Raw PDFs directory: {raw_dir}")

    for upd in check_and_ingest_stream(raw_dir):
        if isinstance(upd, dict):
            i = int(upd.get("i", 0))
            n = int(upd.get("n", 1))
            msg = str(upd.get("msg", "")).strip()
            fname = str(upd.get("file") or "").strip()
            pct = _pct(i, n)
            if fname:
                print(f"[{i}/{n} {pct:3d}%] {msg}: {fname}")
            else:
                print(f"[{i}/{n} {pct:3d}%] {msg}")
        else:
            # Fallback for string messages
            print(str(upd))

    print("✅ Done.")

if __name__ == "__main__":
    main()
