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


def _cap(s: str, limit: int = 200) -> str:
    s = (s or "").strip()
    return s if len(s) <= limit else s[:limit] + "…"


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

            phase = upd.get("phase")

            # --- Image phase: print image progress + caption ---
            if phase == "images":
                img_i = int(upd.get("img_i", 0))
                img_n = int(upd.get("img_n", 1)) or 1
                image_name = upd.get("image") or ""
                caption = _cap(upd.get("caption") or "", 300)
                ipct = _pct(img_i, img_n)
                head = f"[{i}/{n} {pct:3d}%] [img {img_i}/{img_n} {ipct:3d}%]"
                tail = f"{fname} :: {image_name} — {caption}" if image_name else f"{fname} — {caption}"
                print(f"{head} {tail}")
                continue

            # --- Text phase: show how many chunks were added for this PDF ---
            if phase == "text":
                chunks = int(upd.get("chunks", 0))
                print(f"[{i}/{n} {pct:3d}%] Text chunks for {fname}: {chunks}")
                continue

            # --- Per-file summary ---
            if phase == "file_done":
                added_total = int(upd.get("added_total", 0))
                print(f"[{i}/{n} {pct:3d}%] ✅ Finished {fname} — added {added_total} items (images+text).")
                continue

            # --- Generic progress line (planning, removing, etc.) ---
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
