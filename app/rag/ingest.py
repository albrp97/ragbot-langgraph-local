from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Iterable, Tuple
import hashlib, json, shutil, datetime
import fitz
from PIL import Image
import io, torch
from transformers import pipeline

from app.config.settings import settings

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ---------- Manifest utils ----------
# manifest used to track ingested files, versions, etc.

def _manifest_path() -> Path:
    return Path(settings.chroma_path) / ".manifest.json"

def _utc_now() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _file_sha256(p: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b: break
            h.update(b)
    return h.hexdigest()

def _read_manifest() -> Dict:
    mp = _manifest_path()
    if mp.exists():
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _write_manifest(m: Dict) -> None:
    p = _manifest_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------- Embeddings / splitter ----------
def _embeddings():
    device = "cuda" if (settings.device in ("cuda", "auto")) else "cpu"
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_id,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": settings.embedding_batch_size,
        },
    )

def _splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

# ---------- Image captioner (lazy-loaded) ----------
_CAPTION_MODEL = None
_CAPTION_PROCESSOR = None
_CAPTION_PIPE = None
# could be moved to env
# limit to images larger than ~40KB
MIN_IMAGE_BYTES = 40_000

def _ensure_images_dir() -> Path:
    p = Path(settings.images_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _load_captioner():
    global _CAPTION_MODEL, _CAPTION_PROCESSOR, _CAPTION_PIPE
    if _CAPTION_PIPE is not None:
        return

    device = 0 if (settings.device in ("cuda", "auto") and torch.cuda.is_available()) else -1
    _CAPTION_PIPE = pipeline("image-to-text", model=settings.image_retrieval_id, device=device)
    _CAPTION_MODEL = None
    _CAPTION_PROCESSOR = None

# writes JSON for an image
# returns the JSON path
def _write_image_meta_json(
    img_path: Path,
    pdf_path: Path,
    page_no: int,
    caption_short: str | None,
    description: str | None,
    bytes_len: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> Path:
    """
    Write a sidecar JSON with metadata + caption + description for a single extracted image.
    Returns the JSON path.
    """
    meta = {
        "image_id": img_path.name,
        "image_path": str(img_path),
        "file_name": img_path.name,
        "source_pdf": pdf_path.name,
        "source_pdf_path": str(pdf_path),
        "page": page_no,
        "bytes": bytes_len,
        "width": width,
        "height": height,
        "caption": (caption_short or description or ""),   # short if available
        "description": (description or caption_short or ""),  # rich VL output
        "modality": "image",
        "content_type": "image_caption",
        "caption_model_id": settings.image_retrieval_id,
        "created": _utc_now(),
    }
    json_path = img_path.with_suffix(img_path.suffix + ".json")
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return json_path


def _caption_image(img_path: Path) -> tuple[str, str | None]:
    """
    Return (description, caption_short) using a generic image-to-text pipeline.
    - description: an expanded, richer description for retrieval
    - caption_short: the raw short caption from the vision model
    """
    _load_captioner()
    try:
        out = _CAPTION_PIPE(str(img_path), max_new_tokens=min(256, settings.llm_max_new_tokens))
        short = (out[0].get("generated_text", "") if out else "").strip()
    except Exception:
        short = ""

    # Expand the short caption into a richer description (best-effort)
    description = short
    if short:
        try:
            from app.llm.chat import generate
            description = generate(
                "Expand this short caption into a detailed, strictly factual 3-6 sentence description. "
                "Include colors, objects, spatial relations, actions, and any visible text; avoid hallucinations.\n\n"
                f"Short caption: {short}\n\nDetailed description:",
                max_new_tokens=min(256, settings.llm_max_new_tokens),
            ).strip() or short
        except Exception:
            description = short

    return description, (short or None)

# Extract images from a PDF and save them under settings.images_dir.
# returns a list of dicts with info about each saved image.
def _extract_images_for_file(pdf: Path) -> List[dict]:
    """
    Extract images from a PDF and save them under settings.images_dir.
    Returns a list of dicts:
      { "path": Path, "page": int, "bytes": int, "width": int, "height": int, "ext": str }
    Skips images smaller than MIN_IMAGE_BYTES.
    """
    out_dir = _ensure_images_dir()
    saved: List[dict] = []
    try:
        doc = fitz.open(str(pdf))
    except Exception:
        return saved

    stem = pdf.stem
    for pno in range(len(doc)):
        page = doc[pno]
        try:
            images = page.get_images(full=True)
        except Exception:
            images = []
        for idx, img in enumerate(images, start=1):
            xref = img[0]
            try:
                base = doc.extract_image(xref)
                ext = (base.get("ext") or "png").lower()
                img_bytes = base.get("image")
                width = base.get("width")
                height = base.get("height")
                if not img_bytes or len(img_bytes) < MIN_IMAGE_BYTES:
                    continue
                name = f"{stem}_p{pno+1}_img{idx}.{ext}"
                path = out_dir / name
                with open(path, "wb") as f:
                    f.write(img_bytes)
                saved.append({
                    "path": path,
                    "page": pno + 1,
                    "bytes": len(img_bytes),
                    "width": width,
                    "height": height,
                    "ext": ext,
                })
            except Exception:
                continue
    doc.close()
    return saved

# ---------- Core ingestion helpers ----------

# Use a generator to process one PDF at a time with progress updates
# yields events for Gradio progress
def _process_one_pdf_stream(vs: Chroma, pdf: Path) -> Iterable[dict]:
    added_total = 0

    # ---- Images → captions
    entries = _extract_images_for_file(pdf)  # list of dicts
    img_n = len(entries)
    if img_n > 0:
        caption_docs: list[Document] = []
        for j, ent in enumerate(entries, start=1):
            img_path: Path = ent["path"]
            page_no: int = ent["page"]
            bytes_len = ent.get("bytes")
            width = ent.get("width")
            height = ent.get("height")

            try:
                description, short = _caption_image(img_path)
                description = (description or "").strip()
                short = (short or "").strip() or None
            except Exception:
                description, short = "", None

            # sidecar JSON (includes both caption + description)
            _write_image_meta_json(
                img_path=img_path,
                pdf_path=pdf,
                page_no=page_no,
                caption_short=short,
                description=description,
                bytes_len=bytes_len,
                width=width,
                height=height,
            )

            if description:
                meta = {
                    "source_name": pdf.name,
                    "source": str(pdf),
                    "page": page_no,
                    "modality": "image",
                    "image_path": str(img_path),
                    "image_id": img_path.name,
                    "content_type": "image_caption",
                }
                # Store the *rich* description in the vector store
                caption_docs.append(Document(page_content=description, metadata=meta))

            # Stream: show the (rich) description snippet
            snippet = (description[:160] + "…") if len(description) > 160 else description
            yield {
                "phase": "images",
                "file": pdf.name,
                "img_i": j,
                "img_n": img_n,
                "image": img_path.name,
                "caption": snippet,
            }

        if caption_docs:
            vs.add_documents(caption_docs)
            added_total += len(caption_docs)

    # ---- Text chunks
    docs = _load_docs_for_file(pdf)
    splitter = _splitter()
    docs = splitter.split_documents(docs)
    chunks = len(docs)
    if docs:
        vs.add_documents(docs)
        added_total += chunks
    yield {"phase": "text", "file": pdf.name, "chunks": chunks}

    # ---- Final per-file summary
    yield {"phase": "file_done", "file": pdf.name, "added_total": added_total}

# Returns list of Documents for a given PDF file
def _load_docs_for_file(pdf: Path):
    loader = PyMuPDFLoader(str(pdf))
    docs = loader.load()
    # Normalize metadata to include a stable short name for filtering/deleting
    short = pdf.name
    for d in docs:
        d.metadata = dict(d.metadata or {})
        d.metadata["source_name"] = short
    return docs

# returns a Chroma vector store instance
def _vectordb(collection_name: str = "docs_text") -> Chroma:
    return Chroma(
        collection_name=collection_name,
        persist_directory=settings.chroma_path,
        embedding_function=_embeddings(),
    )

def _delete_file_from_vs(vs: Chroma, short_name: str):
    # Delete all chunks from a given file (by our stable 'source_name' key)
    try:
        vs.delete(where={"source_name": short_name})
    except Exception:
        # best effort
        pass

# Ingest a list of PDF files into the vector store
# returns number of chunks added per file
def _ingest_files(vs: Chroma, files: List[Path]) -> Dict[str, int]:
    """Return {short_name: n_chunks} counting text + image-caption docs."""
    splitter = _splitter()
    added: Dict[str, int] = {}
    for pdf in files:
        total_for_file = 0

        # ---- Text chunks
        docs = _load_docs_for_file(pdf)
        docs = splitter.split_documents(docs)
        if docs:
            vs.add_documents(docs)
            total_for_file += len(docs)

        # ---- Image captions -> documents (and JSON sidecars)
        img_entries = _extract_images_for_file(pdf)  # list of dicts
        if img_entries:
            caption_docs: List[Document] = []
            for ent in img_entries:
                img_path: Path = ent["path"]
                page_no: int = ent["page"]
                bytes_len = ent.get("bytes")
                width = ent.get("width")
                height = ent.get("height")

                try:
                    description, short = _caption_image(img_path)
                    description = (description or "").strip()
                    short = (short or "").strip() or None
                except Exception:
                    description, short = "", None

                # sidecar JSON
                _write_image_meta_json(
                    img_path=img_path,
                    pdf_path=pdf,
                    page_no=page_no,
                    caption_short=short,
                    description=description,
                    bytes_len=bytes_len,
                    width=width,
                    height=height,
                )

                if not description:
                    continue
                meta = {
                    "source_name": pdf.name,
                    "source": str(pdf),                # full path to PDF
                    "page": page_no,
                    "modality": "image",
                    "image_path": str(img_path),
                    "image_id": img_path.name,
                    "content_type": "image_caption",
                }
                # Store the rich description in the vector store
                caption_docs.append(Document(page_content=description, metadata=meta))

            if caption_docs:
                vs.add_documents(caption_docs)
                total_for_file += len(caption_docs)

        if total_for_file:
            added[pdf.name] = total_for_file

    return added


# ---------- Public: plan + execute with progress ----------
# Check if ingestion is needed (files changed, embedding changed, etc.)
# returns changes bool and a report dict
def check_and_ingest_if_needed(raw_dir: Path, collection_name: str = "docs_text") -> Tuple[bool, Dict]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest()
    current_embed = settings.embedding_id
    current_caption_model = getattr(settings, "image_retrieval_id", "")

    pdfs = sorted([p for p in raw_dir.glob("*.pdf") if p.is_file()])
    current = {p.name: {"sha256": _file_sha256(p), "path": str(p)} for p in pdfs}

    prev_embed = manifest.get("embedding_id")
    prev_caption_model = manifest.get("caption_model_id")
    full_rebuild = (prev_embed != current_embed) or (prev_caption_model != current_caption_model)

    report = {"full_rebuild": full_rebuild, "added": [], "changed": [], "removed": [], "kept": []}

    if full_rebuild:
        # wipe vector dir (safer) then re-ingest everything
        if Path(settings.chroma_path).exists():
            shutil.rmtree(settings.chroma_path, ignore_errors=True)
        report["added"] = list(current.keys())
        return True, report

    # Otherwise compute diffs
    prev_files: Dict[str, Dict] = manifest.get("files", {})
    prev_set = set(prev_files.keys())
    curr_set = set(current.keys())

    removed = sorted(list(prev_set - curr_set))
    added = sorted(list(curr_set - prev_set))
    kept = sorted(list(curr_set & prev_set))

    changed = [name for name in kept if prev_files.get(name, {}).get("sha256") != current[name]["sha256"]]

    report["removed"] = removed
    report["added"] = added
    report["changed"] = changed
    report["kept"] = sorted(list(set(kept) - set(changed)))

    any_change = bool(removed or added or changed)
    return any_change, report

# Execute plan from check_and_ingest_if_needed (no streaming).
# Returns updated manifest dict.
def run_ingest_plan(raw_dir: Path, report: Dict, collection_name: str = "docs_text") -> Dict:
    """
    Execute plan from check_and_ingest_if_needed (no streaming).
    Returns updated manifest dict.
    """
    raw_dir = Path(raw_dir)
    manifest = {"embedding_id": settings.embedding_id, "updated": _utc_now(), "files": {}}
    vs = _vectordb(collection_name)

    # Remove deleted/changed files from VS
    for name in report.get("removed", []):
        _delete_file_from_vs(vs, name)
    for name in report.get("changed", []):
        _delete_file_from_vs(vs, name)

    # Ingest added+changed (after delete)
    to_ingest = [raw_dir / n for n in (report.get("added", []) + report.get("changed", []))]
    added_counts = _ingest_files(vs, to_ingest)

    # Keep files that were already OK
    kept = report.get("kept", [])

    # Build new manifest
    for name in kept + list(added_counts.keys()):
        p = raw_dir / name
        if p.exists():
            manifest["files"][name] = {
                "sha256": _file_sha256(p),
                "chunks": added_counts.get(name, None),  # None for kept (unknown count)
                "last_ingested": _utc_now() if name in added_counts else _read_manifest().get("files", {}).get(name, {}).get("last_ingested", _utc_now()),
            }

    _write_manifest(manifest)
    return manifest


# ---------- UI-friendly: generator with progress ----------
# Yields progress updates for Gradio.
def check_and_ingest_stream(raw_dir: Path, collection_name: str = "docs_text") -> Iterable[dict | str]:
    """
    Yields progress updates for Gradio.
    Events may include:
      {"msg": "...", "file": "name.pdf", "i": current_file_idx, "n": total_files}
      {"phase":"images", "file":"...", "img_i":j, "img_n":m, "image":"...", "caption":"..."}
      {"phase":"text",   "file":"...", "chunks":K}
      {"phase":"file_done","file":"...", "added_total":T}
    """
    raw_dir = Path(raw_dir)
    yield {"msg": "Checking documents and embedding version…", "file": "", "i": 0, "n": 1}

    changed, plan = check_and_ingest_if_needed(raw_dir, collection_name)

    # ---------- Full rebuild ----------
    if plan.get("full_rebuild"):
        if Path(settings.chroma_path).exists():
            yield {"msg": "Embedding changed → clearing vector store…", "file": "", "i": 0, "n": 1}
            shutil.rmtree(settings.chroma_path, ignore_errors=True)

        files = sorted([p for p in raw_dir.glob("*.pdf") if p.is_file()])
        if not files:
            _write_manifest({"embedding_id": settings.embedding_id, "updated": _utc_now(), "files": {}})
            yield {"msg": "No PDFs found in data/raw_pdfs.", "file": "", "i": 1, "n": 1}
            return

        vs = _vectordb(collection_name)
        counts: Dict[str, int] = {}
        n = len(files)
        for idx, pdf in enumerate(files, start=1):
            # update file-level (PDF) progress bar
            yield {"msg": "Indexing", "file": pdf.name, "i": idx, "n": n}
            added_total = 0
            for ev in _process_one_pdf_stream(vs, pdf):
                # pass through with outer file progress
                if isinstance(ev, dict):
                    ev.setdefault("i", idx)
                    ev.setdefault("n", n)
                yield ev
                if isinstance(ev, dict) and ev.get("phase") == "file_done":
                    added_total = int(ev.get("added_total", 0))
            counts[pdf.name] = added_total

        # Write fresh manifest
        manifest = {
            "embedding_id": settings.embedding_id,
            "caption_model_id": settings.image_retrieval_id,
            "updated": _utc_now(),
            "files": {}
        }
        for p in files:
            manifest["files"][p.name] = {
                "sha256": _file_sha256(p),
                "chunks": counts.get(p.name, 0),
                "last_ingested": _utc_now(),
            }
        _write_manifest(manifest)
        yield {"msg": "Rebuild complete.", "file": "", "i": n, "n": n}
        return

    # ---------- Incremental ----------
    removed = plan.get("removed", []) or []
    added = plan.get("added", []) or []
    changed_files = plan.get("changed", []) or []
    kept = plan.get("kept", []) or []

    if not (removed or added or changed_files):
        yield {"msg": "All documents are up to date.", "file": "", "i": 1, "n": 1}
        return

    prev_manifest = _read_manifest()
    prev_files = prev_manifest.get("files", {}) if isinstance(prev_manifest, dict) else {}

    vs = _vectordb(collection_name)

    ops_total = len(removed) + len(added) + len(changed_files)
    step = 0

    # 1) Remove deleted/changed
    if removed:
        for name in removed:
            step += 1
            yield {"msg": "Removing from index", "file": name, "i": step, "n": ops_total}
            _delete_file_from_vs(vs, name)
    if changed_files:
        for name in changed_files:
            step += 1
            yield {"msg": "Removing outdated chunks", "file": name, "i": step, "n": ops_total}
            _delete_file_from_vs(vs, name)

    # 2) Ingest added + changed with streaming (images + text)
    current_counts: Dict[str, int] = {}
    todo = added + changed_files
    n_files = len(todo) if todo else 1
    for idx, name in enumerate(todo, start=1):
        pdf = raw_dir / name
        yield {"msg": "Indexing", "file": name, "i": idx, "n": n_files}
        if not pdf.exists():
            continue
        added_total = 0
        for ev in _process_one_pdf_stream(vs, pdf):
            if isinstance(ev, dict):
                ev.setdefault("i", idx)
                ev.setdefault("n", n_files)
            yield ev
            if isinstance(ev, dict) and ev.get("phase") == "file_done":
                added_total = int(ev.get("added_total", 0))
        current_counts[name] = added_total

    # 3) Write new manifest (kept + newly ingested; removed omitted)
    manifest = {"embedding_id": settings.embedding_id, "updated": _utc_now(), "files": {}}
    current_files = sorted([p for p in raw_dir.glob("*.pdf") if p.is_file()])
    for p in current_files:
        name = p.name
        if name in current_counts: 
            manifest["files"][name] = {
                "sha256": _file_sha256(p),
                "chunks": current_counts.get(name, 0),
                "last_ingested": _utc_now(),
            }
        elif name in kept:
            prev = prev_files.get(name, {})
            manifest["files"][name] = {
                "sha256": _file_sha256(p),
                "chunks": prev.get("chunks"),
                "last_ingested": prev.get("last_ingested", _utc_now()),
            }
        else:
            prev = prev_files.get(name, {})
            manifest["files"][name] = {
                "sha256": _file_sha256(p),
                "chunks": prev.get("chunks"),
                "last_ingested": prev.get("last_ingested", _utc_now()),
            }

    _write_manifest(manifest)
    yield {"msg": f"Index updated. {len(manifest.get('files', {}))} files tracked.", "file": "", "i": 1, "n": 1}



# ---------- Document management helpers ----------

# returns sorted PDF names from manifest; fall back to raw_pdfs dir if manifest empty.
def list_tracked_files() -> List[str]:
    """Return sorted PDF names from manifest; fall back to raw_pdfs dir if manifest empty."""
    m = _read_manifest()
    files = sorted(list((m.get("files") or {}).keys()))
    if files:
        return files
    raw = Path("data/raw_pdfs")
    return sorted([p.name for p in raw.glob("*.pdf") if p.is_file()])

# Save uploaded files into raw_pdfs; returns the short names copied.
def save_uploaded_pdfs(tmp_paths: List[str | Path], dest_dir: Path | str = "data/raw_pdfs") -> List[str]:
    """Copy uploaded files into raw_pdfs; returns the short names copied."""
    dest = Path(dest_dir); dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for p in tmp_paths:
        src = Path(p)
        if src.suffix.lower() != ".pdf":
            continue
        tgt = dest / src.name
        shutil.copy2(src, tgt)
        saved.append(src.name)
    return saved