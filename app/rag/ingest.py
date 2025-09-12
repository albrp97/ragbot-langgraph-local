from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Iterable, Tuple
import hashlib, json, shutil, datetime

from app.config.settings import settings

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ---------- Manifest utils ----------
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


# ---------- Core ingestion helpers ----------
def _load_docs_for_file(pdf: Path):
    loader = PyMuPDFLoader(str(pdf))
    docs = loader.load()
    # Normalize metadata to include a stable short name for filtering/deleting
    short = pdf.name
    for d in docs:
        d.metadata = dict(d.metadata or {})
        d.metadata["source_name"] = short
    return docs

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

def _ingest_files(vs: Chroma, files: List[Path]) -> Dict[str, int]:
    """Return {short_name: n_chunks} for reporting/manifest."""
    splitter = _splitter()
    added: Dict[str, int] = {}
    for pdf in files:
        docs = _load_docs_for_file(pdf)
        docs = splitter.split_documents(docs)
        if docs:
            vs.add_documents(docs)
            added[pdf.name] = len(docs)
    return added


# ---------- Public: plan + execute with progress ----------
def check_and_ingest_if_needed(raw_dir: Path, collection_name: str = "docs_text") -> Tuple[bool, Dict]:
    """
    Non-UI helper: returns (changed, report) without printing.
    changed=True if we modified the index.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest()
    current_embed = settings.embedding_id

    # Collect current files and hashes
    pdfs = sorted([p for p in raw_dir.glob("*.pdf") if p.is_file()])
    current = {p.name: {"sha256": _file_sha256(p), "path": str(p)} for p in pdfs}

    # Decide if full rebuild is needed
    prev_embed = manifest.get("embedding_id")
    full_rebuild = (prev_embed is None) or (prev_embed != current_embed)

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
def check_and_ingest_stream(raw_dir: Path, collection_name: str = "docs_text") -> Iterable[dict | str]:
    """
    Yields progress updates for Gradio as dicts:
      {"msg": "...", "file": "name.pdf", "i": current_step, "n": total_steps}
    UI shows a progress bar from i/n and the current file being processed.
    """
    raw_dir = Path(raw_dir)
    yield {"msg": "Checking documents and embedding version…", "file": "", "i": 0, "n": 1}

    # Plan work
    changed, plan = check_and_ingest_if_needed(raw_dir, collection_name)

    # Full rebuild path
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
        splitter = _splitter()
        counts: Dict[str, int] = {}

        n = len(files)
        for i, pdf in enumerate(files, start=1):
            yield {"msg": "Indexing", "file": pdf.name, "i": i, "n": n}
            # ingest this single file
            docs = _load_docs_for_file(pdf)
            docs = splitter.split_documents(docs)
            if docs:
                vs.add_documents(docs)
                counts[pdf.name] = len(docs)
            else:
                counts[pdf.name] = 0

        # Write fresh manifest
        manifest = {"embedding_id": settings.embedding_id, "updated": _utc_now(), "files": {}}
        for p in files:
            manifest["files"][p.name] = {
                "sha256": _file_sha256(p),
                "chunks": counts.get(p.name, 0),
                "last_ingested": _utc_now(),
            }
        _write_manifest(manifest)
        yield {"msg": "Rebuild complete.", "file": "", "i": n, "n": n}
        return

    # Incremental path
    removed = plan.get("removed", []) or []
    added = plan.get("added", []) or []
    changed_files = plan.get("changed", []) or []
    kept = plan.get("kept", []) or []

    # Nothing to do
    if not (removed or added or changed_files):
        yield {"msg": "All documents are up to date.", "file": "", "i": 1, "n": 1}
        return

    prev_manifest = _read_manifest()
    prev_files = prev_manifest.get("files", {}) if isinstance(prev_manifest, dict) else {}

    vs = _vectordb(collection_name)
    splitter = _splitter()

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

    # 2) Ingest added + changed
    added_counts: Dict[str, int] = {}
    for name in added + changed_files:
        step += 1
        pdf = raw_dir / name
        yield {"msg": "Indexing", "file": name, "i": step, "n": ops_total}
        if not pdf.exists():
            # Skip if missing
            continue
        docs = _load_docs_for_file(pdf)
        docs = splitter.split_documents(docs)
        if docs:
            vs.add_documents(docs)
            added_counts[name] = len(docs)
        else:
            added_counts[name] = 0

    # 3) Write new manifest (kept + newly ingested; removed are omitted)
    manifest = {"embedding_id": settings.embedding_id, "updated": _utc_now(), "files": {}}
    # current files in dir (after removals)
    current_files = sorted([p for p in raw_dir.glob("*.pdf") if p.is_file()])
    current_names = [p.name for p in current_files]
    for p in current_files:
        name = p.name
        if name in added_counts:  # newly added/changed
            manifest["files"][name] = {
                "sha256": _file_sha256(p),
                "chunks": added_counts.get(name, 0),
                "last_ingested": _utc_now(),
            }
        elif name in kept:
            prev = prev_files.get(name, {})
            manifest["files"][name] = {
                "sha256": _file_sha256(p),  # refresh hash in case it silently changed
                "chunks": prev.get("chunks"),
                "last_ingested": prev.get("last_ingested", _utc_now()),
            }
        else:
            # If a file is present but not in any bucket (edge case), treat as kept
            prev = prev_files.get(name, {})
            manifest["files"][name] = {
                "sha256": _file_sha256(p),
                "chunks": prev.get("chunks"),
                "last_ingested": prev.get("last_ingested", _utc_now()),
            }

    _write_manifest(manifest)
    yield {"msg": f"Index updated. {len(current_names)} files tracked.", "file": "", "i": ops_total, "n": ops_total or 1}


# ---------- Document management helpers ----------

def list_tracked_files() -> List[str]:
    """Return sorted PDF names from manifest; fall back to raw_pdfs dir if manifest empty."""
    m = _read_manifest()
    files = sorted(list((m.get("files") or {}).keys()))
    if files:
        return files
    raw = Path("data/raw_pdfs")
    return sorted([p.name for p in raw.glob("*.pdf") if p.is_file()])

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