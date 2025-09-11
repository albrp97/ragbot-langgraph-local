from app.config.settings import settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def _embeddings():
    device = "cuda" if (settings.device in ("cuda", "auto")) else "cpu"
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_id,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": settings.embedding_batch_size},
    )

def get_retriever(k: int | None = None, collection_name: str = "docs_text", allowed_sources: list[str] | None = None):
    k = k or settings.retriever_k
    vs = Chroma(
        collection_name=collection_name,
        persist_directory=settings.chroma_path,
        embedding_function=_embeddings(),
    )
    search_kwargs = {"k": k}
    
    # Filter by metadata 'source_name' (we set it at ingest)
    if allowed_sources is not None:
        search_kwargs["filter"] = {"source_name": {"$in": allowed_sources}} if allowed_sources else {"source_name": {"$in": []}}

    return vs.as_retriever(search_type="mmr", search_kwargs=search_kwargs)