# app/llm/embeddings_text.py
from typing import List
import torch
from transformers import AutoModel, AutoTokenizer
from app.config.settings import settings

_model = None
_tokenizer = None

# set ups device and dtype
def _device():
    if settings.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def _dtype():
    return torch.float16 if (_device().type == "cuda") else torch.float32

# loads embedding model if not already loaded
def _load_embedding_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    dev = _device()
    _tokenizer = AutoTokenizer.from_pretrained(settings.embedding_id, trust_remote_code=True)
    _model = AutoModel.from_pretrained(
        settings.embedding_id,
        trust_remote_code=True,
        torch_dtype=_dtype(),
    ).to(dev)
    _model.eval()
    return _model, _tokenizer

# mean pooling for sentence embeddings
@torch.inference_mode()
def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

# embeds a list of texts into vectors
@torch.inference_mode()
def embed_texts(texts: List[str]) -> List[List[float]]:
    model, tokenizer = _load_embedding_model()
    dev = next(model.parameters()).device
    out = []

    MAX_LEN = 1024
    BS = 8 if dev.type == "cuda" else 4

    for i in range(0, len(texts), BS):
        batch = texts[i:i+BS]
        toks = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=MAX_LEN)
        toks = {k: v.to(dev, non_blocking=True) for k,v in toks.items()}
        outputs = model(**toks)
        embs = _mean_pool(outputs.last_hidden_state, toks["attention_mask"])
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        out.extend(embs.detach().cpu().tolist())
        del toks, outputs, embs
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return out
