from typing import Optional
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.config.settings import settings

_model = None
_tokenizer = None
_loaded_once = False

def _want_cuda() -> bool:
    return settings.device in ("cuda", "auto") and torch.cuda.is_available()

def _select_dtype() -> torch.dtype:
    return torch.float16 if _want_cuda() else torch.float32

def load_model():
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    dtype = _select_dtype()

    _tokenizer = AutoTokenizer.from_pretrained(
        settings.model_id,
        trust_remote_code=True,
    )

    if settings.device == "cuda":
        # Explicit: load then move to single GPU
        _model = AutoModelForCausalLM.from_pretrained(
            settings.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to("cuda")
    elif settings.device == "auto" and torch.cuda.is_available():
        # Let accelerate shard automatically if needed
        _model = AutoModelForCausalLM.from_pretrained(
            settings.model_id,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        # CPU fallback
        _model = AutoModelForCausalLM.from_pretrained(
            settings.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to("cpu")

    # log device
    try:
        dev = next(_model.parameters()).device
        print(f"🔧 LLM loaded on: {dev} (dtype={dtype})")
    except Exception:
        pass

    return _model, _tokenizer

def preload_model():
    """Call this once at app startup to load & keep the model in memory."""
    load_model()

@torch.inference_mode()
def generate(
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    do_sample: Optional[bool] = None,
    seed: Optional[int] = None,
    thinking: Optional[bool] = None,
) -> str:
    model, tokenizer = load_model()
    # from settings
    temperature = settings.llm_temperature if temperature is None else temperature
    top_p = settings.llm_top_p if top_p is None else top_p
    top_k = settings.llm_top_k if top_k is None else top_k
    repetition_penalty = settings.llm_repetition_penalty if repetition_penalty is None else repetition_penalty
    max_new_tokens = settings.llm_max_new_tokens if max_new_tokens is None else max_new_tokens
    do_sample = settings.llm_do_sample if do_sample is None else do_sample
    seed = settings.llm_seed if seed is None else seed
    enable_thinking = thinking if thinking is not None else settings.llm_reasoning

    if isinstance(seed, int) and seed >= 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    text = prompt
    if hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking, 
        )

    inputs = tokenizer(text, return_tensors="pt")
    
    # move to model device
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    greedy = (do_sample is False) or (temperature is not None and float(temperature) <= 0)

    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "repetition_penalty": float(repetition_penalty),
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": not greedy,  # bool
    }

    if not greedy:
        # Only include sampling knobs when actually sampling
        # (avoid passing None/0 which triggers warnings)
        if temperature is not None and float(temperature) > 0:
            gen_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
        if top_k is not None and int(top_k) > 0:
            gen_kwargs["top_k"] = int(top_k)

    # Call HF generate with a *clean* kwargs set
    gen_ids = model.generate(**inputs, **gen_kwargs)
    
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = gen_ids[0][prompt_len:]
    out = tokenizer.decode(new_tokens, skip_special_tokens=True)

    if not enable_thinking:
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
        out = re.sub(r"\n\s*\n", "\n\n", out).strip()
    out = re.sub(r"^\s*(?:assistant|Assistant)\s*[:：-]*\s*", "", out).strip()

    # Trim prompt when using chat template
    if hasattr(tokenizer, "apply_chat_template") and prompt in out:
        out = out.split(prompt, 1)[-1].strip()
    return out.strip()
