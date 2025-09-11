from typing import List, Tuple
from transformers import AutoTokenizer
from app.llm.chat import generate
from app.config.settings import settings

# Tokenizer for counting tokens
_tokenizer = AutoTokenizer.from_pretrained(settings.model_id, trust_remote_code=True)

def _count_tokens(text: str) -> int:
    return len(_tokenizer(text, add_special_tokens=False).input_ids)

def summarize_history(pairs: List[Tuple[str, str]], budget_tokens: int) -> str:
    """Resume turns (user, assistant) en <= budget_tokens."""
    if not pairs:
        return ""
    convo_lines = []
    for u, a in pairs:
        if u:
            convo_lines.append(f"User: {u}")
        if a:
            convo_lines.append(f"Assistant: {a}")
    convo_text = "\n".join(convo_lines)

    prompt = (
        "You are a concise conversation summarizer.\n"
        "Summarize the dialog below into bullet points of facts, decisions, and open questions.\n"
        f"Target length: <= {settings.token_limit_summary} tokens.\n\n"
        f"# Dialog\n{convo_text}\n\n# Summary:"
    )
    summary = generate(prompt, thinking=False, max_new_tokens=settings.token_limit_summary).strip()

    # Defensive trimming in case it exceeded
    while _count_tokens(summary) > budget_tokens:
        summary = summary[: int(len(summary) * 0.9)].strip()
    return summary

def compress_if_needed(history_pairs: List[Tuple[str, str]]) -> str:
    """Devuelve memoria comprimida si el historial supera TOKEN_LIMIT_HISTORY y la imprime."""
    if not history_pairs:
        return ""

    # count tokens in raw history
    raw_text = "\n".join([(u or "") + ("\n" + (a or "") if a else "") for (u, a) in history_pairs])
    raw_tokens = _count_tokens(raw_text)

    # if within limit, no summary needed
    if raw_tokens <= settings.token_limit_history:
        return ""

    # summarize all but last turn
    head = history_pairs[:-1]
    summary = summarize_history(head, settings.token_limit_summary).strip()
    sum_tokens = _count_tokens(summary)

    # logs
    print("🧠 [Memory] Summary created "
          f"({raw_tokens} → {sum_tokens} tokens; "
          f"limit={settings.token_limit_history}, target={settings.token_limit_summary})")
    print(summary)
    print("-" * 80)

    return summary
