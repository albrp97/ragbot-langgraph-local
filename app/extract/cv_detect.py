from __future__ import annotations
import re
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

from langchain_community.document_loaders import PyMuPDFLoader
from app.llm.chat import generate
from app.extract.ocr_utils import ocr_first_pages

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s\(\)]{6,}\d)")
DATE_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
URL_RE = re.compile(r"https?://[^\s)]+|(?:www\.)?[A-Za-z0-9-]+\.(?:com|org|net|io|ai|edu|dev)(?:/[^\s)]*)?")

DEBUG_DETECT = False
def _dbgd(*a):
    if DEBUG_DETECT:
        print(*a)

CV_TERMS = (
    "curriculum vitae","résumé","resume","cv","objective","summary","profile",
    "education","coursework","gpa","experience","work experience","research experience",
    "projects","skills","technologies","publications","certifications","awards",
    "leadership","activities","languages"
)

def _load_first_pages_text(pdf: Path, max_pages: int = 2) -> Tuple[str, int]:
    docs = PyMuPDFLoader(str(pdf)).load()
    total_pages = max((d.metadata or {}).get("page", 0) for d in docs) + 1 if docs else 0
    subset = [d for d in docs if (d.metadata or {}).get("page", 0) < max_pages]
    text = "\n".join(d.page_content for d in subset).strip()

    # OCR fallback for image-only PDFs
    if len(text) < 40:
        try:
            text = ocr_first_pages(pdf, max_pages=max_pages, dpi=220, lang="eng")
        except Exception:
            pass

    return text, total_pages

# Heuristic CV detection based on presence of keywords, email, phone, dates, and document length.
# Returns a score in [0.0, 1.0] and list of reasons contributing to the score.
def heuristic_cv_score(text: str, total_pages: int) -> Tuple[float, List[str]]:
    t = text.lower()
    reasons = []
    if not t.strip():
        return 0.0, ["empty_text"]

    term_hits = sum(1 for k in CV_TERMS if k in t)
    has_email = bool(EMAIL_RE.search(text))
    has_phone = bool(PHONE_RE.search(text))
    year_hits = len(DATE_RE.findall(text))
    short_doc_bonus = 0.15 if total_pages <= 3 else 0.0

    score = (
        (term_hits / 8.0) * 0.5 +         # section keywords
        (0.2 if has_email else 0.0) +
        (0.1 if has_phone else 0.0) +
        min(year_hits, 6) * 0.02 +        # date patterns
        short_doc_bonus
    )
    if has_email: reasons.append("email")
    if has_phone: reasons.append("phone")
    if short_doc_bonus > 0: reasons.append("short_doc")
    if term_hits: reasons.append(f"terms:{term_hits}")
    if year_hits: reasons.append(f"years:{year_hits}")
    # scale to [0.0, 1.0]
    score = max(0.0, min(1.0, score))
    return score, reasons

LLM_CLASSIFY_PROMPT = """You are a strict document classifier.
Decide if the text is a CV/resume (job-seeking document listing education, experience, skills).
Return ONLY minified JSON: {"is_cv": true|false}.

Text:
{doc}

JSON:"""

def llm_is_cv(text: str) -> bool:
    raw = generate(
        LLM_CLASSIFY_PROMPT.format(doc=text[:6000]), # only first 6k chars
        thinking=False, max_new_tokens=16
    )
    s = raw.strip().lower()
    if s.startswith("```"):
        s = s.strip("` \n")
        if s.startswith("json"): s = s[4:].lstrip()
    return '"is_cv": true' in s or "'is_cv': true" in s

def detect_cv(pdf_path: str | Path) -> Dict[str, Any]:
    pdf = Path(pdf_path)
    text, total_pages = _load_first_pages_text(pdf, max_pages=2)
    score, reasons = heuristic_cv_score(text, total_pages)
    decision = (score >= 0.65)
    uncertain = (0.35 <= score < 0.65)
    _dbgd(f"[detect] {pdf.name} pages={total_pages} len(text)={len(text)} score={score:.2f} reasons={reasons}")
    if not decision and uncertain:
        decision = llm_is_cv(text)
        reasons.append("llm_fallback=" + str(decision))
        _dbgd(f"[detect] LLM fallback → {decision}")
    return {
        "path": str(pdf),
        "is_cv": bool(decision),
        "score": float(score),
        "reasons": reasons,
        "pages": total_pages,
        "text_len": len(text),
    }