from __future__ import annotations
from pathlib import Path
from typing import Optional, Type, Dict, Any
import json

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from app.llm.chat import generate
from app.extract.schemas import SCHEMAS

# Debugging utility
DEBUG_DEFAULT = False
def _dbg(debug: bool, *args):
    if debug:
        print(*args)
        
# Example JSON to guide the model
EXAMPLE_CV_JSON = """
{
  "file_name": "studentaffairs.stanford.edu/cdc",
  "pages": 3,
  "is_cv": true,
  "detector_score": 0.85,
  "name": "ADRIANA SMITHFIELD",
  "email": "asmithfield12@stanford.edu",
  "phone": "xxx-xxx-xxxx",
  "location": "Stanford, CA",
  "websites": ["https://studentaffairs.stanford.edu/cdc", "https://www.stanford.edu"],
  "social_handles": [
    "https://www.stanford.edu/cdc"
  ],
  "objective": "To provide healthcare services to underserved communities and support underrepresented populations.",
  "summary": "This resume highlights Adriana Smithfield's academic background, professional experience, and contributions to healthcare and education.",
  "education": [
    {
      "institution": "Stanford University",
      "location": "Stanford, CA",
      "degree": "B.A. Candidate Human Biology | Global Infectious Disease and Women's Health",
      "year": "9/xx - Present"
    },
    {
      "institution": "Oxford University",
      "location": "Oxford, England",
      "degree": "Relevant coursework: Tutorial in International Health - studied social determinants of health, global governance, and behavior change",
      "year": "3/xx - 6/xx"
    }
  ],
  "experience": [
    {
      "company": "Center for Health Research in Women’s and Sex Differences in Medicine",
      "location": "Stanford, CA",
      "role": "Intern",
      "start_date": "6/xx",
      "end_date": Present,
      "description": "Research ethical challenges to enrolling women in research studies globally. Organized Global Women’s Health Conference and presentation for a conference speaker. Created and designed a course investigating the physical, emotional, and mental effects of sexual abuse through the life course and from multiple perspectives. Identified course topics and drafted course syllabus."
    },
    {
      "company": "Stanford Health 4 America",
      "location": "Stanford Prevention Center, School of Medicine, Stanford, CA",
      "role": "Intern",
      "start_date": "6/xx",
      "end_date": Present,
      "description": "Assist with the launch of an innovative professional certificate program. Develop admission process, fellow handbook, and memorandum of understanding between Stanford Health 4 America and Community Partners. Create promotional animations while working on marketing strategy and outreach."
    },
    {
      "company": "Undergraduate Research Assistant",
      "location": "Department of Psychiatry and Behavioral Sciences, Stanford, CA",
      "role": "Assistant",
      "start_date": "3/xx",
      "end_date": "2/xx",
      "description": "Assisted with the development of a clinical trial investigating use of a novel drug in children with autism. Awarded a $6,000 Bio-X Undergraduate Summer Research Grant from Stanford University, culminating in a presentation at Bio-X symposium. Presented research at the Symposia for Undergraduate Research and Public Service (SURPS)."
    }
  ],
  "projects": [
    {
      "title": "SPLASH Underserved Student Recruiter and Teacher",
      "location": "Stanford, CA",
      "description": "Communicated with primary contacts at various low-income high schools in the bay area to draw hundreds of students to attend Fall SPLASH 2012. Assisted in the logistical planning as a member of the administration team. Taught classes on the biology and historical context of lactose intolerance to students attending Spring SPLASH 20XX."
    },
    {
      "title": "ThinkMath Instructor, Trainer, and Assistant Team Lead",
      "location": "Stanford, CA",
      "description": "Taught elementary school students from a Singaporean math curriculum. Led training sessions for new ThinkMath instructors about lesson planning and teaching techniques. Organized placement results for students and communicated with parents on site."
    }
  ],
  "skills": [
    "Languages: German (proficient), Spanish (conversational)",
    "Computer Skills: MS Office Suite, Macromedia Suite, DreamWeaver, PhotoShop",
    "Other: Alpha Kappa Delta Phi Sorority Vice President of Community Service & Philanthropy, Multicultural Greek Council Representative & Recruitment Chair, Data Intern at Center for Interdisciplinary Brain Science Research, Stanford Immersion in Medicine Physician Shadowing Program"
  ],
  "certifications": ["AzURE AI-900", "Stanford University CPR/AED for Professional Rescuers"],
  "awards": ["Bio-X Undergraduate Summer Research Grant"],
  "publications": ["Smithfield A, Doe J. Ethical Challenges in Enrolling Women in Research Studies. Journal of Medical Ethics, 2020."],
  "languages": [
    "German (proficient), Spanish (conversational)"
  ],
  "activities": [
    "SPLASH Underserved Student Recruiter and Teacher",
    "ThinkMath Instructor, Trainer, and Assistant Team Lead"
  ]
}
""".strip()

# Prompt template for structured extraction
PROMPT = """You extract structured data from a CV/resume.

First, here is an EXAMPLE JSON (for format/style guidance ONLY — do not copy its values):
{example}

Now, output ONLY one valid RFC 8259 JSON object with EXACTLY these top-level keys:
{keys}

STRICT rules:
- Valid JSON only (double quotes, no trailing commas, no comments, no extra keys).
- If a field is missing, use null (or [] for lists).
- Keep bullets short.
- Do NOT output markdown fences or any extra text besides the JSON object.

# Document text:
{doc}

# JSON:"""

# Loads text from PDF, optionally only a page range, and applies OCR if needed
# TODO test OCR capability
# returns text and whether OCR was used
def _load_text(pdf_path: Path, pages: Optional[tuple[int, int]] = None, chunk_size: int = 3000) -> tuple[str, bool]:
    docs = PyMuPDFLoader(str(pdf_path)).load()
    if pages is not None:
        start, end = pages
        docs = [d for d in docs if (d.metadata or {}).get("page", -1) in range(start, end)]
    text = "\n\n".join(d.page_content for d in docs).strip()

    used_ocr = False
    if len(text) < 40:
        try:
            from app.extract.ocr_utils import ocr_first_pages
            text = ocr_first_pages(pdf_path, max_pages=2, dpi=220, lang="eng")
            used_ocr = True
        except Exception:
            pass

    if not text:
        return "", used_ocr

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    chunks = splitter.split_text(text)
    return "\n\n".join(chunks), used_ocr

# Main function to extract structured data from PDF using LLM
# TODO add guardrails for JSON validity
# returns path to the raw JSON file
def extract_to_json(
    pdf_path: str | Path,
    schema_name: str = "cv_standard",
    out_dir: str | Path = "data/structured",
    pages: Optional[tuple[int, int]] = None,
    meta: Optional[Dict[str, Any]] = None,
    debug: bool = DEBUG_DEFAULT,
) -> Path:
    """
    Minimal extractor that saves the raw LLM output to disk.
    Writes:
      - <stem>.<schema>.raw.json  (exactly what the LLM returned; may not be valid JSON)
      - <stem>.<schema>.meta.json (small metadata blob)
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # we use the schema to tell the model which keys to use
    if schema_name not in SCHEMAS:
        raise ValueError(f"Unknown schema '{schema_name}'. Available: {list(SCHEMAS)}")
    model_cls: Type = SCHEMAS[schema_name]
    keys = list(model_cls.model_fields.keys())

    doc_text, used_ocr = _load_text(pdf_path, pages=pages)
    _dbg(debug, f"[extract] file={pdf_path.name} used_ocr={used_ocr} len(text)={len(doc_text)}")

    if not doc_text:
        # write meta even if empty
        meta_path = out_dir / f"{pdf_path.stem}.{schema_name}.meta.json"
        meta_obj = {
            "file_name": pdf_path.name,
            "schema": schema_name,
            "used_ocr": used_ocr,
            "text_len": 0,
            "pages": (meta or {}).get("pages"),
            "is_cv": (meta or {}).get("is_cv"),
            "detector_score": (meta or {}).get("score"),
            "note": "empty text after OCR; raw omitted",
        }
        meta_path.write_text(json.dumps(meta_obj, indent=2, ensure_ascii=False), encoding="utf-8")
        return meta_path

    base_prompt = PROMPT
    if schema_name == "cv_standard":
        prompt = base_prompt.format(keys=", ".join(keys), doc=doc_text[:8000], example=EXAMPLE_CV_JSON)
    else:
        prompt = base_prompt.format(keys=", ".join(keys), doc=doc_text[:8000])
    
    _dbg(debug, "[extract] prompt preview:\n", prompt[:600], "...\n---")

    raw = generate(prompt, thinking=False, max_new_tokens=5000) # enough tokens
    _dbg(debug, "[extract] raw LLM output (head):\n", (raw[:600] + "..." if len(raw) > 600 else raw), "\n---")

    # Save raw output exactly as returned by the model
    raw_path = out_dir / f"{pdf_path.stem}.{schema_name}.raw.json"
    raw_path.write_text(raw, encoding="utf-8")

    # Save a small meta file next to it
    meta_path = out_dir / f"{pdf_path.stem}.{schema_name}.meta.json"
    meta_obj = {
        "file_name": pdf_path.name,
        "schema": schema_name,
        "used_ocr": used_ocr,
        "text_len": len(doc_text),
        "pages": (meta or {}).get("pages"),
        "is_cv": (meta or {}).get("is_cv"),
        "detector_score": (meta or {}).get("score"),
        "raw_file": raw_path.name,
    }
    meta_path.write_text(json.dumps(meta_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    _dbg(debug, f"[extract] wrote RAW → {raw_path}")
    _dbg(debug, f"[extract] wrote META → {meta_path}")
    return raw_path