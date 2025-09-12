from __future__ import annotations
from pathlib import Path
from typing import Optional
import io
import os

import fitz 
from PIL import Image
import pytesseract

_TESSERACT_HINTS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]
if "TESSERACT_CMD" in os.environ:
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
else:
    for p in _TESSERACT_HINTS:
        if Path(p).exists():
            pytesseract.pytesseract.tesseract_cmd = p
            break

def ocr_first_pages(pdf_path: str | Path, max_pages: int = 2, dpi: int = 200, lang: str = "eng") -> str:
    """Render first N pages via PyMuPDF and OCR them with Tesseract."""
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    n = min(max_pages, len(doc))
    texts = []
    for i in range(n):
        page = doc[i]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang=lang)
        texts.append(text)
    return "\n\n".join(texts).strip()
