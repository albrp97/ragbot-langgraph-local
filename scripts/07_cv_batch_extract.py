from pathlib import Path
import sys, json, time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from app.llm.chat import preload_model
from app.extract.cv_detect import detect_cv
from app.extract.structured import extract_to_json

RAW = Path("data/raw_pdfs")
OUT = Path("data/structured/cv"); OUT.mkdir(parents=True, exist_ok=True)

def main():
    preload_model()
    pdfs = sorted([p for p in RAW.glob("*.pdf") if p.is_file()])
    if not pdfs:
        print("No PDFs found in data/raw_pdfs.")
        return

    summary = {"total": len(pdfs), "cv": 0, "non_cv": 0, "outputs": []}
    t0 = time.time()

    for i, pdf in enumerate(pdfs, 1):
        d = detect_cv(pdf)
        if d["is_cv"]:
            out = extract_to_json(pdf, schema_name="cv_standard", out_dir=OUT, meta=d, debug=False)
            summary["cv"] += 1
            summary["outputs"].append({"pdf": pdf.name, "raw_json": str(out)})
            print(f"[{i}/{len(pdfs)}] ✅ CV: {pdf.name} -> {Path(out).name}")
        else:
            summary["non_cv"] += 1
            print(f"[{i}/{len(pdfs)}] ⏭️  Not CV: {pdf.name} (score={d['score']:.2f}, {','.join(d['reasons'])})")

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s. CVs: {summary['cv']}/{summary['total']}")
    (OUT / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
