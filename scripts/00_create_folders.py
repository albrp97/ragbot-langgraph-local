from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = [
    "app/ui",
    "data/raw_pdfs",
    "data/chroma",
    "scripts",
]

def main():
    for d in DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    print("✅ Minimal folders ready.")

if __name__ == "__main__":
    main()
