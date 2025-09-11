from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.llm.chat import generate

if __name__ == "__main__":
    print("Default settings:")
    print(generate("Respond with a single word: hello or hi."))

    print("\nOverrides config:")
    print(generate(
        "Respond with a single word: hello or hi.",
        temperature=0.1,
        top_p=0.8,
        max_new_tokens=800,
        seed=42,
        thinking=True,
    ))
