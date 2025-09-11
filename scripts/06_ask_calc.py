from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph.rag_graph import build_rag_graph
from app.llm.chat import preload_model

# --- Quiet Transformers warnings (must run before importing anything that imports transformers) ---
import os, warnings, logging
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # default level inside HF

try:
    # This import sets the HF logger; safe & fast
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_default_handler()  # optional: avoid duplicate console handlers
except Exception:
    pass

# Extra belt & suspenders: silence loggers by name
for name in ("transformers", "transformers.generation", "transformers.generation.utils"):
    logging.getLogger(name).setLevel(logging.ERROR)
    logging.getLogger(name).propagate = False

# (Optional) kill that specific text if some version still emits via warnings
warnings.filterwarnings("ignore", message=r"The following generation flags are not valid and may be ignored")
# -------------------------------------------------------------------------

if __name__ == "__main__":
    preload_model()
    app = build_rag_graph()
    for q in [
        "make a summary of [3, 5, 10, 12]",
        "what is 12345 * 6789",
        "what are the most dangerous languages to dissapear?",
        "simulate 5 dice rolls and give me the mean",
        "What are the best NASA missions?",
        "make a python list of the first 10 square numbers",
    ]:
        out = app.invoke({"query": q, "history": []})
        print("\nQ:", q)
        print("A:", out["answer"])
        print("Sources:", out["sources"])
