from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from app.config.settings import settings

print("MODEL_ID:", settings.model_id)
print("EMBEDDING_ID:", settings.embedding_id)
print("CHROMA_PATH:", settings.chroma_path)
print("DEVICE:", settings.device)
