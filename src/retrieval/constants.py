import os

PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "db", "chroma_db")
EMBEDDING_MODEL = "gemini-embedding-2-preview"

# Same cooldown used in generation.py — keeps behaviour consistent.
# Embedding API shares the same free-tier RPM pool as the LLM.
KEY_ROTATION_DELAY = 12
