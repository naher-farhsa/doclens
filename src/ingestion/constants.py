import os

# ── Constants ────────────────────────────────────────────────────────────────
DOCS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "documents/Transformer_Architecturequit.pdf")
PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "db", "chroma_db")
EMBEDDING_MODEL = "gemini-embedding-2-preview"
GROQ_SUMMARY_MODEL = "llama-3.3-70b-versatile"


