import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from src.utility.config import key_manager
from .constants import PERSIST_DIR, EMBEDDING_MODEL


# ══════════════════════════════════════════════════════════════════════════════
#  VECTOR STORE (shared by both pipelines)
# ══════════════════════════════════════════════════════════════════════════════

def create_vector_store(chunks: list, persist_directory: str = PERSIST_DIR):
    """Embed chunks and persist to ChromaDB with batching and key rotation."""
    print("\n💾 Creating embeddings and storing in ChromaDB (batched)...")

    def get_embedding_model():
        return GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=key_manager.get_current_key()
        )

    embedding_model = get_embedding_model()

    # Initialise empty vector store
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # Batching to avoid quota exhaustion (429)
    batch_size = 2  # Very small batch for preview model
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"  📥 Storing batch {i // batch_size + 1}/{len(chunks) // batch_size + 1}...")

        while True:
            try:
                vectorstore.add_documents(batch)
                break
            except Exception as e:
                if "429" in str(e) or "Resource has been exhausted" in str(e):
                    if key_manager.rotate_key():
                        # Refresh embedding model in vectorstore
                        vectorstore._embedding_function = get_embedding_model()
                        print("  🔄 API Key rotated. Retrying batch...")
                        continue
                    else:
                        print(f"  ⏳ All keys exhausted. Waiting 45s for quota reset...")
                        time.sleep(45)
                else:
                    print(f"  ❌ Error adding batch: {e}")
                    raise e

        if i + batch_size < len(chunks):
            print(f"  ⏳ Safety delay (5s) to prevent quota exhaustion...")
            time.sleep(5)

    print(f"\n  ✅ Vector store updated at: {persist_directory}")
    print(f"     Total vectors: {vectorstore._collection.count()}")
    return vectorstore
