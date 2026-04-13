import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from src.utility.config import key_manager
from src.utility.timer import PipelineTimer
from .constants import PERSIST_DIR, EMBEDDING_MODEL
from .multimodal import partition_documents, chunk_by_modality, process_chunks
from .fallback import _fallback_load_documents, _fallback_chunk_documents
from .vector_store import create_vector_store


# ══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_ingestion():
    """Run the full ingestion pipeline. Tries multimodal first, falls back to text-only."""
    print("=" * 60)
    print("  📥 DocLens – Ingestion Pipeline (Multimodal)")
    print("=" * 60)

    # Check if vector store exists and is not empty
    if os.path.exists(PERSIST_DIR):
        try:
            embedding_model = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=key_manager.get_current_key()
            )
            vectorstore = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embedding_model,
                collection_metadata={"hnsw:space": "cosine"},
            )
            count = vectorstore._collection.count()
            if count > 0:
                print(f"\n✅ Vector store already exists at '{PERSIST_DIR}' with {count} vectors.")
                print("   Skipping ingestion. Delete the db/ folder to re-ingest.\n")
                return vectorstore
            else:
                print(f"\n⚠️  Vector store at '{PERSIST_DIR}' is empty. Re-ingesting...")
        except Exception as e:
            print(f"\n⚠️  Error loading existing vector store ({e}). Re-ingesting...")

    # Try multimodal pipeline first
    try:
        print("\n🚀 Attempting multimodal pipeline (Unstructured ETL)...")
        with PipelineTimer("1. Partition Documents"):
            elements = partition_documents()
        with PipelineTimer("2. Chunk by Modality"):
            chunks = chunk_by_modality(elements)
        with PipelineTimer("3. Process & Summarize Chunks"):
            processed = process_chunks(chunks)
        with PipelineTimer("4. Create Vector Store"):
            vectorstore = create_vector_store(processed)

    except Exception as e:
        print(f"\n⚠️  Multimodal pipeline failed: {e}")
        print("    Falling back to text-only pipeline...\n")
        documents = _fallback_load_documents()
        chunks = _fallback_chunk_documents(documents)
        vectorstore = create_vector_store(chunks)

    print("\n" + "=" * 60)
    print("  ✅ Ingestion complete!")
    print("=" * 60)
    return vectorstore

if __name__ == "__main__":
    run_ingestion()
