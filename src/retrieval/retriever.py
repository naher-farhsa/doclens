import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from src.utility.config import key_manager
from .constants import PERSIST_DIR, EMBEDDING_MODEL

# ── Helper: refresh embedding function after key change ──────────────────────
def _refresh_embeddings(retriever, new_emb):
    """
    Swap the embedding function inside a retriever (or EnsembleRetriever)
    after an API key rotation.
    """
    try:
        for r in getattr(retriever, "retrievers", []):
            if hasattr(r, "vectorstore"):
                r.vectorstore._embedding_function = new_emb
        if hasattr(retriever, "vectorstore"):
            retriever.vectorstore._embedding_function = new_emb
    except Exception as err:
        print(f"  ⚠️  Failed to refresh retriever embeddings: {err}")


# ── Step 1: Load Vector Store ────────────────────────────────────────────────
def load_vector_store(persist_directory: str = PERSIST_DIR):
    """Load the persisted ChromaDB vector store with current API key."""
    api_key = key_manager.get_current_key()
    print(f"\n📦 Loading vector store (API Key {key_manager.current_idx + 1})...")

    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"Vector store not found at '{persist_directory}'. "
            "Run the ingestion pipeline first."
        )

    embedding_model = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

    count = vectorstore._collection.count()
    print(f"  ✅ Loaded vector store with {count} vectors")
    return vectorstore


# ── Step 2: Create Hybrid Retriever ──────────────────────────────────────────
def create_hybrid_retriever(vectorstore: Chroma):
    """
    Create a hybrid retriever combining:
      - Vector search (semantic similarity)  → weight 0.7
      - BM25 keyword search                  → weight 0.3

    Deduplication note: EnsembleRetriever can return the same chunk from both
    the vector and BM25 legs. batch_retrieve() deduplicates on page_content
    after collection, so duplicates never reach the LLM prompt.
    """
    print("\n🔍 Setting up Hybrid Retriever (Vector + BM25)...")

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    all_docs = vectorstore.get(include=["documents", "metadatas"])
    from langchain_core.documents import Document

    documents = [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]

    if not documents:
        print("  ⚠️  No documents found in vector store. Returning vector retriever only.")
        return vector_retriever

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3

    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3],
    )

    print("  ✅ Hybrid Retriever ready (vector 0.7 + BM25 0.3)")
    return hybrid_retriever
