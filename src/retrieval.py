"""
DocLens – Retrieval Pipeline
==============================
Hybrid Search: Vector (Chroma) + Keyword (BM25) via EnsembleRetriever

Bugs fixed in this file
------------------------
BUG A (retrieve) – Key rotation fired the next embedding request immediately
                   with no delay, so every key hit 429 in under a second.
                   Fixed: added KEY_ROTATION_DELAY sleep after each rotation.

BUG B (retrieve) – After all keys exhausted + sleep, the code looped back
                   but current_idx was still at the last exhausted key, and
                   the embedding function was never refreshed.
                   Fixed: reset_keys() + refresh embedding after the wait.
"""

import os
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from dotenv import load_dotenv
from src.config import key_manager

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────
PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "chroma_db")
EMBEDDING_MODEL = "gemini-embedding-2-preview"

# Same cooldown used in generation.py — keeps behaviour consistent.
# Embedding API shares the same free-tier RPM pool as the LLM.
KEY_ROTATION_DELAY = 12


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


# ── Step 3: Retrieve ─────────────────────────────────────────────────────────
def retrieve(query: str, retriever):
    """Retrieve relevant chunks for a single query with retry and key rotation."""
    max_retries = len(key_manager.keys) * 2

    for attempt in range(max_retries):
        try:
            relevant_docs = retriever.invoke(query)
            return relevant_docs

        except Exception as e:
            if "429" in str(e) or "Resource has been exhausted" in str(e):
                if key_manager.rotate_key():
                    # ── BUG A FIX ────────────────────────────────────────────
                    # BEFORE: refreshed embeddings then continued immediately
                    #         → new key hit 429 within milliseconds too
                    #
                    # AFTER:  sleep KEY_ROTATION_DELAY before using the new key
                    #         so its quota window has breathing room.
                    # ─────────────────────────────────────────────────────────
                    print(f"  ⏳ Cooling down {KEY_ROTATION_DELAY}s before using new key...")
                    time.sleep(KEY_ROTATION_DELAY)              # ← BUG A FIX

                    new_emb = GoogleGenerativeAIEmbeddings(
                        model=EMBEDDING_MODEL,
                        google_api_key=key_manager.get_current_key(),
                    )
                    _refresh_embeddings(retriever, new_emb)
                    print(f"  🔄 Retrying retrieval with Key {key_manager.current_idx + 1}...")
                    continue

                else:
                    # ── BUG B FIX ────────────────────────────────────────────
                    # BEFORE: time.sleep(wait_time) then looped back with the
                    #         same exhausted key and stale embedding function
                    #         → immediate 429 on every post-sleep attempt
                    #
                    # AFTER:  sleep → reset key index → rebuild embedding with
                    #         key 0 → refresh retriever → continue
                    # ─────────────────────────────────────────────────────────
                    wait_time = 60 * (attempt + 1)
                    print(f"  ⏳ All retrieval keys exhausted. Waiting {wait_time}s...")
                    time.sleep(wait_time)

                    key_manager.reset_keys()                    # ← BUG B FIX
                    new_emb = GoogleGenerativeAIEmbeddings(     # ← BUG B FIX
                        model=EMBEDDING_MODEL,
                        google_api_key=key_manager.get_current_key(),
                    )
                    _refresh_embeddings(retriever, new_emb)     # ← BUG B FIX
                    continue                                    # ← BUG B FIX
            else:
                print(f"  ❌ Retrieval error: {e}")
                raise e

    return []


# ── Step 4: Batch Retrieve ───────────────────────────────────────────────────
def batch_retrieve(queries: list, retriever):
    """
    Retrieve chunks for multiple queries and return deduplicated results.

    Deduplication: both vector and BM25 legs can return the same chunk.
    We deduplicate on page_content so the LLM never sees the same text twice.
    The 5s inter-query delay also helps avoid burning quota mid-batch.
    """
    print(f"\n🔎 Batch retrieving for {len(queries)} queries...")
    all_docs = []

    for i, q in enumerate(queries):
        if i > 0:
            time.sleep(5)  # gap between queries to avoid mid-batch 429

        print(f"  🔍 Query {i+1}/{len(queries)}: {q}")
        docs = retrieve(q, retriever)
        all_docs.extend(docs)

    # Deduplicate on page_content (handles overlap between vector + BM25 legs)
    unique_docs = []
    seen_content = set()
    for doc in all_docs:
        if doc.page_content not in seen_content:
            unique_docs.append(doc)
            seen_content.add(doc.page_content)

    print(f"  ✅ Batch retrieval complete: {len(unique_docs)} unique chunks found.")
    return unique_docs


if __name__ == "__main__":
    vs = load_vector_store()
    retriever = create_hybrid_retriever(vs)

    test_query = "What is AGI?"
    print(f"\n🔎 Test query: '{test_query}'")
    docs = retrieve(test_query, retriever)
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content[:200])