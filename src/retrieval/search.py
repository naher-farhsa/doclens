import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.utility.config import key_manager
from .constants import EMBEDDING_MODEL, KEY_ROTATION_DELAY
from .retriever import _refresh_embeddings


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
