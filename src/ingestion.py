"""
DocLens – Ingestion Pipeline
=============================
Load PDFs → Semantic Chunking → Embed → Store in ChromaDB
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from src.config import key_manager

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────
DOCS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "documents")
PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "chroma_db")
EMBEDDING_MODEL = "gemini-embedding-2-preview"


# ── Step 1: Document Loading ────────────────────────────────────────────────
def load_documents(docs_path: str = DOCS_PATH):
    """Load all PDF files from the documents directory."""
    print(f"\n📂 Loading documents from: {docs_path}")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory '{docs_path}' does not exist. "
            "Please create it and add your PDF files."
        )

    pdf_files = [f for f in os.listdir(docs_path) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{docs_path}'. Please add your documents."
        )

    all_documents = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(docs_path, pdf_file)
        print(f"  📄 Loading: {pdf_file}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)

    print(f"  ✅ Loaded {len(all_documents)} pages from {len(pdf_files)} PDF(s)")

    # Preview first 2 documents
    for i, doc in enumerate(all_documents[:2]):
        print(f"\n  📃 Page {i + 1}:")
        print(f"     Source : {doc.metadata.get('source', 'N/A')}")
        print(f"     Length : {len(doc.page_content)} characters")
        print(f"     Preview: {doc.page_content[:120]}...")

    return all_documents


# ── Step 2: Semantic Chunking ────────────────────────────────────────────────
def chunk_documents(documents: list):
    """Split documents into semantically meaningful chunks."""
    print("\n✂️  Chunking documents (Semantic Chunking)...")

    embedding_model = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=key_manager.get_current_key()   # ← add this
    )
    try:
        semantic_splitter = SemanticChunker(
            embeddings=embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=70,
        )
        chunks = semantic_splitter.split_documents(documents)
        print(f"  ✅ Semantic chunking produced {len(chunks)} chunks")

    except Exception as e:
        print(f"  ⚠️  Semantic chunking failed ({e}), falling back to recursive splitter...")
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = fallback_splitter.split_documents(documents)
        print(f"  ✅ Recursive chunking produced {len(chunks)} chunks")

    # Preview first 3 chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n  --- Chunk {i + 1} ({len(chunk.page_content)} chars) ---")
        print(f"  {chunk.page_content[:150]}...")
        print(f"  Source: {chunk.metadata.get('source', 'N/A')}")

    if len(chunks) > 3:
        print(f"\n  ... and {len(chunks) - 3} more chunks")

    return chunks


from src.config import key_manager

# ── Step 3: Embedding + Vector Store ─────────────────────────────────────────
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
    import time
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
            time.sleep(5)  # Reduced delay for smaller docs

    print(f"\n  ✅ Vector store updated at: {persist_directory}")
    print(f"     Total vectors: {vectorstore._collection.count()}")
    return vectorstore


# ── Orchestrator ─────────────────────────────────────────────────────────────
def run_ingestion():
    """Run the full ingestion pipeline (skip if DB already exists)."""
    print("=" * 60)
    print("  📥 DocLens – Ingestion Pipeline")
    print("=" * 60)

    # Check if vector store exists and is not empty
    is_empty = True
    if os.path.exists(PERSIST_DIR):
        try:
            embedding_model = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=key_manager.get_current_key()   # ← add this
            )
            vectorstore = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embedding_model,
                collection_metadata={"hnsw:space": "cosine"},
            )
            count = vectorstore._collection.count()
            if count > 0:
                is_empty = False
                print(f"\n✅ Vector store already exists at '{PERSIST_DIR}' with {count} vectors.")
                print("   Skipping ingestion. Delete the db/ folder to re-ingest.\n")
                return vectorstore
            else:
                print(f"\n⚠️  Vector store at '{PERSIST_DIR}' is empty. Re-ingesting...")
        except Exception as e:
            print(f"\n⚠️  Error loading existing vector store ({e}). Re-ingesting...")

    # Run pipeline if empty or doesn't exist
    documents = load_documents()
    chunks = chunk_documents(documents)
    vectorstore = create_vector_store(chunks)

    print("\n" + "=" * 60)
    print("  ✅ Ingestion complete!")
    print("=" * 60)
    return vectorstore


if __name__ == "__main__":
    run_ingestion()
