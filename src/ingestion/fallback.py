import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.utility.config import key_manager
from .constants import DOCS_PATH, EMBEDDING_MODEL


# ══════════════════════════════════════════════════════════════════════════════
#  FALLBACK PIPELINE (Text-Only — original SemanticChunker approach)
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_load_documents(path: str = DOCS_PATH):
    """Fallback: Load PDF file(s) using PyPDFLoader (text-only)."""
    print(f"\n📂 [Fallback] Loading from: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"The path '{path}' does not exist. "
            "Please create it or supply a valid path."
        )

    if os.path.isfile(path):
        if not path.lower().endswith(".pdf"):
            raise ValueError(f"Fallback loader only supports PDFs. Found: {path}")
        pdf_paths = [path]
    else:
        pdf_paths = [
            os.path.join(path, f) for f in os.listdir(path) 
            if f.lower().endswith(".pdf")
        ]

    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found at '{path}'.")

    all_documents = []
    for pdf_path in pdf_paths:
        doc_file = os.path.basename(pdf_path)
        print(f"  📄 Loading: {doc_file}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)

    print(f"  ✅ Loaded {len(all_documents)} pages from {len(pdf_paths)} PDF(s)")

    # Preview first 2 documents
    for i, doc in enumerate(all_documents[:2]):
        print(f"\n  📃 Page {i + 1}:")
        print(f"     Source : {doc.metadata.get('source', 'N/A')}")
        print(f"     Length : {len(doc.page_content)} characters")
        print(f"     Preview: {doc.page_content[:120]}...")

    return all_documents


def _fallback_chunk_documents(documents: list):
    """Fallback: Split documents using SemanticChunker (text-only)."""
    print("\n✂️  [Fallback] Chunking documents (Semantic Chunking)...")

    embedding_model = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=key_manager.get_current_key()
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
