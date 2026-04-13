import os
import json
import time
from typing import List, Dict

print("  [DEBUG] Importing Unstructured module (this may take a few minutes if Matplotlib is building font cache)...", flush=True)
# Unstructured for multimodal document parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
print("  [DEBUG] Unstructured imported successfully!", flush=True)


from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from .constants import DOCS_PATH, GROQ_SUMMARY_MODEL

# ══════════════════════════════════════════════════════════════════════════════
#  MULTIMODAL PIPELINE (Primary)
# ══════════════════════════════════════════════════════════════════════════════

# ── Step 1+2: Partition Documents ────────────────────────────────────────────
def partition_documents(path: str = DOCS_PATH):
    """Extract elements from PDF using unstructured."""
    print(f"\n📄 Partitioning document: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"The path '{path}' does not exist.")

    elements = partition_pdf(
        filename=path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True
    )

    print(f"\n  ✅ Extracted {len(elements)} elements")

    # All types of different atomic elements we see from unstructured
    unique_element_types = set([str(type(el).__name__) for el in elements])
    print(f"  📊 Unique element types: {unique_element_types}")

    # Gather all images
    images = [element for element in elements if getattr(element, 'category', type(element).__name__) == 'Image']
    print(f"  🖼️ Found {len(images)} images")

    # Gather all tables
    tables = [element for element in elements if getattr(element, 'category', type(element).__name__) == 'Table']
    print(f"  📊 Found {len(tables)} tables")

    return elements


# ── Step 3: Chunk by Modality ────────────────────────────────────────────────
def chunk_by_modality(elements: list):
    """
    Create modality-aware chunks using Unstructured's title-based strategy.
    Groups text under headings, preserves tables/images as orig_elements.
    """
    print("\n✂️  Chunking by title (modality-aware)...")

    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500,
    )

    print(f"  ✅ Created {len(chunks)} chunks")

    # All unique types
    unique_chunk_types = set([str(type(chunk).__name__) for chunk in chunks])
    print(f"  📊 Unique chunk types: {unique_chunk_types}")

    return chunks


# ── Step 4: Separate Content Types ───────────────────────────────────────────
def separate_content_types(chunk) -> Dict:
    """
    Analyze a chunk and separate its content by modality.
    Returns dict with text, tables (HTML), images (base64), and type tags.
    """
    content_data = {
        "text": chunk.text,
        "tables": [],
        "images": [],
        "types": ["text"],
    }

    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            # Handle tables
            if element_type == "Table":
                content_data["types"].append("table")
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)

            # Handle images
            elif element_type == "Image":
                if hasattr(element, "metadata") and hasattr(element.metadata, "image_base64"):
                    content_data["types"].append("image")
                    content_data["images"].append(element.metadata.image_base64)

    content_data["types"] = list(set(content_data["types"]))
    return content_data


# ── Step 5: AI Summary (Groq) ───────────────────────────────────────────────
def create_ai_summary(text: str, tables: List[str], images: List[str]) -> str:
    """
    Create a searchable text summary for mixed-content chunks using Groq.
      - Tables: HTML is passed as text for the LLM to interpret
      - Images: Tagged as [Contains N image(s)] (Groq has no vision support)
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")

        llm = ChatGroq(
            model=GROQ_SUMMARY_MODEL,
            temperature=0,
            api_key=api_key,
        )

        prompt_text = (
            "You are creating a searchable description for document content retrieval.\n\n"
            "CONTENT TO ANALYZE:\n"
            f"TEXT CONTENT:\n{text}\n\n"
        )

        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"

        if images:
            prompt_text += f"[This section contains {len(images)} image(s)/diagram(s)]\n\n"

        prompt_text += (
            "YOUR TASK:\n"
            "Generate a comprehensive, searchable description that covers:\n"
            "1. Key facts, numbers, and data points from text and tables\n"
            "2. Main topics and concepts discussed\n"
            "3. Questions this content could answer\n"
            "4. If tables are present, summarize their structure and key data\n"
            "5. Alternative search terms users might use\n\n"
            "Make it detailed and searchable — prioritize findability over brevity.\n\n"
            "SEARCHABLE DESCRIPTION:"
        )

        response = llm.invoke([HumanMessage(content=prompt_text)])
        return response.content

    except Exception as e:
        print(f"     ❌ AI summary failed: {e}")
        # Fallback to simple summary
        summary = f"{text[:500]}"
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary


# ── Steps 4+5+6: Process Chunks ─────────────────────────────────────────────
def process_chunks(chunks: list) -> List[Document]:
    """
    Process all chunks: separate modalities, create AI summaries for
    mixed-content chunks, and build LangChain Documents with rich metadata.
    """
    print("\n🧠 Processing chunks (separate → summarize → metadata)...")

    langchain_documents = []
    total = len(chunks)
    stats = {"text_only": 0, "with_tables": 0, "with_images": 0, "ai_summarized": 0}

    for i, chunk in enumerate(chunks):
        current = i + 1
        content_data = separate_content_types(chunk)

        has_tables = len(content_data["tables"]) > 0
        has_images = len(content_data["images"]) > 0

        # AI summary only for mixed-content chunks (tables or images)
        if has_tables or has_images:
            print(
                f"  📝 Chunk {current}/{total} — types: {content_data['types']} "
                f"(tables: {len(content_data['tables'])}, images: {len(content_data['images'])})"
            )
            print(f"     → Creating AI summary for mixed content...")

            enhanced_content = create_ai_summary(
                content_data["text"],
                content_data["tables"],
                content_data["images"],
            )
            stats["ai_summarized"] += 1
            print(f"     → AI summary created ({len(enhanced_content)} chars)")

            # Small delay between Groq calls to avoid rate limits
            time.sleep(2)
        else:
            enhanced_content = content_data["text"]
            stats["text_only"] += 1

        if has_tables:
            stats["with_tables"] += 1
        if has_images:
            stats["with_images"] += 1

        # Build LangChain Document with rich metadata (Step 6)
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "chunk_index": i,
                "content_types": json.dumps(content_data["types"]),
                "original_content": json.dumps({
                    "raw_text": content_data["text"],
                    "tables_html": content_data["tables"],
                    # Store image presence flag instead of full base64 to keep metadata lean.
                    # Full base64 can be very large and may hit ChromaDB metadata limits.
                    "has_images": len(content_data["images"]) > 0,
                    "image_count": len(content_data["images"]),
                }),
            },
        )
        langchain_documents.append(doc)

    print(f"\n  ✅ Processed {len(langchain_documents)} chunks")
    print(f"  📊 Stats: {stats}")

    return langchain_documents
