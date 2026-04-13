import os
import json
import time
import asyncio
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

from .constants import LLM_MODEL, MAX_CONTEXT_CHUNKS, SYSTEM_PROMPT, FAST_PASS_GREETINGS
from .fastpass import _fast_pass_response, trim_history

# ── Step 1: Load Model ────────────────────────────────────────────────────────
def load_model():
    """Initialise the Groq LLM."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Get a free key at https://console.groq.com and add it to .env"
        )
    print(f"\n🤖 Loading Groq model ({LLM_MODEL})...")
    return ChatGroq(model=LLM_MODEL, temperature=0.3, api_key=api_key)


# ── Step 3: Generate Answer ───────────────────────────────────────────────────
def generate_answer(model, chat_history: list, query: str, relevant_docs: list) -> str:
    """Generate an answer using Groq LLM with top retrieved chunks as context."""

    clean_input = query.lower().strip("?.!, ")
    if clean_input in FAST_PASS_GREETINGS:
        print("  ⚡ FastPass Response (local).")
        return _fast_pass_response(clean_input)

    top_docs = relevant_docs[:MAX_CONTEXT_CHUNKS]
    if len(relevant_docs) > MAX_CONTEXT_CHUNKS:
        print(f"  ✂️  Capping context: {len(relevant_docs)} chunks → {MAX_CONTEXT_CHUNKS}")

    # Build rich context: include table HTML from original_content if available
    context_parts = []
    for i, doc in enumerate(top_docs):
        part = f"**[Document {i+1}]**\n{doc.page_content}"

        # Append original table HTML so the LLM can see exact data
        if "original_content" in doc.metadata:
            try:
                original = json.loads(doc.metadata["original_content"])
                tables = original.get("tables_html", [])
                if tables:
                    part += "\n\n**Tables (raw data):**\n"
                    for j, table in enumerate(tables):
                        part += f"Table {j+1}:\n{table}\n"
            except (json.JSONDecodeError, KeyError):
                pass

        context_parts.append(part)

    context_block = "\n\n".join(context_parts)

    user_prompt    = f"**Context:**\n{context_block}\n\n**Question:** {query}"
    recent_history = trim_history(chat_history)

    messages = (
        [SystemMessage(content=SYSTEM_PROMPT)]
        + recent_history
        + [HumanMessage(content=user_prompt)]
    )

    for attempt in range(3):
        try:
            result = model.invoke(messages)
            return result.content

        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait_time = 20 * (attempt + 1)
                print(f"  ⏳ Groq rate limit hit (generate). Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  ❌ LLM error (generate): {e}")
                raise e

    return (
        "I'm sorry, I hit Groq's rate limit. "
        "Please wait a moment and try again, or switch LLM_MODEL to 'llama-3.1-8b-instant'."
    )


# ── Step 4: Generate Answer (Async) ──────────────────────────────────────────
async def generate_answer_async(model, chat_history: list, query: str, relevant_docs: list) -> str:
    """Async version of generate_answer for batch processing."""

    clean_input = query.lower().strip("?.!, ")
    if clean_input in FAST_PASS_GREETINGS:
        return _fast_pass_response(clean_input)

    top_docs = relevant_docs[:MAX_CONTEXT_CHUNKS]
    
    context_parts = []
    for i, doc in enumerate(top_docs):
        part = f"**[Document {i+1}]**\n{doc.page_content}"
        if "original_content" in doc.metadata:
            try:
                original = json.loads(doc.metadata["original_content"])
                tables = original.get("tables_html", [])
                if tables:
                    part += "\n\n**Tables (raw data):**\n"
                    for j, table in enumerate(tables):
                        part += f"Table {j+1}:\n{table}\n"
            except (json.JSONDecodeError, KeyError):
                pass
        context_parts.append(part)

    context_block = "\n\n".join(context_parts)
    user_prompt    = f"**Context:**\n{context_block}\n\n**Question:** {query}"
    recent_history = trim_history(chat_history)

    messages = (
        [SystemMessage(content=SYSTEM_PROMPT)]
        + recent_history
        + [HumanMessage(content=user_prompt)]
    )

    for attempt in range(3):
        try:
            result = await model.ainvoke(messages)
            return result.content
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait_time = 20 * (attempt + 1)
                await asyncio.sleep(wait_time)
                continue
            else:
                raise e

    return "I'm sorry, I hit Groq's rate limit. Please try again later."
