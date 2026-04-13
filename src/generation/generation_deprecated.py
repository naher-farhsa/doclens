"""
DocLens – Generation Pipeline
===============================
Gemini 2.0 Flash LLM with ChatML-style prompting and history-aware query rewriting.

Bugs fixed / improvements in this file
----------------------------------------
BUG 1 (rewrite_query)   – After sleeping for quota cooldown, the old model
                           object (holding the exhausted key) was reused.
                           Fixed: call reset_keys() + load_model() after sleep.

BUG 2 (generate_answer) – Same root cause as Bug 1 in the generate path.
                           Fixed: same pattern – reset + reload after sleep.

BUG 3 (load_model)      – bind_tools() called with a plain dict caused a
                           missing-module crash. Removed entirely since the
                           RAG pipeline handles retrieval via ChromaDB + BM25.

BUG 4 (rewrite_query)   – Short-query skip block had wrong indentation.
                           Fixed: print + return are now inside the if block.

BUG 5 (generate_answer) – Key rotation fired the next key immediately with
                           no delay, so all keys exhausted quota in < 1 second.
                           Fixed: added KEY_ROTATION_DELAY sleep after each
                           successful rotation before retrying.

IMPROVEMENT 1 (both)    – Full chat_history grew unboundedly wasting tokens.
                           Fixed: trim to last HISTORY_WINDOW (2) messages.

IMPROVEMENT 2 (generate) – All retrieved chunks (up to 6) were sent to the
                            LLM. Heavy on tokens, slow, hits quota faster.
                            Fixed: cap at MAX_CONTEXT_CHUNKS (3) top chunks.
"""

import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────
LLM_MODEL = "gemini-2.0-flash"

# Seconds to wait after rotating to a new key before firing the next request.
# Gives the new key's quota window a moment to be clean.
# Free-tier RPM resets every 60s — 12s covers ~1/5 of the window safely.
KEY_ROTATION_DELAY = 12

# Max number of retrieved chunks to include in the LLM prompt.
# 3 is enough for most queries; 6 costs ~2x tokens with little gain.
MAX_CONTEXT_CHUNKS = 3

# How many recent chat messages to send (1 human + 1 AI turn = 2).
HISTORY_WINDOW = 2

SYSTEM_PROMPT = (
    "You are **DocLens**, an intelligent document analysis assistant.\n\n"
    "Rules:\n"
    "1. For generic greetings, social interactions (e.g., 'Hi', 'How are you?'), "
    "or general bot-related questions, reply directly and politely without "
    "referencing documents or search.\n"
    "2. For document-related questions, answer using ONLY the provided context "
    "documents first.\n"
    "3. If the context does not contain enough information for a document query, "
    "say so clearly and provide the best answer you can from your training knowledge.\n"
    "4. Be concise, accurate, and cite specifics from the context when possible.\n"
    "5. Format your response in clean markdown for readability."
)

from src.config import key_manager


# ── History Helper ───────────────────────────────────────────────────────────
def trim_history(chat_history: list) -> list:
    """Return only the last HISTORY_WINDOW messages from the full history."""
    return chat_history[-HISTORY_WINDOW:] if len(chat_history) > HISTORY_WINDOW else chat_history


# ── Step 1: Load Model ───────────────────────────────────────────────────────
def load_model():
    """Initialise the Gemini model with the currently active API key."""
    api_key = key_manager.get_current_key()
    print(f"\n🤖 Loading Gemini model (API Key {key_manager.current_idx + 1})...")
    model = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.3,
        google_api_key=api_key,
    )
    return model


# ── Step 2: History-Aware Query Rewriting ────────────────────────────────────
def rewrite_query(model, chat_history: list, user_question: str) -> list:
    """
    Rewrite user question into a standalone search query.

    Returns ["SKIP_RETRIEVAL"] for greetings / social queries.
    Returns a list with one rewritten search query otherwise.
    """

    # ── FastPass: hardcoded greetings (zero API calls) ───────────────────────
    fast_pass_greetings = {
        "hi", "hello", "hey", "how are you", "good morning",
        "good afternoon", "good evening", "what's up", "yo", "greetings",
    }
    clean_input = user_question.lower().strip("?.! ")
    if clean_input in fast_pass_greetings:
        print("  ⚡ FastPass triggered (local check). Skipping RAG pipeline.")
        return ["SKIP_RETRIEVAL"]

    # Short queries with no history don't need an LLM rewrite
    if len(user_question.split()) <= 6 and not chat_history:
        print("  ⚡ Short query – skipping rewrite to save quota.")
        return [user_question]

    recent_history = trim_history(chat_history)

    messages = [
        SystemMessage(
            content=(
                "Analyze the user's new question and the conversation history.\n"
                "1. If the question is a generic greeting (like 'hi', 'hello'), a social "
                "pleasantry, or doesn't require document search (e.g., 'who are you?'), "
                "return exactly: SKIP_RETRIEVAL\n"
                "2. Otherwise, rewrite the user's new question into ONE standalone, "
                "search-friendly version. Return ONLY the rewritten question."
            )
        ),
    ] + recent_history + [
        HumanMessage(content=f"New question: {user_question}"),
    ]

    max_retries = len(key_manager.keys) * 2

    for attempt in range(max_retries):
        try:
            result = model.invoke(messages)
            content = result.content.strip()

            if "SKIP_RETRIEVAL" in content:
                print("  💬 General query detected. Skipping RAG pipeline.")
                return ["SKIP_RETRIEVAL"]

            print(f"  🔄 Rewritten query (Key {key_manager.current_idx + 1}): {content}")
            return [content]

        except Exception as e:
            if "429" in str(e) or "Resource has been exhausted" in str(e):
                if key_manager.rotate_key():
                    # ── BUG 5 FIX (rewrite path) ─────────────────────────────
                    # BEFORE: model = load_model(); continue
                    #         → fired new key instantly, quota hit again
                    # AFTER:  sleep KEY_ROTATION_DELAY seconds first so the
                    #         new key's quota window has breathing room.
                    # ─────────────────────────────────────────────────────────
                    print(f"  ⏳ Cooling down {KEY_ROTATION_DELAY}s before using new key...")
                    time.sleep(KEY_ROTATION_DELAY)   # ← BUG 5 FIX
                    model = load_model()
                    continue
                else:
                    # All keys gone — wait a full minute then reset
                    wait_time = 60 * (attempt + 1)
                    print(f"  ⏳ All keys exhausted (rewrite). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    key_manager.reset_keys()
                    model = load_model()
                    continue
            else:
                print(f"  ⚠️  LLM error ({e}). Falling back to original query.")
                return [user_question]

    return [user_question]


# ── Step 3: Generate Answer ──────────────────────────────────────────────────
def generate_answer(model, chat_history: list, query: str, relevant_docs: list) -> str:
    """Generate an answer with retry and API key rotation for 429s."""

    # ── FastPass: hardcoded greetings (zero API calls) ───────────────────────
    fast_pass_greetings = {
        "hi", "hello", "hey", "how are you", "good morning",
        "good afternoon", "good evening", "what's up", "yo", "greetings",
    }
    clean_input = query.lower().strip("?.! ")
    if clean_input in fast_pass_greetings:
        print("  ⚡ FastPass Response (local).")
        return "Hello! I'm your DocLens assistant. How can I help you with your documents today?"

    # ── IMPROVEMENT 2: cap chunks to reduce token usage ──────────────────────
    # BEFORE: all relevant_docs sent (up to 6) → heavy prompt, burns quota fast
    # AFTER:  only top MAX_CONTEXT_CHUNKS (3) sent → lighter, faster, cheaper
    # ─────────────────────────────────────────────────────────────────────────
    top_docs = relevant_docs[:MAX_CONTEXT_CHUNKS]
    if len(relevant_docs) > MAX_CONTEXT_CHUNKS:
        print(f"  ✂️  Capping context: {len(relevant_docs)} chunks → {MAX_CONTEXT_CHUNKS}")

    context_block = "\n\n".join(
        [f"**[Document {i+1}]**\n{doc.page_content}" for i, doc in enumerate(top_docs)]
    )

    user_prompt = f"**Context:**\n{context_block}\n\n**Question:** {query}"
    recent_history = trim_history(chat_history)

    messages = (
        [SystemMessage(content=SYSTEM_PROMPT)]
        + recent_history
        + [HumanMessage(content=user_prompt)]
    )

    max_retries = len(key_manager.keys) * 2

    for attempt in range(max_retries):
        try:
            result = model.invoke(messages)
            return result.content

        except Exception as e:
            if "429" in str(e) or "Resource has been exhausted" in str(e):
                if key_manager.rotate_key():
                    # ── BUG 5 FIX (generate path) ────────────────────────────
                    # BEFORE: model = load_model(); continue
                    #         → all 3 keys fired back-to-back in < 1 second,
                    #           every one of them hit 429, then waited 135s.
                    # AFTER:  sleep KEY_ROTATION_DELAY before using new key
                    #         so its quota window is not already exhausted.
                    # ─────────────────────────────────────────────────────────
                    print(f"  ⏳ Cooling down {KEY_ROTATION_DELAY}s before using new key...")
                    time.sleep(KEY_ROTATION_DELAY)   # ← BUG 5 FIX
                    model = load_model()
                    continue
                else:
                    # All keys gone — wait a full minute then reset
                    wait_time = 60 * (attempt + 1)
                    print(f"  ⏳ All keys exhausted (generate). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    key_manager.reset_keys()
                    model = load_model()
                    continue
            else:
                print(f"  ❌ LLM error (generate): {e}")
                raise e

    return (
        "I'm sorry, I hit a persistent rate limit even after rotating keys and waiting. "
        "Please try again in a few minutes."
    )