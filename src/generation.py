"""
DocLens – Generation Pipeline
===============================
Groq LLM (Llama 3.3 70B) + history-aware query rewriting.

REFACTOR: Structured planner output (full cleanup)
----------------------------------------------------
BEFORE — magic strings leaked across two layers:
    Layer 1: LLM prompt told model to say "SKIP_RETRIEVAL"
    Layer 2: Code checked  if "SKIP_RETRIEVAL" in content
    Layer 3: Return value was ["SKIP_RETRIEVAL"]
    Layer 4: main.py checked search_queries == ["SKIP_RETRIEVAL"]

AFTER — one consistent contract end-to-end:
    LLM prompt   → instructs model to return "ACTION:skip" or "ACTION:retrieve\nQUERY:..."
    Code parses  → content.startswith("ACTION:skip") — no magic string
    Return value → {"action": "skip", "queries": []}  or
                   {"action": "retrieve", "queries": ["rewritten query"]}
    main.py reads → plan["action"] == "retrieve"

    The word "SKIP_RETRIEVAL" exists nowhere in this file.
"""

import os
import time
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────
LLM_MODEL          = "llama-3.3-70b-versatile"
MAX_CONTEXT_CHUNKS = 3
HISTORY_WINDOW     = 2

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

# ── FastPass set (shared by rewrite_query + generate_answer) ─────────────────
FAST_PASS_GREETINGS = {
    # Greetings
    "hi", "hello", "hey", "how are you", "good morning",
    "good afternoon", "good evening", "what's up", "yo",
    "greetings", "howdy", "sup", "hiya", "heya",
    # Casual / positive
    "great", "good", "awesome", "fantastic", "nice", "not bad",
    "all good", "how's it going", "how are you doing",
    "how do you do", "good day", "pleased to meet you",
    # Bot-directed
    "who are you", "what are you", "what can you do",
    "are you a bot", "are you an ai", "what is doclens",
    # Negative / frustrated (social — skip RAG)
    "this is useless", "you are useless", "you suck",
    "this doesn't work", "terrible bot", "worst bot",
    # Conversation enders
    "ok", "sure", "got it", "bye", "goodbye", "see you",
    "take care", "thanks", "thank you", "ok bye",
    "that's all", "nevermind", "forget it",
}

_NEGATIVE_PHRASES = {"this is useless", "you are useless", "you suck",
                     "this doesn't work", "terrible bot", "worst bot"}
_FAREWELL_PHRASES = {"bye", "goodbye", "see you", "take care", "ok bye", "that's all"}


# ── FastPass Response ─────────────────────────────────────────────────────────
def _fast_pass_response(clean_input: str) -> str:
    """Return the appropriate canned reply for a FastPass hit."""
    if clean_input in _NEGATIVE_PHRASES:
        return (
            "Sorry to hear that! Try rephrasing your question or check "
            "that your documents are loaded correctly."
        )
    if clean_input in _FAREWELL_PHRASES:
        return "Goodbye! Feel free to come back anytime. 👋"
    return "Hello! I'm your DocLens assistant. How can I help you with your documents today?"


# ── History Helper ────────────────────────────────────────────────────────────
def trim_history(chat_history: list) -> list:
    """Return only the last HISTORY_WINDOW messages to save tokens."""
    return chat_history[-HISTORY_WINDOW:] if len(chat_history) > HISTORY_WINDOW else chat_history


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


# ── Step 2: Planner — rewrites query and decides action ──────────────────────
def rewrite_query(model, chat_history: list, user_question: str) -> dict:
    """
    Analyse the user question and return a plan dict:

        {"action": "skip",     "queries": []}
            → greeting / social / no retrieval needed

        {"action": "retrieve", "queries": ["rewritten standalone query"]}
            → document question; retriever should run

    LLM prompt format:
        The model is instructed to respond in one of two exact formats:
            ACTION:skip
            — or —
            ACTION:retrieve
            QUERY:<rewritten question>

        This keeps the LLM response structured and parseable without
        any magic sentinel strings leaking into the codebase.
    """

    clean_input = user_question.lower().strip("?.!, ")

    # ── FastPass: zero API calls ──────────────────────────────────────────────
    if clean_input in FAST_PASS_GREETINGS:
        print("  ⚡ FastPass triggered (local check). Skipping RAG pipeline.")
        return {"action": "skip", "queries": []}

    # ── Short query optimisation: skip LLM rewrite ───────────────────────────
    if len(user_question.split()) <= 6 and not chat_history:
        print("  ⚡ Short query – skipping rewrite to save quota.")
        return {"action": "retrieve", "queries": [user_question]}

    # ── LLM rewrite ──────────────────────────────────────────────────────────
    recent_history = trim_history(chat_history)

    messages = [
        SystemMessage(
            content=(
                "Analyze the user's new question and the conversation history.\n\n"
                "Respond in EXACTLY one of these two formats — nothing else:\n\n"
                "Format 1 (greeting, social, or no document search needed):\n"
                "ACTION:skip\n\n"
                "Format 2 (document question that needs search):\n"
                "ACTION:retrieve\n"
                "QUERY:<rewritten standalone search-friendly version of the question>\n\n"
                "Rules:\n"
                "- Use ACTION:skip for greetings, pleasantries, bot-directed questions.\n"
                "- Use ACTION:retrieve for anything that needs searching documents.\n"
                "- For ACTION:retrieve, the QUERY line must be a complete standalone "
                "question that makes sense without the conversation history.\n"
                "- Output ONLY the format above. No explanation, no extra text."
            )
        ),
    ] + recent_history + [
        HumanMessage(content=f"New question: {user_question}"),
    ]

    for attempt in range(3):
        try:
            result  = model.invoke(messages)
            content = result.content.strip()

            # ── Parse ACTION:skip ─────────────────────────────────────────────
            if content.startswith("ACTION:skip"):
                print("  💬 General query detected. Skipping RAG pipeline.")
                return {"action": "skip", "queries": []}

            # ── Parse ACTION:retrieve ─────────────────────────────────────────
            if content.startswith("ACTION:retrieve"):
                query_line = next(
                    (line for line in content.splitlines() if line.startswith("QUERY:")),
                    None,
                )
                rewritten = query_line.replace("QUERY:", "").strip() if query_line else user_question
                print(f"  🔄 Rewritten query: {rewritten}")
                return {"action": "retrieve", "queries": [rewritten]}

            # ── Malformed response: model ignored the format ──────────────────
            # Default to retrieve with the original question so the user
            # still gets an answer rather than a silent skip.
            print(f"  ⚠️  Unexpected planner format. Defaulting to retrieve.")
            print(f"      Model said: {content[:80]}")
            return {"action": "retrieve", "queries": [user_question]}

        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait_time = 20 * (attempt + 1)
                print(f"  ⏳ Groq rate limit hit (rewrite). Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  ⚠️  LLM error ({e}). Falling back to original query.")
                return {"action": "retrieve", "queries": [user_question]}

    # Exhausted retries — fall back to original question
    return {"action": "retrieve", "queries": [user_question]}


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

    context_block = "\n\n".join(
        [f"**[Document {i+1}]**\n{doc.page_content}" for i, doc in enumerate(top_docs)]
    )

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