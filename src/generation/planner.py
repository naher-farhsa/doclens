import time
import asyncio
from langchain_core.messages import SystemMessage, HumanMessage
from .constants import FAST_PASS_GREETINGS
from .fastpass import trim_history

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
# ── Step 3: Planner (Async) ──────────────────────────────────────────────────
async def rewrite_query_async(model, chat_history: list, user_question: str) -> dict:
    """Async version of rewrite_query."""
    clean_input = user_question.lower().strip("?.!, ")

    if clean_input in FAST_PASS_GREETINGS:
        return {"action": "skip", "queries": []}

    if len(user_question.split()) <= 6 and not chat_history:
        return {"action": "retrieve", "queries": [user_question]}

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
            result  = await model.ainvoke(messages)
            content = result.content.strip()

            if content.startswith("ACTION:skip"):
                return {"action": "skip", "queries": []}

            if content.startswith("ACTION:retrieve"):
                query_line = next(
                    (line for line in content.splitlines() if line.startswith("QUERY:")),
                    None,
                )
                rewritten = query_line.replace("QUERY:", "").strip() if query_line else user_question
                return {"action": "retrieve", "queries": [rewritten]}

            return {"action": "retrieve", "queries": [user_question]}

        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait_time = 20 * (attempt + 1)
                await asyncio.sleep(wait_time)
                continue
            else:
                return {"action": "retrieve", "queries": [user_question]}

    return {"action": "retrieve", "queries": [user_question]}
