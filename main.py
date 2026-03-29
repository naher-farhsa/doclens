"""
DocLens – RAG Document Chat Bot
=================================
CLI entry point that ties the ingestion, retrieval, and generation pipelines together.

Usage:
    python main.py

REFACTOR: Structured planner output
-------------------------------------
rewrite_query() now returns a plan dict instead of a list with a sentinel string.

BEFORE (broken design):
    search_queries = rewrite_query(model, chat_history, user_input)

    if search_queries == ["SKIP_RETRIEVAL"]:    ← magic string comparison
        relevant_docs = []
    else:
        relevant_docs = batch_retrieve(search_queries, retriever)

    if "SKIP_RETRIEVAL" not in search_queries:  ← magic string check again
        if relevant_docs: ...

AFTER (clean design):
    plan = rewrite_query(model, chat_history, user_input)
    # plan = {"action": "skip",     "queries": []}
    # plan = {"action": "retrieve", "queries": ["rewritten query"]}

    if plan["action"] == "retrieve":
        relevant_docs = batch_retrieve(plan["queries"], retriever)
    else:
        relevant_docs = []

    No magic strings. No sentinel values. Control flow is explicit and readable.
"""

import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from src.ingestion import run_ingestion
from src.retrieval import load_vector_store, create_hybrid_retriever, batch_retrieve
from src.generation import load_model, rewrite_query, generate_answer

load_dotenv()


def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "  🔍 DocLens – RAG Document Chat Bot  ".center(57) + "║")
    print("╚" + "═" * 58 + "╝")

    # ── 1. Ingestion (skip if DB exists) ──────────────────────────────────────
    run_ingestion()

    # ── 2. Load vector store + hybrid retriever ───────────────────────────────
    vectorstore = load_vector_store()
    retriever   = create_hybrid_retriever(vectorstore)

    # ── 3. Load LLM ───────────────────────────────────────────────────────────
    model = load_model()

    # ── 4. Interactive chat loop ──────────────────────────────────────────────
    chat_history = []

    print("\n" + "─" * 60)
    print("  💬 Chat started! Ask me anything about your documents.")
    print("     Type 'quit' or 'exit' to end the session.")
    print("     Type 'clear' to reset conversation history.")
    print("─" * 60)

    while True:
        try:
            user_input = input("\n🧑 You:\n ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("\n👋 Goodbye! Thanks for using DocLens.")
            break
        if user_input.lower() == "clear":
            chat_history.clear()
            print("🗑️  Conversation history cleared.")
            continue

        # ── Step A: Planning ──────────────────────────────────────────────────
        print("\n 🤔 Thinking..", end="", flush=True)
        time.sleep(0.5)
        print("\r 📋 Planning..", end="", flush=True)
        time.sleep(0.5)

        # REFACTOR: rewrite_query now returns a plan dict, not a list
        #
        # BEFORE:
        #   search_queries = rewrite_query(model, chat_history, user_input)
        #   if search_queries == ["SKIP_RETRIEVAL"]: ...
        #
        # AFTER:
        #   plan = rewrite_query(model, chat_history, user_input)
        #   if plan["action"] == "retrieve": ...
        
        plan = rewrite_query(model, chat_history, user_input)

        # ── Step B: Retrieval (only if planner says so) ───────────────────────
        if plan["action"] == "retrieve":
            print("\r  🔍 Searching..", end="", flush=True)
            time.sleep(0.5)
            print("\r  📥 Retrieving...", end="", flush=True)

            relevant_docs = batch_retrieve(plan["queries"], retriever)

            if relevant_docs:
                print(f"\r  📚 Found {len(relevant_docs)} relevant chunk(s)")
            else:
                print("\r  ⚠️  No relevant documents found.")
        else:
            # action == "skip" — greeting/social/no retrieval needed
            relevant_docs = []

        # ── Step C: Generation ────────────────────────────────────────────────
        print("  🧠 Generating answer...", end="", flush=True)
        answer = generate_answer(model, chat_history, user_input, relevant_docs)
        print(f"\n🤖 DocLens:\n{answer}")

        # ── Step D: Append to history ─────────────────────────────────────────
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")