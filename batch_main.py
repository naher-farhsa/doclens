import os
import asyncio
import re
from dotenv import load_dotenv

from src.ingestion import run_ingestion
from src.retrieval import load_vector_store, create_hybrid_retriever, batch_retrieve
from src.generation import load_model, rewrite_query_async, generate_answer_async

load_dotenv()

QUESTIONS_FILE = os.path.join("qna", "questions.txt")
ANSWERS_FILE   = os.path.join("qna", "answers.txt")

# Concurrency bottleneck as requested
semaphore = asyncio.Semaphore(2)

def clear_answers():
    """Wipes the answers file for a fresh start."""
    if os.path.exists(ANSWERS_FILE):
        with open(ANSWERS_FILE, "w", encoding="utf-8") as f:
            f.write("")
    print(f"🗑️  Cleared {ANSWERS_FILE}")

def parse_questions():
    """Parses qna/questions.txt into a list of question strings."""
    if not os.path.exists(QUESTIONS_FILE):
        print(f"❌ Error: {QUESTIONS_FILE} not found.")
        return []

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by "Number. " pattern
    # Questions are like: 1. In simple 2 lines... \n 2. Can you explain...
    questions = re.split(r'\n?\d+\.\s+', content)
    # Remove empty first element if it exists and strip whitespace
    questions = [q.strip() for q in questions if q.strip()]
    
    return questions

async def process_single_question(model, retriever, question_text, index):
    """Processes one question through the RAG pipeline."""
    async with semaphore:
        print(f"🔄 Processing Q{index}: {question_text[:50]}...")
        
        try:
            # 1. Planning (Async)
            plan = await rewrite_query_async(model, [], question_text)
            
            # 2. Retrieval (Sync, wrapped in a thread to keep loop free)
            relevant_docs = []
            if plan["action"] == "retrieve":
                # batch_retrieve is currently synchronous
                relevant_docs = await asyncio.to_thread(batch_retrieve, plan["queries"], retriever)
            
            # 3. Generation (Async)
            answer = await generate_answer_async(model, [], question_text, relevant_docs)
            
            return {
                "index": index,
                "question": question_text,
                "answer": answer
            }
        except Exception as e:
            print(f"❌ Error on Q{index}: {e}")
            return {
                "index": index,
                "question": question_text,
                "answer": f"Error: {e}"
            }

async def async_batch_process():
    """Main entry point for batch processing."""
    print("\n🚀 Starting Batch Processing (Concurrency: 2)...")
    
    # ── 1. Preparation ────────────────────────────────────────────────────────
    # Ensure ingestion is done (sync)
    run_ingestion()
    
    vectorstore = load_vector_store()
    retriever   = create_hybrid_retriever(vectorstore)
    model       = load_model()
    
    questions = parse_questions()
    if not questions:
        print("⚠️  No questions found to process.")
        return

    # ── 2. Run Tasks ──────────────────────────────────────────────────────────
    tasks = []
    for i, q in enumerate(questions, 1):
        tasks.append(process_single_question(model, retriever, q, i))
    
    results = await asyncio.gather(*tasks)
    
    # ── 3. Write Output ───────────────────────────────────────────────────────
    with open(ANSWERS_FILE, "a", encoding="utf-8") as f:
        for res in results:
            f.write(f"Question {res['index']}: {res['question']}\n")
            f.write(f"Answer: {res['answer']}\n")
            f.write("-" * 60 + "\n\n")
    
    print(f"\n✅ Finished! All responses saved to {ANSWERS_FILE}")

if __name__ == "__main__":
    asyncio.run(async_batch_process())
