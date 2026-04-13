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
    "5. Context may include tables in HTML format. Interpret and reference "
    "table data (rows, columns, values) accurately when answering.\n"
    "6. Format your response in clean markdown for readability."
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
