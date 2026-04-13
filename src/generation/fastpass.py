from .constants import _NEGATIVE_PHRASES, _FAREWELL_PHRASES, HISTORY_WINDOW

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

def trim_history(chat_history: list) -> list:
    """Return only the last HISTORY_WINDOW messages to save tokens."""
    return chat_history[-HISTORY_WINDOW:] if len(chat_history) > HISTORY_WINDOW else chat_history
