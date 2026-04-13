"""
DocLens – Config / Key Manager
================================
Manages multiple API keys with round-robin rotation and
a reset mechanism so exhausted keys can be retried after a cooldown.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class KeyManager:
    """
    Holds a list of Google API keys and exposes helpers for rotation.

    Keys are read from environment variables:
        GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ... (up to 10)
    Falls back to the plain GOOGLE_API_KEY if no numbered keys are set.
    """

    def __init__(self):
        self.keys = self._load_keys()
        self.current_idx = 0

        if not self.keys:
            raise ValueError(
                "No Google API keys found. "
                "Set GOOGLE_API_KEY or GOOGLE_API_KEY_1..N in your .env file."
            )

        print(f"🔑 KeyManager initialised with {len(self.keys)} key(s).")

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_keys(self) -> list:
        """Read all available keys from environment variables."""
        keys = []

        # Try numbered keys first: GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ...
        for i in range(1, 11):
            key = os.getenv(f"GOOGLE_API_KEY_{i}")
            if key:
                keys.append(key)

        # Fall back to plain GOOGLE_API_KEY
        if not keys:
            key = os.getenv("GOOGLE_API_KEY")
            if key:
                keys.append(key)

        return keys

    # ── Public API ─────────────────────────────────────────────────────────

    def get_current_key(self) -> str:
        """Return the currently active API key."""
        return self.keys[self.current_idx]

    def rotate_key(self) -> bool:
        """
        Advance to the next key in the list.

        Returns:
            True  – a fresh key is now active.
            False – all keys have been cycled; caller should wait before retrying.
        """
        next_idx = self.current_idx + 1

        if next_idx < len(self.keys):
            self.current_idx = next_idx
            print(f"  🔄 API key rotated → Key {self.current_idx + 1}/{len(self.keys)}")
            return True

        print(f"  ❌ All {len(self.keys)} key(s) exhausted.")
        return False

    # ── BUG FIX ────────────────────────────────────────────────────────────
    # Previously there was NO reset method.  After the sleep() cooldown the
    # code looped back but current_idx was still pointing at the last
    # (rate-limited) key, so every retry immediately re-hit the same limit.
    #
    # reset_keys() rewinds the pointer to key 0 so the full rotation is
    # available again after a cooldown period.
    def reset_keys(self) -> None:
        """
        Reset the key index back to 0 after a cooldown wait.

        Call this immediately before reloading the model so that the next
        round of retries starts from the first key rather than the last
        exhausted one.
        """
        self.current_idx = 0
        print(f"  ♻️  Key index reset to 0. Rotation will restart from Key 1.")


# Singleton – imported everywhere as `from src.config import key_manager`
key_manager = KeyManager()