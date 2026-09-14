"""
dialogue_listener.py

Speech-to-text now happens client-side in the browser (Web Speech
Recognition API - see static/app.js), since that's the realistic place to
capture microphone audio for a web app and keeps raw audio off the server
entirely. This module just wraps already-transcribed text for logging.
"""

import time


class DialogueListener:
    """Kept as a thin wrapper so SessionManager's interface doesn't change,
    even though transcription itself now happens in the browser."""

    def __init__(self, language="en-US"):
        self.language = language

    def package_transcript(self, text):
        if not text or not text.strip():
            return None
        return (time.time(), text.strip())
