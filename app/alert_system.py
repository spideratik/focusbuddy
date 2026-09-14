"""
alert_system.py

Decides WHAT alert to fire and WHEN, during ACTIVE mode.

  Corrective (LOOKING_AWAY, PLAYING_WITH_PHONE, AWAY_FROM_DESK, SLEEPING_DROWSY)
      Fires after the situation has been sustained for
      config.INTERVENTION_TRIGGER_THRESHOLD seconds (handled in
      session_manager.py), then respects its own cooldown.

  Encouragement (READING, WRITING, MEMORIZING)
      Fires once when the child transitions INTO one of these activities,
      respecting a separate, longer cooldown.

  TALKING is deliberately neutral - no alert either way. We don't know if
  the child is discussing the material or something unrelated, so this
  build doesn't guess and doesn't interrupt.

Each situation has its own cooldown timer, so encouragement and corrective
alerts never block each other.

Delivery: a real recorded audio clip the parent has placed (or recorded
in-app) under static/audio/<user_id>/<situation>/ is preferred. If none
exists yet, a browser-spoken fallback phrase is returned instead (Web
Speech Synthesis - clearly synthetic, never a voice clone).
"""

import os
import time
import random

SITUATION_FOLDERS = {
    "LOOKING_AWAY": "looking_away",
    "PLAYING_WITH_PHONE": "playing_with_phone",
    "AWAY_FROM_DESK": "away_from_desk",
    "SLEEPING_DROWSY": "sleeping_drowsy",
    "TALKING": "talking",
    "READING": "reading",
    "WRITING": "writing",
    "MEMORIZING": "memorizing",
}

CORRECTIVE_SITUATIONS = {"LOOKING_AWAY", "PLAYING_WITH_PHONE", "AWAY_FROM_DESK", "SLEEPING_DROWSY", "TALKING"}
ENCOURAGEMENT_SITUATIONS = {"READING", "WRITING", "MEMORIZING"}

AUDIO_EXTENSIONS = (".mp3", ".wav", ".ogg", ".m4a", ".webm")

STATIC_AUDIO_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "static", "audio"
)

FALLBACK_PHRASES = {
    "LOOKING_AWAY": [
        "{name}, চলো আবার বইয়ের দিকে মনোযোগ দিই। তুমি খুব ভালো করছো!",
        "{name}, একটু মনোযোগ ফিরিয়ে আনি, তুমি পারবে।",
    ],
    "PLAYING_WITH_PHONE": [
        "{name}, ওটা একটু রেখে দাও, চলো এই পাতাটা শেষ করি।",
        "{name}, চলো আবার পড়ায় ফিরে যাই।",
    ],
    "AWAY_FROM_DESK": [
        "{name}, তুমি কি ফিরে আসতে পারবে? চলো আবার শুরু করি।",
        "{name}, ডেস্কে ফিরে আসি, তুমি পারবে।",
    ],
    "SLEEPING_DROWSY": [
        "{name}, একটু ক্লান্ত লাগছে বুঝি? একটা ছোট বিরতি নাও।",
        "{name}, চোখেমুখে পানি দিয়ে আসো, তারপর আবার শুরু করি।",
    ],
    "TALKING": [
        "{name}, একটু শান্ত হয়ে আবার পড়ায় মনোযোগ দিই।",
        "{name}, চলো আবার বইয়ের দিকে ফিরি।",
    ],
    "READING": [
        "{name}, তুমি খুব ভালো পড়ছো। এভাবেই চালিয়ে যাও!",
        "চমৎকার, {name}!",
    ],
    "WRITING": [
        "{name}, খুব সুন্দর লিখছো, চালিয়ে যাও!",
        "{name}, এভাবেই করতে থাকো।",
    ],
    "MEMORIZING": [
        "{name}, একটু থেমে মনে করাটাও ঠিক আছে। তুমি ভালো করছো।",
        "ঠিক আছে {name}, একটু ভেবে নাও।",
    ],
}


def _list_clips(user_id, situation):
    folder = SITUATION_FOLDERS.get(situation)
    if not folder:
        return []
    dir_path = os.path.join(STATIC_AUDIO_ROOT, str(user_id), folder)
    if not os.path.isdir(dir_path):
        return []
    return [
        f"/static/audio/{user_id}/{folder}/{fname}"
        for fname in os.listdir(dir_path)
        if fname.lower().endswith(AUDIO_EXTENSIONS)
    ]


class AlertSystem:
    def __init__(self, user_id, child_name, language="en",
                 corrective_cooldown_seconds=45,
                 encouragement_cooldown_seconds=90):
        self.user_id = user_id
        self.child_name = child_name
        self.language = language
        self.corrective_cooldown_seconds = corrective_cooldown_seconds
        self.encouragement_cooldown_seconds = encouragement_cooldown_seconds
        self._last_alert_time = {}  # situation -> timestamp, independent per situation

    def _cooldown_ok(self, situation):
        cooldown = (
            self.corrective_cooldown_seconds
            if situation in CORRECTIVE_SITUATIONS
            else self.encouragement_cooldown_seconds
        )
        last = self._last_alert_time.get(situation, 0)
        return (time.time() - last) >= cooldown

    def _build_payload(self, situation):
        clips = _list_clips(self.user_id, situation)
        self._last_alert_time[situation] = time.time()

        if clips:
            return {"audio_url": random.choice(clips), "text": None, "situation": situation}

        phrase = random.choice(FALLBACK_PHRASES[situation]).format(name=self.child_name)
        return {"audio_url": None, "text": phrase, "situation": situation}

    def trigger_alert(self, situation):
        """Corrective alert. Caller (session_manager) already enforces the
        sustained-duration requirement; this only enforces the cooldown."""
        if situation not in CORRECTIVE_SITUATIONS:
            return None
        if not self._cooldown_ok(situation):
            return None
        return self._build_payload(situation)

    def trigger_encouragement(self, situation):
        """Praise alert, fired on transition into the activity."""
        if situation not in ENCOURAGEMENT_SITUATIONS:
            return None
        if not self._cooldown_ok(situation):
            return None
        return self._build_payload(situation)
