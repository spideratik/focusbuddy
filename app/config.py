"""
FocusBuddy configuration.

No biometric templates (face vectors, voice clones) are persisted.
Everything here is either a runtime tunable or a per-session, in-memory value.
"""

# --- Timing thresholds (seconds) ---
DISTRACTED_GAZE_AWAY_THRESHOLD = 15      # eyes-away this long -> LOOKING_AWAY
AWAY_FROM_DESK_THRESHOLD = 15             # face missing this long -> AWAY_FROM_DESK (else UNKNOWN, brief)
DROWSY_STILLNESS_THRESHOLD = 45           # eyes-away AND near-zero motion this long -> SLEEPING_DROWSY
INTERVENTION_TRIGGER_THRESHOLD = 15      # sustained "distracted" decision before an alert fires
ALERT_COOLDOWN_SECONDS = 45              # minimum gap between two corrective alerts
POSITIVE_ALERT_COOLDOWN_SECONDS = 90     # minimum gap between two praise/encouragement alerts
FRAME_ANALYSIS_INTERVAL = 5              # classify behavior every N seconds

# --- Activity detection (heuristic, see README for accuracy caveats) ---
WRITING_MOTION_THRESHOLD = 12             # mean pixel diff -> guess "hand/pen motion" (writing)
LOW_MOTION_THRESHOLD = 3                  # below this, treat as "essentially still" (for drowsy detection)

# --- Language / personalization (set per child profile, not per parent voice) ---
CHILD_NAME = "Alex"
LANGUAGE = "bn-BD"

# --- Privacy switches ---
STORE_RAW_VIDEO = False        # must stay False; only rolling in-memory buffer allowed
STORE_AUDIO_CLIPS = False      # transcribe in-memory, discard audio after transcription
PERSIST_FACIAL_VECTORS = False # explicitly disabled - no biometric templates saved
USE_VOICE_CLONING = False      # explicitly disabled - see README for rationale

# --- Report ---
REPORT_OUTPUT_DIR = "reports"
