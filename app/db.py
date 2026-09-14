"""
db.py

Stores accounts (for multi-family use) and, per family, only what's needed
for the parent-facing history: per-session aggregate metrics, the generated
report text, and transcribed dialogue lines. No video, no audio, no
biometric data ever reaches this database.

CRITICAL: every session/history query is scoped by user_id. This is the
one thing standing between "families each see their own data" and "one
family sees another family's child's transcripts" - never remove the
user_id filter from a query here without thinking hard about why.
"""

import sqlite3
import os
import time
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "focusbuddy.db")


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                child_name TEXT NOT NULL,
                started_at REAL NOT NULL,
                ended_at REAL NOT NULL,
                focus_seconds REAL NOT NULL,
                distraction_seconds REAL NOT NULL,
                disruption_seconds REAL NOT NULL,
                report_text TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        # Lightweight migrations - safe to run every startup; SQLite errors
        # on a duplicate column, which we ignore.
        for column, coltype in (
            ("reading_seconds", "REAL DEFAULT 0"),
            ("writing_seconds", "REAL DEFAULT 0"),
            ("memorizing_seconds", "REAL DEFAULT 0"),
            ("looking_away_seconds", "REAL DEFAULT 0"),
            ("playing_with_phone_seconds", "REAL DEFAULT 0"),
            ("talking_seconds", "REAL DEFAULT 0"),
            ("away_from_desk_seconds", "REAL DEFAULT 0"),
            ("sleeping_drowsy_seconds", "REAL DEFAULT 0"),
            ("unknown_seconds", "REAL DEFAULT 0"),
            ("user_id", "INTEGER DEFAULT 0"),  # for pre-multi-user databases
        ):
            try:
                conn.execute(f"ALTER TABLE sessions ADD COLUMN {column} {coltype}")
            except sqlite3.OperationalError:
                pass

        conn.execute("""
            CREATE TABLE IF NOT EXISTS dialogue_lines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                text TEXT NOT NULL,
                context TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                head_direction TEXT,
                face_detected INTEGER,
                person_detected INTEGER,
                pose TEXT,
                hand_position TEXT,
                book_detected INTEGER,
                phone_detected INTEGER,
                activity TEXT NOT NULL,
                distraction_score INTEGER,
                intervention_given TEXT,
                parent_feedback TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
        """)


# ---------- Users ----------

def create_user(username, password):
    username = username.strip().lower()
    if not username or not password:
        return {"ok": False, "error": "Username and password are required."}
    if len(password) < 8:
        return {"ok": False, "error": "Password must be at least 8 characters."}

    with _connect() as conn:
        existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            return {"ok": False, "error": "That username is already taken."}
        password_hash = generate_password_hash(password)
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, time.time())
        )
        return {"ok": True, "user_id": cur.lastrowid, "username": username}


def verify_user(username, password):
    username = username.strip().lower()
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if not row or not check_password_hash(row["password_hash"], password):
            return None
        return {"id": row["id"], "username": row["username"]}


def get_user(user_id):
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT id, username FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


# ---------- Sessions (scoped by user_id) ----------

def start_session(user_id, child_name, started_at):
    """Creates a placeholder session row immediately when a session starts,
    so live session_events can attach to a real session_id via foreign key
    throughout the session - finish_session() fills in the aggregates once
    it ends."""
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO sessions
               (user_id, child_name, started_at, ended_at, focus_seconds,
                distraction_seconds, disruption_seconds, report_text)
               VALUES (?, ?, ?, 0, 0, 0, 0, '')""",
            (user_id, child_name, started_at)
        )
        return cur.lastrowid


def finish_session(session_id, ended_at, state_seconds, dialogue_log, report_text):
    """
    state_seconds: dict covering all 9 states from app/vision_pipeline.py
    (READING, WRITING, MEMORIZING, LOOKING_AWAY, PLAYING_WITH_PHONE,
    TALKING, AWAY_FROM_DESK, SLEEPING_DROWSY, UNKNOWN) -> seconds spent.
    The legacy focus/distraction/disruption columns are kept as derived
    summaries (GOOD_STATES total / a couple of PROBLEM_STATES groupings)
    so older report-reading code still works.
    """
    good = state_seconds.get("READING", 0) + state_seconds.get("WRITING", 0) + state_seconds.get("MEMORIZING", 0)
    distraction = state_seconds.get("LOOKING_AWAY", 0) + state_seconds.get("TALKING", 0) + state_seconds.get("SLEEPING_DROWSY", 0)
    disruption = state_seconds.get("AWAY_FROM_DESK", 0) + state_seconds.get("PLAYING_WITH_PHONE", 0)

    with _connect() as conn:
        conn.execute(
            """UPDATE sessions SET
                 ended_at = ?, focus_seconds = ?, distraction_seconds = ?,
                 disruption_seconds = ?, report_text = ?,
                 reading_seconds = ?, writing_seconds = ?, memorizing_seconds = ?,
                 looking_away_seconds = ?, playing_with_phone_seconds = ?,
                 talking_seconds = ?, away_from_desk_seconds = ?,
                 sleeping_drowsy_seconds = ?, unknown_seconds = ?
               WHERE id = ?""",
            (ended_at, good, distraction, disruption, report_text,
             state_seconds.get("READING", 0), state_seconds.get("WRITING", 0),
             state_seconds.get("MEMORIZING", 0), state_seconds.get("LOOKING_AWAY", 0),
             state_seconds.get("PLAYING_WITH_PHONE", 0), state_seconds.get("TALKING", 0),
             state_seconds.get("AWAY_FROM_DESK", 0), state_seconds.get("SLEEPING_DROWSY", 0),
             state_seconds.get("UNKNOWN", 0), session_id)
        )
        for d in dialogue_log:
            conn.execute(
                "INSERT INTO dialogue_lines (session_id, timestamp, text, context) VALUES (?, ?, ?, ?)",
                (session_id, d["timestamp"], d["text"], d.get("context", ""))
            )


def save_session(user_id, child_name, started_at, ended_at, state_seconds, dialogue_log, report_text):
    """Legacy one-shot save (kept for smoke_test.py / any direct
    SessionManager use outside the live web flow). The live web flow uses
    start_session() + finish_session() instead so events can attach to a
    real session_id throughout."""
    session_id = start_session(user_id, child_name, started_at)
    finish_session(session_id, ended_at, state_seconds, dialogue_log, report_text)
    return session_id


def list_sessions(user_id, limit=20):
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY started_at DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]


def get_session(user_id, session_id):
    """Scoped by user_id so one family can never fetch another's session by
    guessing/incrementing an id."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id)
        ).fetchone()
        return dict(row) if row else None


# ---------- Session events (the per-tick metadata timeline) ----------
# NOTE ON SCOPING: these are looked up by session_id, and session_id itself
# is only ever obtained through get_session()/list_sessions() above (both
# user_id-scoped) - so as long as callers always get the session_id from a
# scoped lookup first, a family can't read another family's timeline. See
# server.py: every route that touches session_events first calls
# db.get_session(user_id, session_id) and 404s if that returns nothing.

def log_event(session_id, metadata, intervention_given=None, parent_feedback=None):
    with _connect() as conn:
        conn.execute(
            """INSERT INTO session_events
               (session_id, timestamp, head_direction, face_detected, person_detected,
                pose, hand_position, book_detected, phone_detected, activity,
                distraction_score, intervention_given, parent_feedback)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, metadata.get("timestamp", time.time()),
             metadata.get("head_direction"),
             _to_int_or_none(metadata.get("face_detected")),
             _to_int_or_none(metadata.get("person_detected")),
             metadata.get("pose"),
             metadata.get("hand_position"),
             _to_int_or_none(metadata.get("book_detected")),
             _to_int_or_none(metadata.get("phone_detected")),
             metadata.get("activity", "UNKNOWN"),
             metadata.get("distraction_score"),
             intervention_given,
             parent_feedback)
        )


def add_parent_feedback(session_id, feedback_text):
    """Appends a standalone feedback event to the timeline (e.g. the parent
    tapping 'distracted' - see the session-log format this mirrors)."""
    with _connect() as conn:
        conn.execute(
            """INSERT INTO session_events
               (session_id, timestamp, activity, parent_feedback)
               VALUES (?, ?, ?, ?)""",
            (session_id, time.time(), "PARENT_FEEDBACK", feedback_text)
        )


def get_timeline(session_id):
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM session_events WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def _to_int_or_none(value):
    if value is None:
        return None
    return int(bool(value))
