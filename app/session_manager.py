"""
session_manager.py

Implements the IDLE -> ACTIVE -> REPORTING state machine, runs each frame
through VisionPipeline (the full Camera -> ... -> Voice intervention
pipeline - see app/vision_pipeline.py), decides interventions (corrective +
encouragement), logs child dialogue, and persists the per-tick metadata
timeline to the database AS IT HAPPENS (db.log_event) rather than only at
the end - so nothing is lost if the process crashes mid-session.
"""

import time
from app import config, db
from app.vision_pipeline import VisionPipeline, GOOD_STATES, PROBLEM_STATES
from app.alert_system import AlertSystem
from app.dialogue_listener import DialogueListener
from app.report_generator import generate_report

ALL_STATES = ("READING", "WRITING", "MEMORIZING", "LOOKING_AWAY",
              "PLAYING_WITH_PHONE", "TALKING", "AWAY_FROM_DESK",
              "SLEEPING_DROWSY", "UNKNOWN")


class SessionManager:
    def __init__(self, user_id, child_name=None, child_age=None, task=None,
                 language=None, material_position="desk"):
        self.user_id = user_id
        self.child_name = child_name or config.CHILD_NAME
        self.child_age = child_age
        self.task = task
        self.language = language or config.LANGUAGE

        self.pipeline = VisionPipeline(
            gaze_away_threshold=config.DISTRACTED_GAZE_AWAY_THRESHOLD,
            writing_motion_threshold=config.WRITING_MOTION_THRESHOLD,
            material_position=material_position,
            user_id=user_id,
        )
        self.alerts = AlertSystem(
            user_id, self.child_name, self.language,
            corrective_cooldown_seconds=config.ALERT_COOLDOWN_SECONDS,
            encouragement_cooldown_seconds=config.POSITIVE_ALERT_COOLDOWN_SECONDS,
        )
        self.listener = DialogueListener(self.language)

        self.state = "IDLE"
        self.db_session_id = None
        self.state_seconds = {s: 0 for s in ALL_STATES}
        self.dialogue_log = []

        self._bad_decision_start = None
        self._last_focused_activity = None
        self._session_start = None
        self._session_end = None
        self._last_tick_time = None

    def start(self):
        self.state = "ACTIVE"
        self._session_start = time.time()
        self._last_tick_time = self._session_start
        self.state_seconds = {k: 0 for k in ALL_STATES}
        self.dialogue_log = []
        self._bad_decision_start = None
        self._last_focused_activity = None
        self.db_session_id = db.start_session(self.user_id, self.child_name, self._session_start)

    def stop(self):
        self._session_end = time.time()
        self.state = "REPORTING"

        timeline = db.get_timeline(self.db_session_id)
        report = generate_report(
            state_seconds=self.state_seconds,
            dialogue_log=self.dialogue_log,
            timeline=timeline,
            session_start=self._session_start,
            session_end=self._session_end,
            child_name=self.child_name,
            child_age=self.child_age,
            task=self.task,
            session_number=self.db_session_id,
        )
        db.finish_session(self.db_session_id, self._session_end, self.state_seconds,
                           self.dialogue_log, report)
        return report

    def record_parent_label(self, label):
        """Called when the parent clicks a live label button - the
        'parent_feedback' field from the requested schema, logged as its
        own timeline event."""
        if self.db_session_id is not None:
            db.add_parent_feedback(self.db_session_id, label)

    def tick(self, frame=None, transcript_text=None):
        if self.state != "ACTIVE":
            return None

        now = time.time()
        elapsed = now - (self._last_tick_time or now)
        elapsed = max(elapsed, 1)
        self._last_tick_time = now

        # TALKING is decided from the transcript - a real signal, not a
        # video guess. If the child said something this tick, the pipeline
        # treats that as talking regardless of what the camera shows.
        speaking = bool(transcript_text and transcript_text.strip())
        self.pipeline.set_talking_hint(speaking)

        result = self.pipeline.analyze_frame(frame, now=now)
        activity = result["activity"]
        self.state_seconds[activity] = self.state_seconds.get(activity, 0) + elapsed

        if activity in GOOD_STATES:
            decision = "FOCUSED"
        elif activity in PROBLEM_STATES:
            decision = "DISTRACTED"
        else:
            decision = "UNKNOWN"

        alert_payload = None
        alert_kind = None

        if decision == "DISTRACTED":
            self._last_focused_activity = None
            if self._bad_decision_start is None:
                self._bad_decision_start = now
            elif now - self._bad_decision_start >= config.INTERVENTION_TRIGGER_THRESHOLD:
                alert_payload = self.alerts.trigger_alert(activity)
                if alert_payload:
                    alert_kind = "corrective"
        else:
            self._bad_decision_start = None
            if activity in GOOD_STATES and activity != self._last_focused_activity:
                alert_payload = self.alerts.trigger_encouragement(activity)
                if alert_payload:
                    alert_kind = "encouragement"
            self._last_focused_activity = activity if activity in GOOD_STATES else None

        transcript = self.listener.package_transcript(transcript_text)
        if transcript:
            ts, text = transcript
            context = "after a reminder" if alert_kind else "during study"
            self.dialogue_log.append({"timestamp": ts, "text": text, "context": context})

        db.log_event(self.db_session_id, result, intervention_given=alert_kind)

        return {
            "state": decision,
            "activity": activity,
            "distraction_score": result.get("distraction_score", 0),
            "detection_backend": result.get("detection_backend"),
            "alert_audio_url": (alert_payload or {}).get("audio_url"),
            "alert_text": (alert_payload or {}).get("text"),
            "alert_kind": alert_kind,
            "state_seconds": {k: round(v, 1) for k, v in self.state_seconds.items()},
        }
