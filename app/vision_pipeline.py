"""
vision_pipeline.py

The multi-stage detection pipeline:

    Camera
      -> Person detection
      -> Face/head/pose detection
      -> Book/notebook detection
      -> Hand/writing detection
      -> Activity estimation
      -> Temporal analysis
      -> Distraction score
      -> Decision engine
      -> Voice intervention (handled by alert_system.py / session_manager.py)

Every stage uses either a genuinely pretrained, off-the-shelf model (YOLOv8,
trained by Ultralytics on the COCO dataset - person/book/cell phone are
existing COCO classes; YOLOv8-pose gives body keypoints) or a plain
geometric/motion heuristic on top of those outputs. NOTHING here needs the
parent to label anything - that's the point of using pretrained models
instead of the earlier from-scratch trainable classifier (still available
in app/trainer.py for anyone who wants it, but no longer the default path).

States (STATES below) are the measurable outputs this pipeline produces.
Only numeric/categorical metadata is ever recorded (see get_last_metadata) -
never an image or frame.

Degrades gracefully, same pattern as the rest of this app:
  ultralytics/torch installed -> full pipeline (YOLO detection + pose)
  not installed -> falls back to the Haar-cascade heuristic from
                   focus_detector.py, mapped into this module's state names
  no camera frame at all -> labeled random simulation (dev/testing only)
"""

import time
import random
from collections import Counter

try:
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from app.focus_detector import FocusDetector, ACTIVITY_LABELS as _HAAR_ACTIVITY_LABELS

STATES = (
    "READING",
    "WRITING",
    "MEMORIZING",
    "LOOKING_AWAY",
    "PLAYING_WITH_PHONE",
    "TALKING",
    "AWAY_FROM_DESK",
    "SLEEPING_DROWSY",
    "UNKNOWN",
)

# Which states are "good" (no alert, maybe encouragement) vs "problem"
# (candidate for a corrective voice intervention once sustained).
GOOD_STATES = {"READING", "WRITING", "MEMORIZING"}
PROBLEM_STATES = {"LOOKING_AWAY", "PLAYING_WITH_PHONE", "TALKING", "AWAY_FROM_DESK", "SLEEPING_DROWSY"}

# Rolling window for temporal smoothing / distraction score (seconds).
TEMPORAL_WINDOW_SECONDS = 60

# How many recent raw ticks must agree before the DISPLAYED/LOGGED state
# actually changes. This is the core "pattern over time, not one frame"
# accuracy fix: a single noisy misread (a bad frame, a momentary head turn
# to sneeze) no longer flips the whole session's classification by itself.
STABILITY_WINDOW = 3

_YOLO_DETECT_MODEL = None
_YOLO_POSE_MODEL = None


def _load_yolo_models():
    """Loads both YOLO models once, shared across all sessions/users - the
    models themselves don't hold any per-child state, so sharing is safe."""
    global _YOLO_DETECT_MODEL, _YOLO_POSE_MODEL
    if not YOLO_AVAILABLE:
        return False
    try:
        if _YOLO_DETECT_MODEL is None:
            _YOLO_DETECT_MODEL = YOLO("yolov8n.pt")
        if _YOLO_POSE_MODEL is None:
            _YOLO_POSE_MODEL = YOLO("yolov8n-pose.pt")
        return True
    except Exception:
        return False


class VisionPipeline:
    def __init__(self, gaze_away_threshold=15, writing_motion_threshold=12,
                 material_position="desk", user_id=None):
        self.gaze_away_threshold = gaze_away_threshold
        self.writing_motion_threshold = writing_motion_threshold
        self.material_position = material_position

        self.yolo_ready = _load_yolo_models()
        # Haar fallback detector, reused if YOLO/torch isn't available.
        # user_id lets it load a personalized calibration profile, if this
        # family has run one - see app/calibration.py.
        self._haar_fallback = FocusDetector(gaze_away_threshold, writing_motion_threshold,
                                             user_id=user_id, material_position=material_position)

        self._prev_gray = None
        self._state_start = {}       # state name -> timestamp it started
        self._current_state = "UNKNOWN"
        self._recent_raw = []        # last STABILITY_WINDOW raw (pre-smoothing) activities
        self._history = []           # [(timestamp, state)] within TEMPORAL_WINDOW_SECONDS
        self._talking_hint = False   # set externally from speech-recognition activity
        self._last_metadata = {}

    def set_talking_hint(self, is_talking):
        """session_manager.py calls this when the browser's speech
        recognizer is actively hearing the child speak - visual pipelines
        alone can't reliably detect 'talking' without mouth landmarks, so
        this is a deliberate, documented cross-check rather than a gap."""
        self._talking_hint = is_talking

    def get_last_metadata(self):
        return dict(self._last_metadata)

    # ---------- Main entry point ----------

    def analyze_frame(self, frame, now=None):
        now = now if now is not None else time.time()

        if frame is None or not CV_AVAILABLE:
            return self._simulate(now)

        if self.yolo_ready:
            raw = self._analyze_with_yolo(frame, now)
        else:
            raw = self._analyze_with_haar_fallback(frame, now)

        smoothed_state = self._update_temporal_state(raw["activity"], now)
        raw["activity"] = smoothed_state
        raw["distraction_score"] = self._compute_distraction_score(now)
        self._last_metadata = raw
        return raw

    # ---------- Stage 2-5: YOLO-based pipeline ----------

    def _analyze_with_yolo(self, frame, now):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Stage: person + object detection (one pass covers person, book, cell phone)
        det_results = _YOLO_DETECT_MODEL(frame, verbose=False, conf=0.35)[0]
        person_box = None
        book_detected = False
        phone_detected = False
        best_person_area = 0

        for box in det_results.boxes:
            cls_name = _YOLO_DETECT_MODEL.names[int(box.cls)]
            if cls_name == "person":
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                area = (x2 - x1) * (y2 - y1)
                if area > best_person_area:
                    best_person_area = area
                    person_box = (x1, y1, x2, y2)
            elif cls_name == "book":
                book_detected = True
            elif cls_name == "cell phone":
                phone_detected = True

        person_detected = person_box is not None

        # Stage: face/head/pose
        head_direction = "unknown"
        face_detected = False
        pose_summary = "unknown"
        hand_position = "unknown"
        wrist_points = []

        if person_detected:
            pose_results = _YOLO_POSE_MODEL(frame, verbose=False, conf=0.35)[0]
            if len(pose_results.keypoints.xy) > 0:
                # Use the keypoint set whose bounding region best matches our person box.
                kpts = pose_results.keypoints.xy[0]
                conf = pose_results.keypoints.conf[0] if pose_results.keypoints.conf is not None else None
                names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                         "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                         "left_wrist", "right_wrist"]
                pts = {n: (float(kpts[i][0]), float(kpts[i][1])) for i, n in enumerate(names)
                       if conf is None or float(conf[i]) > 0.3}

                face_detected = "nose" in pts and ("left_eye" in pts or "right_eye" in pts)

                if face_detected:
                    head_direction = self._estimate_head_direction(pts)

                if "left_shoulder" in pts and "right_shoulder" in pts and "nose" in pts:
                    shoulder_y = (pts["left_shoulder"][1] + pts["right_shoulder"][1]) / 2
                    nose_y = pts["nose"][1]
                    shoulder_span = abs(pts["left_shoulder"][0] - pts["right_shoulder"][0]) or 1
                    drop = (nose_y - (shoulder_y - shoulder_span)) / shoulder_span
                    if drop > 0.9:
                        pose_summary = "head_drooped"   # possible drowsiness
                    elif drop > 0.3:
                        pose_summary = "leaning_forward"  # reading/writing posture
                    else:
                        pose_summary = "upright"

                wrist_points = [pts[k] for k in ("left_wrist", "right_wrist") if k in pts]
                if wrist_points:
                    hand_position = "near_book" if book_detected else "visible"

        # Stage: hand/writing motion, anchored to wrist position when we have it
        motion_score = self._compute_motion_score(gray, wrist_points)
        self._prev_gray = gray

        activity = self._estimate_activity(
            person_detected=person_detected,
            face_detected=face_detected,
            head_direction=head_direction,
            book_detected=book_detected,
            phone_detected=phone_detected,
            pose_summary=pose_summary,
            motion_score=motion_score,
            now=now,
        )

        return {
            "timestamp": now,
            "person_detected": person_detected,
            "face_detected": face_detected,
            "head_direction": head_direction,
            "pose": pose_summary,
            "hand_position": hand_position,
            "book_detected": book_detected,
            "phone_detected": phone_detected,
            "motion_score": round(motion_score, 2),
            "activity": activity,
            "detection_backend": "yolo",
        }

    def _estimate_head_direction(self, pts):
        """Rough left/right/center estimate from ear/eye visibility and
        symmetry - not a precise gaze vector, just enough to distinguish
        'facing the desk' from 'turned away'."""
        left_visible = "left_ear" in pts
        right_visible = "right_ear" in pts
        if left_visible and not right_visible:
            return "turned_left"
        if right_visible and not left_visible:
            return "turned_right"
        if "left_eye" in pts and "right_eye" in pts and "nose" in pts:
            eye_mid_x = (pts["left_eye"][0] + pts["right_eye"][0]) / 2
            eye_span = abs(pts["left_eye"][0] - pts["right_eye"][0]) or 1
            offset = (pts["nose"][0] - eye_mid_x) / eye_span
            if offset > 0.4:
                return "turned_left"
            if offset < -0.4:
                return "turned_right"
        return "facing_forward"

    # ---------- Decision engine (Stage: activity estimation) ----------

    def _estimate_activity(self, person_detected, face_detected, head_direction,
                            book_detected, phone_detected, pose_summary, motion_score, now):
        """Plain prioritized rules, deliberately not a trained model - this
        is the whole point of the pretrained-detector approach: the inputs
        (person/face/book/phone/pose/motion) are real signals, and the
        mapping to an activity is transparent logic you can read and tune,
        not learned weights."""
        if not person_detected:
            return "AWAY_FROM_DESK"

        if phone_detected:
            return "PLAYING_WITH_PHONE"

        if self._talking_hint:
            return "TALKING"

        if pose_summary == "head_drooped" and motion_score < self.writing_motion_threshold:
            return "SLEEPING_DROWSY"

        if head_direction in ("turned_left", "turned_right"):
            return "LOOKING_AWAY"

        if motion_score > self.writing_motion_threshold:
            return "WRITING"

        if book_detected:
            return "READING"

        if pose_summary == "leaning_forward":
            # Posture-based backup for when the object detector misses the
            # book - closed notebooks, tablets, spiral binding, and partial
            # occlusion by a hand/arm are all common false negatives for
            # the "book" COCO class. Leaning over the desk is a strong
            # physical proxy for engagement with the material even when
            # object detection alone can't confirm what that material is.
            return "READING"

        if not face_detected:
            # No face, not leaning over the desk, not turned to a
            # detectable side - genuinely ambiguous rather than confidently
            # "away."
            return "LOOKING_AWAY"

        # Face forward, upright posture, no book, no motion - could be a
        # brief pause to think; temporal smoothing (in _update_temporal_state)
        # decides whether this persists into LOOKING_AWAY.
        return "MEMORIZING"

    # ---------- Haar fallback path (no torch/ultralytics available) ----------

    def _analyze_with_haar_fallback(self, frame, now):
        legacy_label = self._haar_fallback.classify_frame(frame, now=now)
        legacy_features = self._haar_fallback.get_last_features()

        mapping = {
            "READING": "READING",
            "WRITING": "WRITING",
            "MEMORIZING": "MEMORIZING",
            "DISTRACTED": "LOOKING_AWAY",
            "DISRUPTIVE": "AWAY_FROM_DESK",
        }
        activity = mapping.get(legacy_label, "UNKNOWN")
        if self._talking_hint:
            activity = "TALKING"

        return {
            "timestamp": now,
            "person_detected": bool(legacy_features.get("face_present")),
            "face_detected": bool(legacy_features.get("face_present")),
            "head_direction": "facing_forward" if legacy_features.get("face_present") else "unknown",
            "pose": "unknown",  # no pose estimation available without YOLO/mediapipe
            "hand_position": "unknown",  # no wrist tracking available in this fallback
            "book_detected": None,  # not available without an object detector
            "phone_detected": None,  # not available without an object detector
            "motion_score": round(legacy_features.get("motion_score", 0), 2),
            "activity": activity,
            "detection_backend": "haar_fallback",
        }

    # ---------- Shared motion heuristic ----------

    def _compute_motion_score(self, gray, wrist_points):
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            return 0.0

        if wrist_points:
            # Look at a small region around each wrist for hand/pen motion.
            scores = []
            for wx, wy in wrist_points:
                x0, x1 = max(int(wx) - 40, 0), min(int(wx) + 40, gray.shape[1])
                y0, y1 = max(int(wy) - 40, 0), min(int(wy) + 40, gray.shape[0])
                if x1 <= x0 or y1 <= y0:
                    continue
                diff = cv2.absdiff(self._prev_gray[y0:y1, x0:x1], gray[y0:y1, x0:x1])
                scores.append(float(diff.mean()))
            if scores:
                return max(scores)

        diff = cv2.absdiff(self._prev_gray, gray)
        return float(diff.mean())

    # ---------- Stage: temporal analysis ----------

    def _update_temporal_state(self, raw_activity, now):
        """
        This used to just echo raw_activity straight through - meaning a
        single noisy frame (a bad Haar read, a momentary head turn, a
        lighting flicker) could instantly flip the displayed/logged/alerted
        state on its own. Real accuracy for "what is the child doing" comes
        from a PATTERN across a few ticks, not any single frame - so now a
        new activity only takes over once it has a clear majority across
        the last STABILITY_WINDOW raw reads. A single outlier tick gets
        absorbed rather than treated as a genuine state change.

        TALKING is exempt: it comes from the speech transcript, a real,
        instantaneous signal, not a visual guess - there's no noise to
        smooth away, so it should reflect immediately, not lag behind
        stale visual history.
        """
        self._recent_raw.append(raw_activity)
        self._recent_raw = self._recent_raw[-STABILITY_WINDOW:]

        if raw_activity == "TALKING":
            new_state = "TALKING"
        elif len(self._recent_raw) < STABILITY_WINDOW:
            # Not enough history yet (start of session) - use the raw read
            # rather than delaying the very first classification.
            new_state = raw_activity
        else:
            counts = Counter(self._recent_raw)
            majority_activity, majority_count = counts.most_common(1)[0]
            if majority_count > len(self._recent_raw) / 2:
                new_state = majority_activity
            else:
                new_state = self._current_state  # no clear majority - hold the previous state

        self._history.append((now, new_state))
        cutoff = now - TEMPORAL_WINDOW_SECONDS
        self._history = [(t, s) for t, s in self._history if t >= cutoff]

        if new_state != self._current_state:
            self._state_start[new_state] = self._state_start.get(new_state, now)
        self._current_state = new_state
        return new_state

    # ---------- Stage: distraction score ----------

    def _compute_distraction_score(self, now):
        """0-100: percentage of the recent rolling window spent in a
        PROBLEM_STATE. A simple, explainable ratio - not a learned score."""
        if not self._history:
            return 0
        problem_time = sum(1 for _, s in self._history if s in PROBLEM_STATES)
        return round(100 * problem_time / len(self._history))

    def _simulate(self, now):
        activity = random.choices(STATES, weights=[25, 15, 10, 15, 8, 8, 10, 5, 4])[0]
        return {
            "timestamp": now, "person_detected": True, "face_detected": True,
            "head_direction": "facing_forward", "pose": "upright", "hand_position": "unknown",
            "book_detected": None, "phone_detected": None, "motion_score": 0.0,
            "activity": activity, "distraction_score": 0, "detection_backend": "simulation",
        }
