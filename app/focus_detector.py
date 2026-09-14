"""
focus_detector.py

The Haar-cascade fallback detector - used directly by vision_pipeline.py
when YOLO/torch isn't available (see that module's docstring for the full
pipeline). This file stays a simple 5-state classifier
(READING/WRITING/MEMORIZING/DISTRACTED/DISRUPTIVE); vision_pipeline.py maps
these into the richer 9-state taxonomy at its fallback boundary, so this
file doesn't need to know about that taxonomy at all.

Also still the home of the numeric feature vector (get_last_features) that
app/trainer.py trains on, and still loads a per-user trained model if one
exists, exactly as before - the optional from-scratch trainable path this
app has always supported alongside pretrained detection.

No raw images, facial vectors, or biometric templates are ever stored -
only these numbers, computed per-frame and discarded.
"""

import time
import random

try:
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

HAAR_AVAILABLE = False
if CV_AVAILABLE:
    try:
        _FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"
        HAAR_AVAILABLE = True
    except Exception:
        HAAR_AVAILABLE = False

from app.trainer import FEATURE_NAMES, load_model as load_trained_model

ACTIVITY_LABELS = ("READING", "WRITING", "MEMORIZING")
PROBLEM_LABELS = ("DISTRACTED", "DISRUPTIVE")
STATES = ACTIVITY_LABELS + PROBLEM_LABELS


class FocusDetector:
    def __init__(self, gaze_away_threshold=15, writing_motion_threshold=12,
                 user_id=None, material_position="desk"):
        """
        material_position: "desk" (default) or "screen".

        This exists because "eyes visible to the camera" means opposite
        things depending on where the study material actually is:
          - "desk": a physical book/notebook sits below the camera. Looking
            DOWN at it is the normal, engaged reading pose - and that's
            exactly when Haar's eye cascade typically fails (eyes
            foreshortened/hidden by eyelids from the camera's angle), so
            "eyes not detected" is the EXPECTED good signal here, not a
            distraction signal. Looking UP toward the camera instead means
            looking away from the material.
          - "screen": the material is on the same screen the camera sits
            on (e.g. reading a PDF or typing on the laptop itself), so
            looking at the material and looking toward the camera are the
            same thing - eyes visible really does mean engaged here.
        Get this backwards for your actual setup and reading/distraction
        will look inverted - see README.
        """
        self.gaze_away_threshold = gaze_away_threshold
        self.writing_motion_threshold = writing_motion_threshold
        self.material_position = material_position if material_position in ("desk", "screen") else "desk"

        # Personalized calibration overrides the generic threshold above,
        # if this user has run the calibration flow - see app/calibration.py.
        if user_id is not None:
            try:
                from app.calibration import load_profile
                profile = load_profile(user_id)
                if profile and "writing_motion_threshold" in profile:
                    self.writing_motion_threshold = profile["writing_motion_threshold"]
            except Exception:
                pass

        # Rolling motion-score smoothing: a single noisy frame (someone
        # briefly adjusting posture, a shadow, a camera auto-exposure
        # flicker) used to be able to instantly trigger WRITING on its own.
        # Averaging over the last few frames means a real accuracy signal
        # (sustained motion) is needed, not a one-frame spike.
        self._motion_history = []
        self.MOTION_SMOOTHING_WINDOW = 3

        self._eyes_hidden_start = None   # duration eyes have been NOT detected
        self._eyes_visible_start = None  # duration eyes HAVE been detected
        self._face_missing_start = None
        self._prev_gray = None
        self._last_features = {name: 0 for name in FEATURE_NAMES}

        self.mp_face_mesh = None
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1, refine_landmarks=True,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5
                )
            except Exception:
                self.mp_face_mesh = None

        self.face_cascade = None
        self.eye_cascade = None
        if HAAR_AVAILABLE:
            try:
                self.face_cascade = cv2.CascadeClassifier(_FACE_CASCADE_PATH)
                self.eye_cascade = cv2.CascadeClassifier(_EYE_CASCADE_PATH)
                if self.face_cascade.empty() or self.eye_cascade.empty():
                    self.face_cascade = None
                    self.eye_cascade = None
            except Exception:
                self.face_cascade = None
                self.eye_cascade = None

        self.trained_model = load_trained_model(user_id) if user_id is not None else None

    def get_last_features(self):
        return dict(self._last_features)

    def classify_frame(self, frame, now=None):
        if frame is None or not CV_AVAILABLE:
            return self._simulate_state()

        if self.face_cascade is not None:
            return self._classify_with_haar(frame, now=now)

        if self.mp_face_mesh is not None:
            return self._classify_with_mediapipe(frame, now=now)

        return self._simulate_state()

    def _compute_motion_score(self, gray, face_box):
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            return 0.0
        if face_box is None:
            diff = cv2.absdiff(self._prev_gray, gray)
            return float(diff.mean())
        x, y, w, h = face_box
        y0 = min(y + h, gray.shape[0] - 1)
        y1 = gray.shape[0]
        x0 = max(x - w // 2, 0)
        x1 = min(x + w + w // 2, gray.shape[1])
        if y0 >= y1 or x0 >= x1:
            return 0.0
        region_prev = self._prev_gray[y0:y1, x0:x1]
        region_curr = gray[y0:y1, x0:x1]
        if region_prev.size == 0 or region_curr.size == 0:
            return 0.0
        diff = cv2.absdiff(region_prev, region_curr)
        return float(diff.mean())

    def _classify_with_haar(self, frame, now=None):
        now = now if now is not None else time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape[:2]
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            motion_score = self._compute_motion_score(gray, None)
            self._prev_gray = gray
            face_present, face_w_ratio, face_h_ratio = 0, 0.0, 0.0
            face_x_ratio, face_y_ratio, num_eyes = 0.0, 0.0, 0
        else:
            self._face_missing_start = None
            face_box = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face_box
            motion_score = self._compute_motion_score(gray, face_box)
            self._prev_gray = gray

            face_roi = gray[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(
                face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15)
            )

            face_present = 1
            face_w_ratio = w / fw
            face_h_ratio = h / fh
            face_x_ratio = (x + w / 2) / fw
            face_y_ratio = (y + h / 2) / fh
            num_eyes = len(eyes)

        if face_present == 0:
            if self._face_missing_start is None:
                self._face_missing_start = now
            face_missing_duration = now - self._face_missing_start
        else:
            face_missing_duration = 0.0

        # Track BOTH durations - which one matters for the decision depends
        # on material_position (see __init__ docstring).
        if face_present == 1 and num_eyes == 0:
            if self._eyes_hidden_start is None:
                self._eyes_hidden_start = now
            eyes_hidden_duration = now - self._eyes_hidden_start
            self._eyes_visible_start = None
            eyes_visible_duration = 0.0
        elif face_present == 1:
            if self._eyes_visible_start is None:
                self._eyes_visible_start = now
            eyes_visible_duration = now - self._eyes_visible_start
            self._eyes_hidden_start = None
            eyes_hidden_duration = 0.0
        else:
            eyes_hidden_duration = 0.0
            eyes_visible_duration = 0.0
            self._eyes_hidden_start = None
            self._eyes_visible_start = None

        self._last_features = {
            "face_present": face_present,
            "face_w_ratio": face_w_ratio,
            "face_h_ratio": face_h_ratio,
            "face_x_ratio": face_x_ratio,
            "face_y_ratio": face_y_ratio,
            "num_eyes": num_eyes,
            "motion_score": motion_score,
            "gaze_away_duration": eyes_hidden_duration if self.material_position == "screen" else eyes_visible_duration,
            "face_missing_duration": face_missing_duration,
        }

        # Smooth motion over the last few frames before using it for a
        # decision - a single spike (posture shift, shadow, auto-exposure
        # flicker) shouldn't be enough on its own to trigger WRITING.
        self._motion_history.append(motion_score)
        self._motion_history = self._motion_history[-self.MOTION_SMOOTHING_WINDOW:]
        smoothed_motion_score = sum(self._motion_history) / len(self._motion_history)

        if self.trained_model is not None:
            vector = [[self._last_features[name] for name in FEATURE_NAMES]]
            try:
                return self.trained_model.predict(vector)[0]
            except Exception:
                pass

        if face_present == 0:
            return "DISRUPTIVE" if face_missing_duration > self.gaze_away_threshold else "DISTRACTED"

        if smoothed_motion_score > self.writing_motion_threshold:
            return "WRITING"

        if self.material_position == "desk":
            # Material is below the camera - looking DOWN at it (eyes not
            # detected) is the normal engaged pose. Looking UP toward the
            # camera instead means looking away from the material.
            if num_eyes == 0:
                return "READING"
            return "DISTRACTED" if eyes_visible_duration > self.gaze_away_threshold else "MEMORIZING"
        else:
            # Material is on-screen, co-located with the camera - looking
            # toward the camera IS looking at the material.
            if num_eyes == 0:
                return "DISTRACTED" if eyes_hidden_duration > self.gaze_away_threshold else "MEMORIZING"
            return "READING"

    def _estimate_gaze_direction(self, landmarks, frame_shape):
        """Placeholder - see README. Currently always reports on-material."""
        return "on_material"

    def _classify_with_mediapipe(self, frame, now=None):
        now = now if now is not None else time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._prev_gray = gray
            return self._handle_face_missing(now)

        self._face_missing_start = None
        landmarks = results.multi_face_landmarks[0]
        xs = [lm.x * frame.shape[1] for lm in landmarks.landmark]
        ys = [lm.y * frame.shape[0] for lm in landmarks.landmark]
        face_box = (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))
        motion_score = self._compute_motion_score(gray, face_box)
        self._prev_gray = gray

        gaze = self._estimate_gaze_direction(landmarks, frame.shape)
        if gaze == "away":
            return self._handle_gaze_away(now)

        self._eyes_hidden_start = None
        return "WRITING" if motion_score > self.writing_motion_threshold else "READING"

    def _handle_face_missing(self, now):
        if self._face_missing_start is None:
            self._face_missing_start = now
        if now - self._face_missing_start > self.gaze_away_threshold:
            return "DISRUPTIVE"
        return "DISTRACTED"

    def _handle_gaze_away(self, now):
        if self._eyes_hidden_start is None:
            self._eyes_hidden_start = now
        if now - self._eyes_hidden_start > self.gaze_away_threshold:
            return "DISTRACTED"
        return "MEMORIZING"

    def _simulate_state(self):
        return random.choices(STATES, weights=[0.45, 0.2, 0.15, 0.12, 0.08])[0]
