"""
server.py

Multi-family web server for FocusBuddy. Each family creates an account;
everything from here on is scoped to that account (their sessions, their
recorded audio clips, their trained model, their event timeline). See
README's "Deploying for multiple families" section for what this does and
does NOT cover before putting this in front of real other families.

Per-session behavior (the Camera -> ... -> Voice intervention pipeline,
alerts, transcription) lives in session_manager.py / vision_pipeline.py.
"""

import base64
import os
import time
import uuid
import secrets
import re
import tempfile
import numpy as np
import cv2
import requests
from functools import wraps
from urllib.parse import urlparse
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from werkzeug.utils import secure_filename

from app import config, db, trainer, video_ingest, calibration
from app.session_manager import SessionManager

app = Flask(__name__)

# SECRET_KEY signs the login session cookie. Set FOCUSBUDDY_SECRET_KEY in
# your deployment environment so sessions survive a server restart.
app.secret_key = os.environ.get("FOCUSBUDDY_SECRET_KEY") or secrets.token_hex(32)

db.init_db()

# One active SessionManager per logged-in user, keyed by user_id.
_sessions = {}

# In-progress calibration runs (see app/calibration.py), keyed by user_id.
_calibration_sessions = {}
_calibration_detectors = {}  # reused across samples so motion_score has frame-to-frame continuity

ALLOWED_SITUATIONS = {
    "looking_away", "playing_with_phone", "away_from_desk", "sleeping_drowsy", "talking",
    "reading", "writing", "memorizing",
}
AUDIO_ROOT = os.path.join(app.root_path, "static", "audio")

MAX_TRAINING_VIDEO_BYTES = 80 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
ALLOWED_DRIVE_HOSTS = {"drive.google.com", "docs.google.com"}


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"ok": False, "error": "Not logged in"}), 401
            return redirect(url_for("login_page"))
        return view(*args, **kwargs)
    return wrapped


def _current_user_id():
    return session.get("user_id")


def _user_audio_dir(user_id, situation):
    path = os.path.join(AUDIO_ROOT, str(user_id), situation)
    os.makedirs(path, exist_ok=True)
    return path


def _decode_frame(data_url):
    if not data_url or "," not in data_url:
        return None
    try:
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ---------- Auth pages ----------

@app.route("/register", methods=["GET", "POST"])
def register_page():
    if request.method == "GET":
        return render_template("register.html")
    payload = request.get_json(force=True) or {}
    result = db.create_user(payload.get("username", ""), payload.get("password", ""))
    if not result["ok"]:
        return jsonify(result), 400
    session.clear()
    session["user_id"] = result["user_id"]
    session["username"] = result["username"]
    return jsonify({"ok": True})


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "GET":
        return render_template("login.html")
    payload = request.get_json(force=True) or {}
    user = db.verify_user(payload.get("username", ""), payload.get("password", ""))
    if not user:
        return jsonify({"ok": False, "error": "Invalid username or password."}), 401
    session.clear()
    session["user_id"] = user["id"]
    session["username"] = user["username"]
    return jsonify({"ok": True})


@app.route("/logout", methods=["POST"])
def logout():
    user_id = session.get("user_id")
    if user_id in _sessions:
        del _sessions[user_id]
    session.clear()
    return jsonify({"ok": True})


# ---------- Dashboard ----------

@app.route("/")
@login_required
def index():
    return render_template("index.html", username=session.get("username"))


# ---------- Session API ----------

@app.route("/api/start", methods=["POST"])
@login_required
def start_session():
    user_id = _current_user_id()
    payload = request.get_json(force=True) or {}
    child_name = payload.get("child_name", config.CHILD_NAME)
    child_age = payload.get("child_age")
    task = payload.get("task")
    language = payload.get("language", config.LANGUAGE)
    material_position = payload.get("material_position", "desk")

    _sessions[user_id] = SessionManager(
        user_id=user_id, child_name=child_name, child_age=child_age,
        task=task, language=language, material_position=material_position,
    )
    _sessions[user_id].start()
    return jsonify({"ok": True, "state": _sessions[user_id].state})


@app.route("/api/tick", methods=["POST"])
@login_required
def tick():
    user_id = _current_user_id()
    active = _sessions.get(user_id)
    if active is None or active.state != "ACTIVE":
        return jsonify({"ok": False, "error": "No active session"}), 400

    payload = request.get_json(force=True) or {}
    frame = _decode_frame(payload.get("frame"))
    transcript_text = payload.get("transcript")

    result = active.tick(frame=frame, transcript_text=transcript_text)
    return jsonify({"ok": True, **result})


@app.route("/api/stop", methods=["POST"])
@login_required
def stop_session():
    user_id = _current_user_id()
    active = _sessions.get(user_id)
    if active is None:
        return jsonify({"ok": False, "error": "No session in progress"}), 400

    report_text = active.stop()
    session_id = active.db_session_id
    del _sessions[user_id]
    return jsonify({"ok": True, "session_id": session_id, "report": report_text})


@app.route("/api/train/label", methods=["POST"])
@login_required
def train_label_live():
    """Live 'parent labels: X' feedback during an active session - logged
    to that session's timeline (app/db.py: add_parent_feedback), separate
    from the optional from-scratch trainer's labeled examples below."""
    user_id = _current_user_id()
    active = _sessions.get(user_id)
    if active is None or active.state != "ACTIVE":
        return jsonify({"ok": False, "error": "No active session"}), 400

    payload = request.get_json(force=True) or {}
    label = payload.get("label")
    if label not in trainer.LABELS:
        return jsonify({"ok": False, "error": f"invalid label: {label}"}), 400

    active.record_parent_label(label)

    # Also feed the optional trainable model, if the parent wants to build one.
    features = active.pipeline._haar_fallback.get_last_features()
    trainer.append_example(user_id, features, label)

    return jsonify({"ok": True, "counts": trainer.label_counts(user_id)})


@app.route("/api/history")
@login_required
def history():
    return jsonify(db.list_sessions(_current_user_id()))


@app.route("/api/history/<int:session_id>")
@login_required
def history_detail(session_id):
    row = db.get_session(_current_user_id(), session_id)
    if not row:
        return jsonify({"ok": False, "error": "Not found"}), 404
    return jsonify(row)


@app.route("/api/history/<int:session_id>/timeline")
@login_required
def history_timeline(session_id):
    """Scoped safely: get_session() is user_id-checked first, and the
    timeline is only ever fetched using a session_id that check returned -
    a family can never read another family's timeline this way."""
    row = db.get_session(_current_user_id(), session_id)
    if not row:
        return jsonify({"ok": False, "error": "Not found"}), 404
    return jsonify(db.get_timeline(session_id))


# ---------- Audio clips (scoped per user) ----------

@app.route("/api/upload-clip", methods=["POST"])
@login_required
def upload_clip():
    user_id = _current_user_id()
    situation = request.form.get("situation")
    if situation not in ALLOWED_SITUATIONS:
        return jsonify({"ok": False, "error": "invalid situation"}), 400

    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"ok": False, "error": "no audio file provided"}), 400

    ext = os.path.splitext(audio_file.filename or "")[1] or ".webm"
    filename = secure_filename(f"parent_{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}")
    dest_dir = _user_audio_dir(user_id, situation)
    audio_file.save(os.path.join(dest_dir, filename))

    return jsonify({
        "ok": True, "filename": filename,
        "url": f"/static/audio/{user_id}/{situation}/{filename}",
    })


@app.route("/api/clips")
@login_required
def list_clips():
    user_id = _current_user_id()
    result = {}
    for situation in ALLOWED_SITUATIONS:
        dir_path = _user_audio_dir(user_id, situation)
        result[situation] = sorted(f for f in os.listdir(dir_path) if not f.startswith("."))
    return jsonify(result)


@app.route("/api/clips/<situation>/<filename>", methods=["DELETE"])
@login_required
def delete_clip(situation, filename):
    user_id = _current_user_id()
    if situation not in ALLOWED_SITUATIONS:
        return jsonify({"ok": False, "error": "invalid situation"}), 400
    path = os.path.join(_user_audio_dir(user_id, situation), secure_filename(filename))
    if os.path.exists(path):
        os.remove(path)
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "not found"}), 404


# ---------- Optional from-scratch trainer (numeric-features model) ----------

@app.route("/api/train/status")
@login_required
def train_status():
    user_id = _current_user_id()
    return jsonify({
        "ok": True,
        "counts": trainer.label_counts(user_id),
        "model_exists": trainer.model_exists(user_id),
        "sklearn_available": trainer.SKLEARN_AVAILABLE,
    })


@app.route("/api/train/fit", methods=["POST"])
@login_required
def train_fit():
    return jsonify(trainer.train_and_save(_current_user_id()))


@app.route("/api/train/reset", methods=["POST"])
@login_required
def train_reset():
    trainer.reset_training_data(_current_user_id())
    return jsonify({"ok": True})


# ---------- Calibration: personalize thresholds to this family's setup ----------

@app.route("/api/calibrate/start", methods=["POST"])
@login_required
def calibrate_start():
    from app.focus_detector import FocusDetector
    user_id = _current_user_id()
    _calibration_sessions[user_id] = calibration.CalibrationSession()
    _calibration_detectors[user_id] = FocusDetector()
    return jsonify({"ok": True})


@app.route("/api/calibrate/sample", methods=["POST"])
@login_required
def calibrate_sample():
    user_id = _current_user_id()
    cal = _calibration_sessions.get(user_id)
    detector = _calibration_detectors.get(user_id)
    if cal is None or detector is None:
        return jsonify({"ok": False, "error": "Calibration not started"}), 400

    payload = request.get_json(force=True) or {}
    phase = payload.get("phase")
    frame = _decode_frame(payload.get("frame"))
    if frame is None:
        return jsonify({"ok": False, "error": "no frame provided"}), 400

    try:
        detector.classify_frame(frame)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    features = detector.get_last_features()
    try:
        cal.add_sample(phase, features["motion_score"], features["face_present"], features["num_eyes"])
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    return jsonify({"ok": True, "counts": cal.counts()})


@app.route("/api/calibrate/finish", methods=["POST"])
@login_required
def calibrate_finish():
    user_id = _current_user_id()
    cal = _calibration_sessions.get(user_id)
    if cal is None:
        return jsonify({"ok": False, "error": "Calibration not started"}), 400

    result = cal.finish(user_id)
    if result["ok"]:
        del _calibration_sessions[user_id]
        del _calibration_detectors[user_id]
    return jsonify(result)


@app.route("/api/calibrate/status")
@login_required
def calibrate_status():
    user_id = _current_user_id()
    profile = calibration.load_profile(user_id)
    cal = _calibration_sessions.get(user_id)
    return jsonify({
        "ok": True,
        "profile": profile,
        "in_progress_counts": cal.counts() if cal else None,
    })


@app.route("/api/calibrate/reset", methods=["POST"])
@login_required
def calibrate_reset():
    user_id = _current_user_id()
    calibration.reset_profile(user_id)
    _calibration_sessions.pop(user_id, None)
    _calibration_detectors.pop(user_id, None)
    return jsonify({"ok": True})


def _extract_drive_file_id(url):
    match = re.search(r"/d/([a-zA-Z0-9_-]{10,})", url)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]{10,})", url)
    if match:
        return match.group(1)
    return None


def _download_from_drive(file_id, dest_path):
    """See README's honesty note: the URL-parsing/size-limit logic here was
    tested directly; the live download from drive.google.com could not be
    tested end-to-end in the sandboxed build environment (its network
    egress is restricted to package registries, not Drive)."""
    session_ = requests.Session()
    base_url = "https://drive.google.com/uc?export=download"
    resp = session_.get(base_url, params={"id": file_id}, stream=True, timeout=20)

    confirm_token = None
    for key, value in resp.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
    if confirm_token is None and resp.headers.get("Content-Type", "").startswith("text/html"):
        match = re.search(r"confirm=([0-9A-Za-z_-]+)", resp.text)
        if match:
            confirm_token = match.group(1)

    if confirm_token:
        resp = session_.get(base_url, params={"id": file_id, "confirm": confirm_token},
                             stream=True, timeout=20)

    if resp.headers.get("Content-Type", "").startswith("text/html"):
        raise ValueError(
            "Drive returned a web page instead of the file - the file may not be "
            "shared as 'Anyone with the link'. Direct upload is more reliable if "
            "this keeps happening."
        )

    total = 0
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            total += len(chunk)
            if total > MAX_TRAINING_VIDEO_BYTES:
                raise ValueError(f"File exceeds the {MAX_TRAINING_VIDEO_BYTES // (1024*1024)}MB limit.")
            f.write(chunk)
    if total == 0:
        raise ValueError("Downloaded file was empty.")


@app.route("/api/train/upload-video", methods=["POST"])
@login_required
def train_upload_video():
    user_id = _current_user_id()
    label = request.form.get("label")
    if label not in trainer.LABELS:
        return jsonify({"ok": False, "error": f"invalid label: {label}"}), 400

    video_file = request.files.get("video")
    if not video_file:
        return jsonify({"ok": False, "error": "no video file provided"}), 400

    ext = os.path.splitext(video_file.filename or "")[1] or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            video_file.save(tmp_path)
        if os.path.getsize(tmp_path) > MAX_TRAINING_VIDEO_BYTES:
            return jsonify({"ok": False, "error": f"Video exceeds the "
                             f"{MAX_TRAINING_VIDEO_BYTES // (1024*1024)}MB limit."}), 400
        result = video_ingest.process_video_file(tmp_path, user_id, label)
        return jsonify(result)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/api/train/upload-drive-link", methods=["POST"])
@login_required
def train_upload_drive_link():
    user_id = _current_user_id()
    payload = request.get_json(force=True) or {}
    label = payload.get("label")
    drive_url = payload.get("drive_url", "")

    if label not in trainer.LABELS:
        return jsonify({"ok": False, "error": f"invalid label: {label}"}), 400

    host = urlparse(drive_url).hostname or ""
    if host not in ALLOWED_DRIVE_HOSTS:
        return jsonify({"ok": False, "error": "Only Google Drive links are supported here."}), 400

    file_id = _extract_drive_file_id(drive_url)
    if not file_id:
        return jsonify({"ok": False, "error": "Could not find a file ID in that link. "
                         "Use the 'Share' link for the file, not a folder."}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        _download_from_drive(file_id, tmp_path)
        result = video_ingest.process_video_file(tmp_path, user_id, label)
        return jsonify(result)
    except (requests.RequestException, ValueError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
