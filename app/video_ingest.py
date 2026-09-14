"""
video_ingest.py

Bulk training data extraction from an uploaded video: the parent uploads a
clip where the child is doing ONE thing the whole time (e.g. "this whole
video is Alex reading"), and this samples frames from it, runs the SAME
feature extraction FocusDetector already uses live, and logs each sampled
frame's numeric features with that one label - no click-per-moment needed.

Same privacy boundary as everything else here: the video file is read
frame-by-frame from a temp file and deleted immediately after (see
server.py's upload handler) - only the extracted numbers ever reach
app/trainer.py's storage, never the video itself or any frame image.

Video-time vs wall-clock time: FocusDetector's gaze/face-missing timers
normally run on wall-clock time (time.time()) for a live session. For a
video file, we want them to reflect the VIDEO's own timeline instead -
otherwise "how long the eyes were away" would reflect how fast this
machine can run OpenCV, not what happened in the video. classify_frame()
accepts a `now` override for exactly this reason.
"""

import os
import cv2

from app import config, trainer
from app.focus_detector import FocusDetector

MAX_FRAMES_PER_UPLOAD = 200
SAMPLE_RATE_HZ = 2  # how many frames per second of video we actually analyze


def process_video_file(path, user_id, label, max_frames=MAX_FRAMES_PER_UPLOAD):
    """
    Reads the video at `path`, samples frames at SAMPLE_RATE_HZ, extracts
    features for each sampled frame, and logs them all under `label`.
    Returns a summary dict. Never touches the caller's copy of the file -
    doesn't delete it; that's the caller's responsibility (see server.py).
    """
    if label not in trainer.LABELS:
        return {"ok": False, "error": f"invalid label: {label}"}

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"ok": False, "error": "Could not open the video file. Is it a supported format (mp4, webm, mov)?"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    frame_interval = max(1, round(fps / SAMPLE_RATE_HZ))

    detector = FocusDetector(
        gaze_away_threshold=config.DISTRACTED_GAZE_AWAY_THRESHOLD,
        writing_motion_threshold=config.WRITING_MOTION_THRESHOLD,
    )

    frame_index = 0
    sampled_count = 0
    video_time = 0.0
    seconds_per_frame = 1.0 / fps

    try:
        while sampled_count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break  # end of video

            if frame_index % frame_interval == 0:
                detector.classify_frame(frame, now=video_time)
                features = detector.get_last_features()
                trainer.append_example(user_id, features, label)
                sampled_count += 1
                video_time += frame_interval * seconds_per_frame
            else:
                video_time += seconds_per_frame

            frame_index += 1
    finally:
        cap.release()

    if sampled_count == 0:
        return {"ok": False, "error": "No frames could be read from this video."}

    return {
        "ok": True,
        "frames_added": sampled_count,
        "label": label,
        "counts": trainer.label_counts(user_id),
    }
