"""
calibration.py

A short guided calibration that personalizes detection thresholds to THIS
child's actual camera angle, desk setup, and lighting - instead of relying
on one generic threshold tuned in the abstract for every possible setup.

Why this matters for accuracy: the single biggest lever for the Haar
fallback's accuracy isn't a cleverer algorithm, it's whether
WRITING_MOTION_THRESHOLD (and similar constants) actually match what
"writing" looks like on THIS camera, at THIS distance, in THIS lighting.
A generic threshold picked in the abstract can be badly wrong for any
specific setup - too sensitive (fidgeting reads as writing) or not
sensitive enough (real writing reads as reading).

How it works: the parent walks the child through 3 short phases (~5s
each) - reading (look at the material, stay still), writing (write
something), looking away (look away from the desk). The frontend sends a
few sample frames per phase; this module runs them through the SAME
feature extraction FocusDetector already does, and records what
motion_score / eye-visibility actually looked like in each phase for THIS
setup. At the end, a personalized threshold is computed from the real
observed gap between "reading" and "writing" motion levels and saved to
disk - numbers only, never a frame, same as everywhere else in this app.
"""

import os
import json
import statistics

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "calibration")
PHASES = ("reading", "writing", "looking_away")


def _profile_path(user_id):
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"{user_id}.json")


class CalibrationSession:
    """In-memory buffer for one calibration run - nothing is written to
    disk until finish() succeeds."""

    def __init__(self):
        self.samples = {phase: [] for phase in PHASES}

    def add_sample(self, phase, motion_score, face_present, num_eyes):
        if phase not in PHASES:
            raise ValueError(f"unknown phase: {phase}")
        self.samples[phase].append({
            "motion_score": motion_score,
            "face_present": face_present,
            "num_eyes": num_eyes,
        })

    def counts(self):
        return {phase: len(samples) for phase, samples in self.samples.items()}

    def finish(self, user_id, min_samples=5):
        counts = self.counts()
        under = {p: c for p, c in counts.items() if c < min_samples}
        if under:
            return {
                "ok": False,
                "error": f"Not enough samples yet for: {', '.join(under)}. Keep the phase running a bit longer.",
                "counts": counts,
            }

        reading_motion = [s["motion_score"] for s in self.samples["reading"]]
        writing_motion = [s["motion_score"] for s in self.samples["writing"]]
        reading_eyes = [s["num_eyes"] for s in self.samples["reading"]]

        reading_motion_avg = statistics.mean(reading_motion)
        writing_motion_avg = statistics.mean(writing_motion)

        if writing_motion_avg <= reading_motion_avg * 1.15:
            return {
                "ok": False,
                "error": "Writing didn't show noticeably more motion than reading in this "
                         "run. Try again with clearer writing motion in view of the camera, "
                         "or check that the desk area below your face is actually visible to "
                         "the camera. Keeping the current thresholds for now.",
                "counts": counts,
                "reading_motion_avg": round(reading_motion_avg, 2),
                "writing_motion_avg": round(writing_motion_avg, 2),
            }

        # Personalized threshold: the midpoint between this setup's OWN
        # observed reading and writing motion levels, not a generic constant.
        writing_motion_threshold = (reading_motion_avg + writing_motion_avg) / 2

        profile = {
            "writing_motion_threshold": round(writing_motion_threshold, 2),
            "reading_motion_baseline": round(reading_motion_avg, 2),
            "writing_motion_baseline": round(writing_motion_avg, 2),
            "eyes_usually_hidden_when_reading": statistics.mean(reading_eyes) < 0.5,
            "sample_counts": counts,
        }

        with open(_profile_path(user_id), "w") as f:
            json.dump(profile, f, indent=2)

        return {"ok": True, "profile": profile}


def load_profile(user_id):
    path = _profile_path(user_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def reset_profile(user_id):
    path = _profile_path(user_id)
    if os.path.exists(path):
        os.remove(path)
