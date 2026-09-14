"""
trainer.py

A small, honest "trained model" pipeline for FocusBuddy - now scoped PER
USER, so one family's labeled examples never train a model that another
family's sessions end up using.

How it works:
  - While a session is active, the parent can click a label button
    ("This is: Reading" / "Writing" / etc.) for whatever the child is
    actually doing right now.
  - append_example() logs the NUMERIC feature vector FocusDetector already
    computed for the most recent frame, tagged with that label, into that
    user's own training log - never a raw image.
  - train_and_save() fits a small RandomForestClassifier on that user's
    examples and saves it to that user's own model file.
  - load_model() loads a given user's model back for focus_detector.py to
    use, if one exists.

Everything here is plain numbers in a local per-user JSONL file - no
images, no facial vectors, nothing that identifies the child beyond "a
face was roughly this big, roughly here, eyes visible or not, this much
motion."
"""

import os
import json
import time

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "training")

# Order matters - this is the exact vector both training and inference use.
FEATURE_NAMES = [
    "face_present",
    "face_w_ratio",
    "face_h_ratio",
    "face_x_ratio",
    "face_y_ratio",
    "num_eyes",
    "motion_score",
    "gaze_away_duration",
    "face_missing_duration",
]

LABELS = ("READING", "WRITING", "MEMORIZING", "LOOKING_AWAY",
          "PLAYING_WITH_PHONE", "AWAY_FROM_DESK", "SLEEPING_DROWSY", "TALKING")

try:
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def _user_dir(user_id):
    path = os.path.join(DATA_ROOT, str(user_id))
    os.makedirs(path, exist_ok=True)
    return path


def _training_log_path(user_id):
    return os.path.join(_user_dir(user_id), "training_data.jsonl")


def _model_path(user_id):
    return os.path.join(_user_dir(user_id), "trained_model.joblib")


def append_example(user_id, features: dict, label: str):
    if label not in LABELS:
        raise ValueError(f"Unknown label: {label}")
    record = {
        "features": {name: features.get(name, 0) for name in FEATURE_NAMES},
        "label": label,
        "ts": time.time(),
    }
    with open(_training_log_path(user_id), "a") as f:
        f.write(json.dumps(record) + "\n")


def load_dataset(user_id):
    path = _training_log_path(user_id)
    if not os.path.exists(path):
        return [], []
    X, y = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            X.append([rec["features"].get(name, 0) for name in FEATURE_NAMES])
            y.append(rec["label"])
    return X, y


def label_counts(user_id):
    _, y = load_dataset(user_id)
    counts = {label: 0 for label in LABELS}
    for label in y:
        counts[label] = counts.get(label, 0) + 1
    return counts


def train_and_save(user_id, min_examples_per_class=5):
    if not SKLEARN_AVAILABLE:
        return {"ok": False, "error": "scikit-learn is not installed on this machine."}

    X, y = load_dataset(user_id)
    counts = label_counts(user_id)

    distinct_labels_present = {label for label in y}
    if len(distinct_labels_present) < 2:
        return {
            "ok": False,
            "error": "Need labeled examples from at least 2 different situations before training.",
            "counts": counts,
        }

    under = {k: v for k, v in counts.items() if 0 < v < min_examples_per_class}
    if under:
        need = ", ".join(f"{k} ({v}/{min_examples_per_class})" for k, v in under.items())
        return {
            "ok": False,
            "error": f"Not enough examples yet for: {need}. Keep labeling.",
            "counts": counts,
        }

    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=0)
    clf.fit(X, y)
    joblib.dump(clf, _model_path(user_id))

    train_accuracy = round(clf.score(X, y), 3)
    return {"ok": True, "examples": len(y), "counts": counts, "train_accuracy": train_accuracy}


def load_model(user_id):
    path = _model_path(user_id)
    if not SKLEARN_AVAILABLE or not os.path.exists(path):
        return None
    try:
        import joblib as _joblib
        return _joblib.load(path)
    except Exception:
        return None


def model_exists(user_id):
    return os.path.exists(_model_path(user_id))


def reset_training_data(user_id):
    for path in (_training_log_path(user_id), _model_path(user_id)):
        if os.path.exists(path):
            os.remove(path)
