"""
smoke_test.py

Quick sanity check of the backend pipeline without a browser - creates a
throwaway test account, simulates a short session with a blank frame (no
real camera) and one piece of injected dialogue, and prints the report.
Useful for confirming your install works before opening the dashboard for
real.

Run: python3 smoke_test.py
"""

from app import db
from app.session_manager import SessionManager

db.init_db()

# Reuse the test account across runs instead of piling up new ones.
user = db.verify_user("smoke_test_user", "smoke_test_password")
if user is None:
    result = db.create_user("smoke_test_user", "smoke_test_password")
    user = {"id": result["user_id"]}

session = SessionManager(user_id=user["id"], child_name="Alex", child_age=8,
                          task="Reading", language="en-US")
session.start()

for i in range(10):
    transcript = "I'm tired" if i == 4 else None
    result = session.tick(frame=None, transcript_text=transcript)
    print(f"tick {i}: activity={result['activity']:<20} state={result['state']:<12} "
          f"backend={result['detection_backend']} audio={result['alert_audio_url']} "
          f"text={result['alert_text']}")

report = session.stop()
print("\n" + report)
