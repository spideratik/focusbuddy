"""
report_generator.py

Builds:
  1. The parent-facing aggregate summary across all 9 states.
  2. A "Session 001 / Child: 8 years old / Task: Reading / Duration: 30 min"
     style plain-text timeline, from the database's session_events rows
     (app/db.py: get_timeline) - one line per activity, plus
     "voice reminder" / "encouragement" / "parent labels: X" event lines.
"""

from datetime import datetime

ALL_STATES = ("READING", "WRITING", "MEMORIZING", "LOOKING_AWAY",
              "PLAYING_WITH_PHONE", "TALKING", "AWAY_FROM_DESK",
              "SLEEPING_DROWSY", "UNKNOWN")

GOOD_STATES = {"READING", "WRITING", "MEMORIZING"}
PROBLEM_STATES = {"LOOKING_AWAY", "PLAYING_WITH_PHONE", "TALKING", "AWAY_FROM_DESK", "SLEEPING_DROWSY"}


def _pct(part, total):
    return round((part / total) * 100, 1) if total > 0 else 0.0


def _format_mmss(seconds):
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def generate_report(state_seconds, dialogue_log, timeline, session_start, session_end,
                     child_name, child_age=None, task=None, session_number=None):
    total_seconds = sum(state_seconds.values())
    total_minutes = round(total_seconds / 60, 1)

    focus_seconds = sum(state_seconds.get(s, 0) for s in GOOD_STATES)
    distract_seconds = sum(state_seconds.get(s, 0) for s in PROBLEM_STATES)
    unknown_seconds = state_seconds.get("UNKNOWN", 0)

    focus_pct = _pct(focus_seconds, total_seconds)
    distract_pct = _pct(distract_seconds, total_seconds)
    unknown_pct = _pct(unknown_seconds, total_seconds)

    activity_lines = "\n".join(
        f"- {label.title().replace('_', ' ')}: {_pct(state_seconds.get(label, 0), total_seconds)}%"
        for label in ALL_STATES
        if state_seconds.get(label, 0) > 0
    ) or "- No activity recorded."

    dialogue_lines = "\n".join(
        f'- {datetime.fromtimestamp(d["timestamp"]).strftime("%H:%M:%S")} - '
        f'Child said: "{d["text"]}" (Context: {d["context"]})'
        for d in dialogue_log
    ) or "- No verbal check-ins recorded this session."

    insights = _generate_insights(focus_pct, distract_pct, state_seconds, total_seconds, dialogue_log)

    aggregate = f"""
## 📊 STUDY SESSION REPORT

### ⏳ Session Summary
- Total Duration: {total_minutes} minutes
- Focus Score: {focus_pct}% ⭐

### 📉 Top-Level Breakdown
- Focused: {focus_pct}% (reading, writing, or memorizing)
- Distracted: {distract_pct}% (looking away, on a phone, talking, away from the desk, or drowsy)
- Unknown: {unknown_pct}% (not enough signal to tell)

### 🔍 Activity Detail
{activity_lines}

### 🗣️ Child's Verbal Feedback & Responses
{dialogue_lines}

### 💡 Parent Insights & Recommendations
{insights}
""".strip()

    timeline_header = [f"Session {session_number:03d}" if session_number else "Session"]
    if child_age is not None:
        timeline_header.append(f"Child: {child_age} years old")
    if task:
        timeline_header.append(f"Task: {task}")
    timeline_header.append(f"Duration: {int(round(total_minutes))} min")

    timeline_lines = []
    for event in timeline:
        elapsed = event["timestamp"] - session_start
        if event.get("activity") == "PARENT_FEEDBACK":
            line = f"parent labels: {(event.get('parent_feedback') or '').lower().replace('_', ' ')}"
        else:
            line = (event.get("activity") or "unknown").lower().replace("_", " ")
        timeline_lines.append(f"{_format_mmss(elapsed)} - {line}")
        if event.get("intervention_given") == "corrective":
            timeline_lines.append(f"{_format_mmss(elapsed)} - voice reminder")
        elif event.get("intervention_given") == "encouragement":
            timeline_lines.append(f"{_format_mmss(elapsed)} - encouragement")

    timeline_block = "\n".join(timeline_header) + ("\n" + "\n".join(timeline_lines) if timeline_lines else "")

    return f"{aggregate}\n\n### 🕒 Full Timeline\n```\n{timeline_block}\n```"


def _generate_insights(focus_pct, distract_pct, state_seconds, total_seconds, dialogue_log):
    insights = []
    if distract_pct > 25:
        insights.append("- Distraction was notable this session. Consider a short "
                         "stretch or water break partway through to reset attention.")

    away_pct = _pct(state_seconds.get("AWAY_FROM_DESK", 0), total_seconds)
    drowsy_pct = _pct(state_seconds.get("SLEEPING_DROWSY", 0), total_seconds)
    talking_pct = _pct(state_seconds.get("TALKING", 0), total_seconds)

    if away_pct > 15:
        insights.append("- Significant time was spent away from the desk. Worth checking "
                         "whether the study slot or environment is a good fit right now.")
    if drowsy_pct > 10:
        insights.append("- Some signs of drowsiness came up. If this is a recurring pattern, "
                         "the study time might be working against their natural energy levels.")
    if talking_pct > 15:
        insights.append("- There was a fair amount of talking during this solo session - "
                         "worth a quick check on what's pulling their attention.")

    if any("tired" in d["text"].lower() for d in dialogue_log):
        insights.append("- Your child mentioned feeling tired - it may be worth checking "
                         "whether the study slot lines up with their natural energy levels.")

    writing_pct = _pct(state_seconds.get("WRITING", 0), total_seconds)
    reading_pct = _pct(state_seconds.get("READING", 0), total_seconds)
    if writing_pct < 5 and reading_pct > 40:
        insights.append("- Mostly reading with little writing this session - mixing in "
                         "some active recall (writing answers, summarizing) tends to help "
                         "retention more than reading alone.")

    if not insights:
        insights.append("- Solid session overall - no major concerns to flag.")
    return "\n".join(insights)
