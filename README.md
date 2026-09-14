# FocusBuddy — Study Focus Companion

A local, full-stack web app for one family: a parent starts a study session from
a browser, the browser's own webcam/mic feed a simple focus-tracking backend,
and the child gets gentle spoken nudges — never a cloned voice, never a stored
biometric profile of anyone.

## Is this "finished"?

**Functionally, yes — it runs start to finish.** You can install it, open it in a
browser, run a real session against your own webcam and mic, get spoken alerts,
and get a saved report. I tested the full backend flow (start → classify frames
→ log speech → stop → generate + save report) end to end.

**As a thing to hand to a parent to run unsupervised on their own child, no.**
See "Before you actually use this" below — there's one real accuracy gap and a
few real-world steps (device install, review, maybe legal check) that are
outside what I can finish inside this chat.

## What changed from the original spec, and why (recap)

| Original spec | This build |
|---|---|
| Clone the parent's real voice for alerts | Browser's built-in text-to-speech voice (Web Speech Synthesis) — clearly synthetic, not the parent |
| Extract & store facial landmarks / a parent "corrective expression" model | No facial vectors stored anywhere; frames are analyzed in memory and discarded |
| Server-side audio capture & storage | Speech-to-text happens in the browser (Web Speech Recognition); only the transcribed *text* ever reaches the server, never audio |

## Architecture

The detection pipeline follows: **Camera → Person detection → Face/head/pose
detection → Book/notebook detection → Hand/writing detection → Activity
estimation → Temporal analysis → Distraction score → Decision engine →
Voice intervention** - see `app/vision_pipeline.py` for exactly how each
stage is implemented and, importantly, which stages are real detections
versus honest placeholders (book detection specifically - see below).

```
┌─────────────────────────── Browser (parent + child's device) ───────────────────────────┐
│  index.html + app.js                                                                      │
│   - asks OS/browser for camera + mic permission directly (real consent prompt)            │
│   - grabs a webcam frame every 5s, sends it to the backend                                │
│   - runs Web Speech Recognition locally -> sends only transcribed TEXT to backend         │
│     (this becomes the TALKING signal - see below)                                         │
│   - runs Web Speech Synthesis locally to SPEAK alerts (no cloned voice, ever)              │
│   - renders live state, distraction score, metrics, transcript, and the final report       │
└───────────────────────────────────────┬────────────────────────────────────────────────┘
                                          │  JSON over HTTP
                                          ▼
┌─────────────────────────────── Flask server (server.py) ────────────────────────────────┐
│  /api/start, /api/tick, /api/stop  -> per-user SessionManager, scoped to that account     │
│  /api/history, /api/history/<id>/timeline -> past sessions, scoped by user_id             │
│                                                                                            │
│  app/session_manager.py  - IDLE/ACTIVE/REPORTING state machine                            │
│  app/vision_pipeline.py  - THE pipeline above: YOLO object+pose detection when available,  │
│                            OpenCV Haar-cascade fallback otherwise, same 9-state output      │
│  app/focus_detector.py   - the Haar-cascade fallback detector specifically                 │
│  app/alert_system.py     - decides alert phrase + cooldown timing (text or real clip)      │
│  app/trainer.py          - OPTIONAL: train a numeric-features model from your own labels    │
│  app/video_ingest.py     - OPTIONAL: bulk-extract labeled examples from an uploaded video   │
│  app/db.py               - SQLite: users, sessions, and the full per-tick event timeline    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### The 9-state taxonomy

| State | Bucket | How it's actually decided |
|---|---|---|
| READING | Focused | face + eyes visible (or YOLO pose "upright"/"leaning_forward"), low motion |
| WRITING | Focused | motion detected in the desk-area region below the face |
| MEMORIZING | Focused | brief eyes-away/pause - treated as normal thinking, not flagged |
| LOOKING_AWAY | Distracted | head turned away from the desk, sustained |
| PLAYING_WITH_PHONE | Distracted | **only fires if `phone_detected` is real** (YOLO's `cell phone` COCO class) - see caveat below |
| TALKING | Distracted | **a real signal** - the browser's live speech transcript, not a video guess |
| AWAY_FROM_DESK | Distracted | no person/face detected, sustained |
| SLEEPING_DROWSY | Distracted | eyes away + very low motion for a long stretch |
| UNKNOWN | Neither | not enough signal to tell (e.g. a brief, ambiguous absence) |

The top-level report bucket (Focused / Distracted / Unknown) and the
alert-triggering logic both derive from this table - nothing needed a
trained model to work, matching the "no training needed" design point.
An **optional** from-scratch trainable model (`app/trainer.py`) still
exists if you want to layer your own labeled examples on top - see
"Training your own model" below - but it's not required for any of this
to function.

### The real vs. honest-placeholder parts

- **Real, when `requirements-vision.txt` installs successfully:** person
  detection, book detection, phone detection (YOLOv8 object detection -
  "book" and "cell phone" are real COCO classes), and body pose (YOLOv8-pose
  keypoints, giving genuine "leaning forward" / "head down" / "upright"
  reads) - see `app/vision_pipeline.py`.
- **Real, always:** face/eye presence and hand-motion proxy (OpenCV Haar
  cascades + frame differencing - ships inside `opencv-python`, no extra
  install), and TALKING (the live speech transcript).
- **Honest placeholder when YOLO isn't installed:** `book_detected` and
  `phone_detected` report `null`, not a guessed `True`/`False` - see the
  Haar-fallback path in `app/vision_pipeline.py`'s docstring. `PLAYING_WITH_PHONE`
  therefore can't fire without YOLO active.

### Metadata schema (recorded instead of images)

Every tick logs one row via `db.log_event()`: `timestamp`, `head_direction`,
`face_detected`, `person_detected`, `pose`, `hand_position`, `book_detected`,
`phone_detected`, `activity`, `intervention_given`. A parent's live label
click logs a matching row with `parent_feedback` set instead. This is the
full `session_events` table - queryable per session via
`GET /api/history/<id>/timeline` (scoped to your account).

### What is, and isn't, ever stored
- **Never stored:** raw video frames, raw audio, facial landmark vectors, any voice model.
- **Stored (SQLite, `data/focusbuddy.db`):** the metadata schema above,
  session start/end times, seconds-per-state totals, the generated report
  text, and transcribed dialogue lines (text only).

## Project layout
```
focusbuddy/
├── server.py                # Flask app + API routes + auth
├── requirements.txt
├── requirements-vision.txt   # optional: real YOLO object/pose detection
├── requirements-optional.txt # optional: MediaPipe (rarely needed - Haar covers most of this)
├── templates/
│   ├── index.html           # dashboard UI (requires login)
│   ├── login.html
│   └── register.html
├── static/
│   ├── app.js                # webcam capture, speech recognition/synthesis, API calls
│   ├── style.css
│   ├── manifest.json          # PWA manifest (installable on mobile)
│   ├── sw.js                  # minimal service worker for PWA installability
│   ├── icons/                 # PWA icons, generated from the tutor avatar art
│   └── audio/<user_id>/...    # per-family recorded clips (created as families record)
├── app/
│   ├── config.py             # thresholds, privacy switches
│   ├── vision_pipeline.py    # THE pipeline: YOLO if available, else Haar fallback, 9-state output
│   ├── focus_detector.py     # the Haar-cascade fallback detector specifically
│   ├── alert_system.py       # alert phrase + cooldown logic, per-user clip lookup
│   ├── dialogue_listener.py  # wraps transcribed text (feeds the TALKING signal)
│   ├── session_manager.py    # state machine + live DB event logging
│   ├── report_generator.py   # aggregate summary + Session-NNN timeline formatting
│   ├── calibration.py        # guided setup calibration - personalizes thresholds
│   ├── trainer.py            # OPTIONAL: per-user trainable numeric-features model
│   ├── video_ingest.py       # OPTIONAL: bulk training-data extraction from uploaded video
│   └── db.py                 # SQLite: users, sessions, and the full per-tick event timeline
└── data/
    ├── focusbuddy.db          # users + session history + event timeline (created on first run)
    └── training/<user_id>/    # per-family OPTIONAL trainer data + trained models
```

## Running it

```bash
cd focusbuddy
pip install -r requirements.txt --break-system-packages
python3 server.py
```

**Optional — real object/pose detection (person, book, phone, body pose):**
```bash
pip install -r requirements-vision.txt --break-system-packages
```
This installs `ultralytics` (YOLOv8), which needs PyTorch. **If you're on
an Intel Mac, this will likely fail or pull an old unsupported PyTorch
build** — Apple's move away from Intel hardware means PyTorch dropped
macOS x86_64 wheels starting with version 2.3 (April 2024). That's fine:
the app detects YOLO's absence automatically and runs the OpenCV
Haar-cascade fallback instead, covering the same 9-state output with real
(if rougher) signals. This should install fine on Apple Silicon Macs and
on Linux, including an ARM64 cloud VM.

**Optional — MediaPipe** (`requirements-optional.txt`): mostly superseded
by the YOLO/Haar setup above now; kept for anyone who already has it working.

Then open **http://127.0.0.1:5000** in Chrome or Edge (Web Speech Recognition
isn't supported in Firefox/Safari as of this writing — check current browser
support if that matters for your setup). You'll be asked to create an
account (`/register`) the first time, then log in. Enter the child's name
(age and task are optional, used for the session report's header), press
**Start session**, and allow the camera/microphone prompt.

See "Deploying for multiple families" below before exposing this beyond
your own network to other households.

## Recording your own guidance clips in-app

Bengali is now the default language on the setup screen (both for speech
recognition and the browser's fallback synthetic voice) — change the
dropdown if you'd rather default to something else. The English fallback
phrases in `app/alert_system.py` were also translated to Bangla, so if no
recorded clip exists yet for a situation, the synthetic voice reads Bangla
text instead of English text in a Bangla accent.

You no longer need to manually drop audio files into folders — the setup
screen has a built-in recorder:

1. Enter the child's name (the Bangla script text updates automatically to
   include it).
2. Read the suggested line aloud — or edit the text box first if you'd
   rather phrase it your own way — and press **Record**.
3. Press **Stop**, listen back to the preview, and press **Save clip** if
   you're happy with it.
4. Repeat for as many variations as you like, for either situation
   ("When attention drifts" / "When actively off-task"). Each save adds to
   the pool the alert system picks from at random — you don't need to
   record over an old one to add variety, and you can keep going even
   after closing and reopening the app.
5. Not happy with a take before saving? Press **Record again** instead of
   Save — it discards the current preview and starts a fresh recording
   right away.
6. Already-saved clips for each situation show up as a small playable list
   underneath, each with its own **Remove** button if you want to delete one.

This is a real recording of your own voice, saved exactly as spoken —
nothing here is synthesized or modeled. It's the in-app equivalent of the
manual "drop files into `static/audio/`" approach described below; both
write to the same folders, so you can mix and match (record some clips in
the browser, add others you recorded elsewhere).

## Adding your own Bangla audio alerts

Instead of (or before) relying on the browser's synthetic voice, you can drop
in real recorded clips — your own voice, in Bangla, or anything else — for
each situation:

```
focusbuddy/static/audio/
├── distracted/    # played when the child's attention has wandered
│   ├── clip1.mp3
│   └── clip2.mp3
├── disruptive/    # played when the child is actively off-task
│   └── clip1.mp3
├── reading/       # praise, played once when reading is detected starting
├── writing/       # praise, played once when writing motion is detected starting
└── memorizing/    # gentle encouragement during a thinking/recall pause
```

- Drop as many `.mp3`, `.wav`, `.ogg`, or `.m4a` files as you like into either
  folder — the app picks one at random each time an alert fires, so a few
  variations keeps it from feeling repetitive.
- If a folder is empty, the app automatically falls back to the browser's
  synthetic text-to-speech voice for that situation instead — nothing breaks
  if you only record clips for one situation to start.
- No transcript or text is required alongside a clip; it just plays the file
  as-is. These are real recordings you provide, not a synthesized or cloned
  voice — that distinction is intentional (see the design note above).
- After adding files, no restart is needed — the folder is checked fresh each
  time an alert fires.
- The in-app recorder (above) saves `.webm` files, which Chrome/Edge play
  back fine but which won't play in Safari — if you're on Safari, either
  test playback in Chrome or convert clips to `.mp3`/`.m4a` afterward.

## Fixing a real bug: reading vs. distracted was inverted for desk setups

Real feedback caught a genuine flaw, not a minor bug: the original heuristic
assumed "eyes visible to the camera" = engaged with material. That's true
if you're reading on the same screen the camera sits on — but for a
physical book or notebook on the desk below the camera, it's backwards.
Looking DOWN at a book naturally hides your eyes from the camera (partly
covered by eyelids, foreshortened angle) — the old logic read that as
"distracted." Looking UP toward the camera/laptop (e.g., while browsing
something unrelated) kept eyes clearly visible — the old logic called that
"reading." Exactly inverted for anyone with material on the desk.

**The fix:** a "Where's the study material?" setting on the setup screen —
**"On the desk"** (default) or **"On this device's screen"** — which flips
which signal means what:

| | Eyes NOT visible to camera | Eyes visible to camera |
|---|---|---|
| **Desk mode** (book/notebook below camera) | READING (expected — looking down at material) | MEMORIZING briefly, then DISTRACTED if sustained (looking up/away from material) |
| **Screen mode** (material co-located with camera) | MEMORIZING briefly, then DISTRACTED if sustained | READING (expected — looking at the screen) |

I verified this with a real image test: a synthetic "looking down at a
book" frame (eyes occluded, face still detected — mimicking what actually
happens when a face cascade sees a downward-tilted head) now correctly
reads as READING in desk mode, where it previously read as distracted.

**The YOLO path (when installed) got a related fix too:** it now uses body
posture ("leaning over the desk," from real pose keypoints) as a backup
signal for READING when the object detector doesn't confirm a book — closed
notebooks, spiral binding, and partial hand occlusion are all common misses
for the "book" object class, so leaning-forward posture catches genuine
reading/writing engagement that book-detection alone would miss.

**Honest limits that remain:** this is still a polarity choice based on a
setting you tell it, not real gaze tracking. It can't tell "looking down at
a book" apart from "looking down at a phone in your lap" in desk mode
without real object detection (which needs YOLO, unlikely to run on an
Intel Mac — see below). And it can't detect genuine drowsiness as
reliably in desk mode, since calm reading and dozing off both look like
"eyes hidden, low motion" from a webcam's perspective.

## Improving accuracy: patterns over time, and personalizing to your setup

Three concrete improvements, all built on the same idea: a single frame
from cheap heuristics is noisy, but *patterns* — across time, and specific
to your actual camera/desk/lighting — are much more reliable.

**1. Motion smoothing.** A single noisy frame (posture shift, a shadow,
camera auto-exposure adjusting) used to be able to instantly trigger
WRITING on its own. Motion is now averaged over the last 3 frames before
being compared to the threshold — a real, sustained motion pattern is
needed, not a one-frame spike.

**2. Temporal hysteresis on the displayed/logged state.** Previously, the
activity reported each tick was just whatever that single frame's raw read
was — meaning one noisy misclassification could flip the whole session's
state for that tick. Now a new activity needs a **majority across the last
3 ticks** (15 seconds) before it actually takes over; a lone outlier gets
absorbed instead of treated as a real change. Verified directly: fed a
sequence of `READING, READING, READING, LOOKING_AWAY, READING, READING` and
confirmed the single outlier tick never flipped the reported state, while a
genuinely sustained run of `LOOKING_AWAY` still took over within two ticks.
TALKING is deliberately exempt from this smoothing — it comes from the live
speech transcript, a real instantaneous signal, not a visual guess, so it
shouldn't lag behind stale visual history.

**3. Calibration — personalizing thresholds to this specific setup.** This
is probably the single biggest lever available. A generic
`WRITING_MOTION_THRESHOLD` tuned in the abstract can be badly wrong for any
particular camera angle, distance, or lighting. The new "Calibrate
detection to your setup" panel on the setup screen walks through three
~5-second phases — reading, writing, looking away — sampling real frames
from your actual setup, and computes a personalized writing-motion
threshold from the actual observed gap between your reading and writing
motion levels. I tested this directly: ran synthetic "reading" (static)
and "writing" (real shifting motion) samples through the full calibration
pipeline, confirmed it correctly computed a threshold sitting between the
two observed levels, confirmed a fresh `FocusDetector` actually loads and
applies that saved profile (not just computes and discards it), and
confirmed one family's calibration is invisible to another's account.

**The remaining biggest lever, not yet exercised: the trainable model.**
`app/trainer.py` — the from-scratch per-user classifier — starts with zero
data and does nothing until you actually label examples (live clicks or
bulk video upload, both already built). A model trained on 20-30 minutes
of your child's real sessions, in your real lighting, at your real desk,
will very likely outperform any of the three fixes above, because it's
learning your setup's actual patterns rather than applying general rules
to them. If accuracy still isn't where you want it after calibrating,
this is the next thing to actually try, not a further heuristic tweak.

## How detection actually works now

Two backends, chosen automatically per session:

- **If YOLOv8 (`requirements-vision.txt`) installs and loads:** real object
  detection (person, book, cell phone are genuine COCO classes) and real
  pose estimation (YOLOv8-pose keypoints) drive `head_direction`, `pose`,
  `book_detected`, and `phone_detected`. **This is very likely NOT
  installable on an Intel Mac** — PyTorch dropped macOS x86_64 support
  starting with version 2.3 (April 2024); the last compatible build was
  2.2.2. It should install fine on Apple Silicon Macs and on Linux,
  including an ARM64 cloud VM (e.g. Oracle Cloud's Always Free tier).
- **Otherwise (your situation, on an Intel Mac):** the OpenCV Haar-cascade
  fallback (`app/focus_detector.py`) runs instead — ships inside
  `opencv-python` itself, no extra install needed. `book_detected` and
  `phone_detected` report `null` in this mode (not a guessed answer), and
  `PLAYING_WITH_PHONE` can't fire without a real phone detector active.

Either way, the live dashboard shows which backend is active (a small
italic line under the state indicator) so you always know what's actually
running your session, not just what the code is capable of.

### Encouragement clips, not just corrections

Because reading/writing/memorizing are good behaviors, the app plays a
short praise clip when your child *starts* one of these (once per
transition, 90-second cooldown) — separate from corrective alerts for the
five distracted states, which require the sustained 15-second threshold.
Record these the same way as corrective clips — three sections on the
setup screen: "While reading well," "While writing well," and "While
pausing to remember something." All optional; skip any you don't want.

## Training your own model (optional, advanced)

Instead of relying only on the built-in heuristic thresholds, you can teach
FocusBuddy from labeled examples of your own child's real sessions.

**How it avoids storing images:** training a typical image classifier means
saving example photos. This one doesn't. Every frame already gets reduced to
nine numbers (face position/size, eye count, motion score, and how long eyes
have been away or the face missing — see `app/trainer.py`'s `FEATURE_NAMES`).
When you label a moment, it's *those numbers* that get logged with your
label, in a local file (`data/training_data.jsonl`) — never an image. A
small `RandomForestClassifier` (from scikit-learn) is then trained on that
numeric dataset.

**How to use it:**
1. On the setup screen, check "Training mode" before starting a session.
2. During the session, a labeling panel appears below the transcript.
   Click whichever button matches what's actually happening right now
   (Reading / Writing / Memorizing / Distracted / Disruptive) — each click
   logs the detector's current numbers tagged with that label.
3. Once you have several examples of at least two different labels
   (the button shows a live count), press **Train model now**.
4. **End the session and start a new one** — the trained model loads once
   when a session begins, so a model trained mid-session won't apply until
   the next one starts.
5. From then on, that session's classifications come from your trained
   model instead of the hardcoded thresholds — the app automatically
   prefers a trained model when one exists (`app/trainer.py: load_model()`),
   falling back to the heuristic only if none exists yet or a prediction
   fails for any reason.

**Be realistic about what this can learn.** It's a genuinely trained model,
not a hardcoded rule — but it's only as good as the nine numbers it sees.
It can learn patterns like "distraction, for this child, tends to come with
this much motion and this face position" — it cannot see raw video, so it
can't learn anything a motion score and eye count don't capture (it won't
notice, say, that a specific object in the room is a phone). More labeled
examples across a wider variety of real conditions (lighting, seating,
time of day) will generalize better than a handful collected in one sitting.

**Currently only wired into the Haar cascade path** (the practical path for
most setups). If MediaPipe is installed on your machine, sessions still use
its own separate heuristic rather than your trained model — extending that
is a reasonable next step but wasn't done here to keep scope manageable.

You can clear everything and start over anytime with **Reset training
data** in the training panel, which deletes both the logged examples and
the trained model file.

## The tutor avatar

There's now a simple animated face on the live session screen — expressions
change with what's happening (calm by default, a brief smile for
encouragement, a gentle concerned look for corrective alerts), and its mouth
moves while an alert clip or synthetic voice is speaking. The caption under
it reads "Your tutor — an animated helper, not a real person," on purpose:
same principle as everything else in this build — the child should always
be able to tell they're talking to software, not a disguised recording of
an actual adult.

The current design (a friendly bespectacled face with glasses, hair, and a
collar) was actually rendered and visually checked — including all three
expression states — before shipping, rather than guessed at blind. If you
want a different look, the whole face is plain SVG in
`templates/index.html` (static parts: collar, ears, hair, glasses, nose)
plus `static/app.js`'s `AVATAR_EXPRESSIONS` (the eyebrow/mouth paths that
change per expression) and `MOUTH_OPEN_PATH` (the talking-animation frame).

**Why not Linly-Talker (or a similar photorealistic digital-human stack):**
those systems need an NVIDIA GPU with CUDA — typically an RTX 3060 or
better for anything close to real-time — plus several GB of downloaded
model weights and a fairly particular PyTorch/conda setup. Two hard blockers
for this project specifically: Macs have no CUDA-capable GPU at all, so it
wouldn't run on the machine this was built for regardless of effort; and
that stack bundles voice-cloning TTS (CosyVoice), which conflicts with the
no-voice-cloning principle this whole app has been built around. What's
here instead is a plain SVG face driven by CSS/JS — no GPU, no downloads,
runs in any browser — with the trade-off that it's a simple animated
character, not a lifelike lip-synced face.

## Bulk-adding training data from a video

Clicking a label button live during a session works, but it's slow — one
example per click. If you have a video where the child is doing one thing
the whole time (a minute of them reading, say), you can bulk-add examples
from it instead, in the training panel's "Bulk-add examples from a video"
section:

- **Direct upload:** pick a video file (mp4/webm/mov, up to 80MB), choose
  the label that describes the whole clip, and upload. The app samples
  ~2 frames per second, runs the same feature extraction used live, and
  logs every sampled frame under that one label.
- **Google Drive link:** paste a Drive share link instead of uploading
  directly, if the video's already there. The link must be shared as
  "Anyone with the link," and it needs to point at a specific file, not a
  folder.

**Same privacy design as everything else:** the video is read frame-by-frame
from a temporary file and deleted immediately after processing (`finally`
block in `server.py`) — only the extracted numbers reach storage, same as
a live labeling click. I verified this with a real synthetic test video:
uploaded a clip, confirmed the numeric features were extracted correctly
(near-zero motion for a static "reading" clip, consistently high motion
for a "writing" clip with movement), trained on the result, and separately
confirmed a second test account saw none of the first account's
bulk-uploaded data.

**Honest caveat on the Google Drive path specifically:** I built and unit-
tested the URL validation and the "reject non-Drive hosts" security guard
(this matters — without it, this feature would let anyone make your server
fetch arbitrary internal/external URLs, a classic SSRF vulnerability). What
I could **not** test is the actual live download from `drive.google.com`,
because the sandboxed environment this was built in only allows outbound
network access to package registries (pypi/npm/github), not Drive. Google
also periodically changes how their "large file" download-confirmation page
works, which this handles via a commonly-used workaround that may need
adjusting over time. If the Drive link path ever silently fails, direct
upload is the reliable fallback — please test the Drive path once you've
deployed, since I couldn't verify it end-to-end myself.

Both paths respect the same 80MB size limit (checked both before and during
download, so an oversized file can't quietly consume all your disk/memory).

## Deploying for multiple families

This section covers what changed to make multi-family use possible, and —
just as important — what you're still responsible for before real other
families' children use this.

### What's now in place

- **Accounts.** Each family registers at `/register` and logs in at
  `/login`. Passwords are hashed with Werkzeug's `generate_password_hash`
  (never stored in plain text).
- **Per-family data isolation**, enforced at every layer:
  - Sessions and history are scoped by `user_id` in every database query
    (`app/db.py`) — one family literally cannot query another's rows, even
    by guessing a session ID (tested: returns 404, not another family's data).
  - Recorded audio clips live under `static/audio/<user_id>/<situation>/`,
    not a shared folder — one family can never hear another's recordings.
  - Trained models and training data live under `data/training/<user_id>/`
    — one family's labeled examples never train a model another family's
    sessions end up using.
  - Every `/api/*` route requires login (`@login_required`); unauthenticated
    requests get a 401, and visiting `/` while logged out redirects to `/login`.
- I verified all of this with actual multi-account tests — registering two
  separate families, generating data under each, and confirming neither
  could see, hear, or fetch the other's data before writing this section.

### What's still on you before onboarding real families

1. **HTTPS is mandatory, not optional.** Browsers refuse camera/mic access
   on any origin that isn't `localhost` or served over HTTPS — this is a
   hard browser rule, not a FocusBuddy setting. See the Cloudflare step below.
2. **A privacy policy and terms of service.** You're now processing camera
   frames, transcribed speech, and behavioral data about other people's
   children. Most jurisdictions expect a privacy policy for this regardless
   of company size, and children's-specific rules (COPPA in the US, GDPR-K
   in the EU, and equivalents elsewhere) may apply to you as an operator.
   I'm not a lawyer and this isn't legal advice — but I'd treat "talk to
   someone who knows children's-data law in your jurisdiction" as a real
   prerequisite here, not a nice-to-have.
3. **The audio-upload feature is now a real abuse surface.** Any account can
   upload arbitrary audio to be played to a child. With only your own
   family using it, that's fine. With strangers as users, you likely want
   at minimum: file-size limits, a review step before a newly uploaded clip
   can play, and a way to report/remove abusive content. None of that is
   built — it's a gap, not a false claim otherwise.
4. **This is still SQLite and a single-process Flask app.** Fine for a
   handful of families on modest hardware; it will not smoothly handle
   real concurrent load. If this grows, plan to move to PostgreSQL and a
   multi-worker setup.
5. **No password reset flow, no email verification, no rate limiting on
   login attempts.** Basic account hygiene features a real product needs
   before wide release — not built here.

### Hosting: Oracle Cloud "Always Free" + Cloudflare (genuinely zero cost)

I checked current options rather than relying on stale knowledge. Most
"free" PaaS tiers (Render, Railway, Fly.io) either sleep after inactivity
(bad — a real alert could get missed during a 10-30 second cold start) or
only offer free credits for a limited time. Oracle Cloud's **Always Free**
tier stands out: a real, persistent ARM VM (up to 24GB RAM, doesn't sleep,
doesn't expire) at no cost. Pairing it with Cloudflare's free tier solves
the HTTPS requirement without you managing certificates by hand.

**1. Get the VM:**
- Sign up at oracle.com/cloud/free (a card is required for identity
  verification, but Always Free resources aren't charged).
- Create a compute instance: shape "VM.Standard.A1.Flex" (Ampere ARM,
  Always Free eligible), Ubuntu 24.04, at least 2 OCPUs / 12GB RAM from
  your Always Free allowance.
- Open port 443 (and optionally 80) in the instance's security list/NSG.

**2. Install FocusBuddy on the VM:**
```bash
ssh ubuntu@<your-vm-ip>
sudo apt update && sudo apt install -y python3-pip python3-venv git
git clone <wherever you're hosting your copy of this code>
cd focusbuddy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export FOCUSBUDDY_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
# Save that secret key somewhere safe - losing it just logs everyone out,
# but you want it stable across restarts.
```

**Worth doing here specifically:** Oracle's Always Free tier is ARM64
(Ampere), and PyTorch has solid Linux ARM64 support — unlike your Intel
Mac, `pip install -r requirements-vision.txt` should actually work here,
giving every family real object/pose detection in production even if your
local dev machine only gets the Haar fallback. Worth testing after deploy.

**3. Run it with gunicorn (not the Flask dev server) in the background:**
```bash
gunicorn --bind 127.0.0.1:8000 --workers 2 server:app
```
For it to survive reboots/crashes, wrap it in a systemd service:
```ini
# /etc/systemd/system/focusbuddy.service
[Unit]
Description=FocusBuddy
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/focusbuddy
Environment="FOCUSBUDDY_SECRET_KEY=<paste your generated key here>"
ExecStart=/home/ubuntu/focusbuddy/.venv/bin/gunicorn --bind 127.0.0.1:8000 --workers 2 server:app
Restart=always

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl enable --now focusbuddy
```

**4. Put Cloudflare in front for free HTTPS:**
- Add your domain to Cloudflare (or use a free subdomain provider if you
  don't have one) and point its DNS at your Oracle VM's public IP.
- Install `cloudflared` and run a Cloudflare Tunnel pointed at
  `http://127.0.0.1:8000` — this gets you a valid HTTPS certificate and
  hides your VM's raw IP, with no certbot/nginx config needed:
  ```bash
  cloudflared tunnel login
  cloudflared tunnel create focusbuddy
  cloudflared tunnel route dns focusbuddy focusbuddy.yourdomain.com
  cloudflared tunnel run --url http://127.0.0.1:8000 focusbuddy
  ```
- Visit `https://focusbuddy.yourdomain.com` — camera/mic access will now work.

### Mobile: PWA, not a native app

A true App Store / Play Store app needs Xcode/Android Studio, developer
accounts, and physical-device testing — none of which exist in the
environment I build in, so I haven't pretended otherwise anywhere in this
project. What I've added instead: a **PWA manifest** (`static/manifest.json`)
and a minimal **service worker** (`static/sw.js`), both wired into
`index.html`. Once deployed over HTTPS, visiting the site on a phone offers
"Add to Home Screen" (iOS Safari) or an install prompt (Android Chrome) —
it then opens full-screen, with its own icon (generated from the tutor
avatar art), no browser chrome. It's not a native app, but it's the
realistic version of "feels like an app" for a Flask-based project like
this one.

## Before you actually use this with your child

1. **Both detection paths are still heuristic, not validated models.** The
   Haar cascade path is a real signal (face present/absent, eyes
   detected/not, motion below the face) but is sensitive to lighting,
   camera angle, and glasses — expect some false positives/negatives. If
   MediaPipe becomes available on your setup later, note its gaze-direction
   estimate (`FocusDetector._estimate_gaze_direction()`) is still a stub
   that always reports "on material" when a face is visible.
2. **The reading/writing/memorizing split is a further layer of heuristic on
   top of an already-rough signal** — it's motion-and-presence detection
   relabeled with friendlier names, not verified activity recognition. The
   "Activity Detail" section of the report is marked experimental for this
   reason; don't treat the percentages as precise.
3. **The optional trained model reports training-set accuracy, not a real
   held-out evaluation score** — with a small dataset (which is what you'll
   have starting out) that number is optimistic. Treat it as "did it learn
   *something*," not "this is how often it'll be right on a new session."
4. **No accuracy testing against real study-session footage has been done.**
   Before trusting the numbers, watch a session or two yourself alongside the
   live dashboard and see whether the state labels match what you're actually
   seeing.
5. **Browser support varies.** Web Speech Recognition (used for the transcript
   feature) is Chromium-based-browser-only today; test in whatever browser
   you'll actually run this in.
6. **This has not had a security review.** It's a small Flask dev server
   intended for one machine on one home network — fine for that, not vetted
   for anything beyond it.
7. **No legal/compliance review has been done.** Recording a child's speech
   (even as text) and monitoring them during study time may have
   jurisdiction-specific expectations around consent and disclosure — worth
   a quick check for wherever you live, especially if this is ever used
   outside your own household.
8. **Native mobile app packaging is out of scope here.** This is a browser-based
   web app, which is the realistic "finished, runnable today" form I could
   build and test in this environment. Turning it into an iOS/Android app
   would mean a separate build (e.g. React Native or a WebView wrapper) with
   its own app-store review — a real next step, but a different project.
