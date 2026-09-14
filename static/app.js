const TICK_INTERVAL_MS = 5000; // must match app/config.py FRAME_ANALYSIS_INTERVAL

const setupPanel = document.getElementById('setup-panel');
const livePanel = document.getElementById('live-panel');
const reportPanel = document.getElementById('report-panel');

const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const downloadBtn = document.getElementById('download-btn');
const newSessionBtn = document.getElementById('new-session-btn');

const video = document.getElementById('camera-feed');
const canvas = document.getElementById('capture-canvas');
const stateDot = document.querySelector('.state-dot');
const stateLabel = document.getElementById('state-label');
const activityLabel = document.getElementById('activity-label');
const alertBanner = document.getElementById('alert-banner');
const transcriptList = document.getElementById('transcript-list');
const reportText = document.getElementById('report-text');

let mediaStream = null;
let tickTimer = null;
let recognizer = null;
let pendingTranscript = '';
let childName = 'Alex';
let lastReportText = '';
let trainingModeEnabled = false;

const STATE_COPY = {
  FOCUSED: 'Focused and working',
  DISTRACTED: 'Attention needed',
  UNKNOWN: 'Not enough signal to tell',
};

// Default Bangla guidance scripts. Parents can edit the text before
// recording - editing marks the field so we stop auto-filling it when the
// child's name changes.
const BANGLA_SCRIPTS = {
  looking_away: (name) => `${name}, চলো আবার বইয়ের দিকে মনোযোগ দিই। তুমি খুব ভালো করছো!`,
  playing_with_phone: (name) => `${name}, ওটা একটু রেখে দাও, চলো এই পাতাটা শেষ করি।`,
  away_from_desk: (name) => `${name}, তুমি কি ফিরে আসতে পারবে? চলো আবার শুরু করি।`,
  sleeping_drowsy: (name) => `${name}, একটু ক্লান্ত লাগছে বুঝি? একটা ছোট বিরতি নাও।`,
  talking: (name) => `${name}, একটু শান্ত হয়ে আবার পড়ায় মনোযোগ দিই।`,
  reading: (name) => `${name}, তুমি খুব ভালো পড়ছো। এভাবেই চালিয়ে যাও!`,
  writing: (name) => `${name}, খুব সুন্দর লিখছো, চালিয়ে যাও!`,
  memorizing: (name) => `${name}, একটু থেমে মনে করাটাও ঠিক আছে। তুমি ভালো করছো।`,
};

const ACTIVITY_COPY = {
  READING: 'Reading',
  WRITING: 'Writing',
  MEMORIZING: 'Pausing to remember',
  LOOKING_AWAY: 'Looking away',
  PLAYING_WITH_PHONE: 'On a phone / object',
  TALKING: 'Talking',
  AWAY_FROM_DESK: 'Away from the desk',
  SLEEPING_DROWSY: 'Drowsy',
  UNKNOWN: 'Unclear',
};

const BACKEND_COPY = {
  yolo: '🔍 Using real object/pose detection',
  haar_fallback: '🔍 Using fallback heuristic (no YOLO/torch available)',
  simulation: '🔍 Simulated (no camera frame)',
};

// ---------- Tutor avatar ----------

const AVATAR_EXPRESSIONS = {
  idle: {
    eyebrowLeft: 'M 61 76 Q 75 68 89 76',
    eyebrowRight: 'M 111 76 Q 125 68 139 76',
    mouthClosed: 'M 78 132 Q 100 144 122 132',
    cheeks: false,
  },
  happy: {
    eyebrowLeft: 'M 61 72 Q 75 64 89 72',
    eyebrowRight: 'M 111 72 Q 125 64 139 72',
    mouthClosed: 'M 74 128 Q 100 154 126 128',
    cheeks: true,
  },
  concerned: {
    eyebrowLeft: 'M 61 80 Q 75 70 89 74',
    eyebrowRight: 'M 111 74 Q 125 70 139 80',
    mouthClosed: 'M 80 136 Q 100 129 120 136',
    cheeks: false,
  },
};

const avatarMouth = document.getElementById('avatar-mouth');
const avatarEyebrowLeft = document.getElementById('avatar-eyebrow-left');
const avatarEyebrowRight = document.getElementById('avatar-eyebrow-right');
const avatarCheekLeft = document.getElementById('avatar-cheek-left');
const avatarCheekRight = document.getElementById('avatar-cheek-right');

let currentExpression = 'idle';
let isTalking = false;
let talkInterval = null;
const MOUTH_OPEN_PATH = 'M 80 130 Q 100 158 120 130 Q 100 142 80 130';

function setAvatarExpression(name) {
  currentExpression = name;
  if (!avatarMouth || !avatarEyebrowLeft || !avatarEyebrowRight) return;
  const expr = AVATAR_EXPRESSIONS[name] || AVATAR_EXPRESSIONS.idle;
  avatarEyebrowLeft.setAttribute('d', expr.eyebrowLeft);
  avatarEyebrowRight.setAttribute('d', expr.eyebrowRight);
  if (avatarCheekLeft) avatarCheekLeft.style.opacity = expr.cheeks ? '0.5' : '0';
  if (avatarCheekRight) avatarCheekRight.style.opacity = expr.cheeks ? '0.5' : '0';
  if (!isTalking) avatarMouth.setAttribute('d', expr.mouthClosed);
}

function setAvatarTalking(talking) {
  isTalking = talking;
  if (!avatarMouth) return;
  if (talking) {
    let open = false;
    talkInterval = setInterval(() => {
      const expr = AVATAR_EXPRESSIONS[currentExpression] || AVATAR_EXPRESSIONS.idle;
      avatarMouth.setAttribute('d', open ? expr.mouthClosed : MOUTH_OPEN_PATH);
      open = !open;
    }, 160);
  } else {
    clearInterval(talkInterval);
    const expr = AVATAR_EXPRESSIONS[currentExpression] || AVATAR_EXPRESSIONS.idle;
    avatarMouth.setAttribute('d', expr.mouthClosed);
  }
}

// Safe listener attachment: never lets one missing element break every
// listener registered after it in the script (a real fragility this
// caught - see README changelog note).
function on(id, event, handler) {
  const el = document.getElementById(id);
  if (!el) {
    console.warn(`FocusBuddy: expected element #${id} not found - skipping its ${event} handler.`);
    return;
  }
  el.addEventListener(event, handler);
}

let activeRecorders = {}; // situation -> MediaRecorder

on('logout-btn', 'click', async () => {
  await fetch('/logout', { method: 'POST' });
  window.location.href = '/login';
});

on('start-btn', 'click', startSession);
on('stop-btn', 'click', stopSession);
on('download-btn', 'click', downloadReport);
on('new-session-btn', 'click', () => {
  reportPanel.hidden = true;
  setupPanel.hidden = false;
  loadHistory();
});

async function startSession() {
  childName = document.getElementById('child-name').value.trim() || 'Alex';
  const childAgeInput = document.getElementById('child-age');
  const childAge = childAgeInput && childAgeInput.value ? parseInt(childAgeInput.value, 10) : null;
  const taskInput = document.getElementById('task');
  const task = taskInput ? taskInput.value.trim() : '';
  const materialPositionSelect = document.getElementById('material-position');
  const materialPosition = materialPositionSelect ? materialPositionSelect.value : 'desk';
  const language = document.getElementById('language').value;
  const trainingCheckbox = document.getElementById('training-mode-checkbox');
  trainingModeEnabled = trainingCheckbox ? trainingCheckbox.checked : false;

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert('This browser can\'t access the camera/microphone from this page. ' +
          'Make sure you\'re using http://127.0.0.1:5000 or http://localhost:5000 ' +
          '(not a plain IP address, and not opened as a local file) in an up-to-date ' +
          'Chrome or Edge.');
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  } catch (err) {
    console.error('getUserMedia failed:', err.name, err.message);
    alert(describeMediaError(err) + '\n\n(Technical detail: ' + err.name + ')');
    return;
  }

  video.srcObject = mediaStream;

  await fetch('/api/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ child_name: childName, child_age: childAge, task, language, material_position: materialPosition }),
  });

  document.querySelector('.transcript-feed h3').textContent = `What ${childName} has said`;
  transcriptList.innerHTML = '<li class="muted">Nothing yet.</li>';
  alertBanner.hidden = true;
  setAvatarExpression('idle');

  const trainingPanel = document.getElementById('training-panel');
  if (trainingPanel) {
    trainingPanel.hidden = !trainingModeEnabled;
    if (trainingModeEnabled) {
      const statusMsg = document.getElementById('train-status-message');
      if (statusMsg) statusMsg.textContent = '';
      refreshTrainingStatus();
    }
  }

  setupPanel.hidden = true;
  livePanel.hidden = false;

  startSpeechRecognition(language);
  tickTimer = setInterval(runTick, TICK_INTERVAL_MS);
}

function runTick() {
  const ctx = canvas.getContext('2d');
  canvas.width = video.videoWidth || 320;
  canvas.height = video.videoHeight || 240;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const frameData = canvas.toDataURL('image/jpeg', 0.6);

  const transcript = pendingTranscript.trim();
  pendingTranscript = '';

  fetch('/api/tick', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ frame: frameData, transcript: transcript || null }),
  })
    .then((r) => r.json())
    .then(handleTickResult)
    .catch(() => {});

  if (transcript) {
    addTranscriptLine(transcript);
  }
}

function handleTickResult(data) {
  if (!data.ok) return;

  stateDot.className = `state-dot ${data.state}`;
  stateLabel.textContent = STATE_COPY[data.state] || data.state;

  if (data.state === 'FOCUSED') {
    setAvatarExpression('idle');
  } else if (data.state === 'DISTRACTED') {
    setAvatarExpression('concerned');
  } else {
    setAvatarExpression('idle');
  }

  if (data.activity && ACTIVITY_COPY[data.activity]) {
    activityLabel.hidden = false;
    let text = ACTIVITY_COPY[data.activity];
    if (typeof data.distraction_score === 'number') {
      text += ` · distraction score: ${data.distraction_score}`;
    }
    activityLabel.textContent = text;
  } else {
    activityLabel.hidden = true;
  }

  const backendLabel = document.getElementById('backend-label');
  if (backendLabel && data.detection_backend) {
    backendLabel.hidden = false;
    backendLabel.textContent = BACKEND_COPY[data.detection_backend] || data.detection_backend;
  }

  const seconds = data.state_seconds || {};
  const focused = (seconds.READING || 0) + (seconds.WRITING || 0) + (seconds.MEMORIZING || 0);
  const distracted = (seconds.LOOKING_AWAY || 0) + (seconds.PLAYING_WITH_PHONE || 0) +
                      (seconds.TALKING || 0) + (seconds.AWAY_FROM_DESK || 0) + (seconds.SLEEPING_DROWSY || 0);
  const unknown = seconds.UNKNOWN || 0;

  document.getElementById('metric-focused').textContent = `${Math.round(focused)}s`;
  document.getElementById('metric-distracted').textContent = `${Math.round(distracted)}s`;
  document.getElementById('metric-unknown').textContent = `${Math.round(unknown)}s`;

  ['READING', 'WRITING', 'MEMORIZING', 'LOOKING_AWAY', 'PLAYING_WITH_PHONE',
   'TALKING', 'AWAY_FROM_DESK', 'SLEEPING_DROWSY'].forEach((key) => {
    const el = document.getElementById(`metric-${key}`);
    if (el) el.textContent = `${Math.round(seconds[key] || 0)}s`;
  });

  if (data.alert_audio_url || data.alert_text) {
    const isEncouragement = data.alert_kind === 'encouragement';
    alertBanner.hidden = false;
    alertBanner.className = `alert-banner ${isEncouragement ? 'encouragement' : ''}`;
    alertBanner.textContent = data.alert_text || (isEncouragement ? '🔊 Nice work!' : '🔊 Playing a reminder…');
    setAvatarExpression(isEncouragement ? 'happy' : 'concerned');
    if (data.alert_audio_url) {
      playClip(data.alert_audio_url);
    } else if (data.alert_text) {
      speak(data.alert_text);
    }
  }
}

function describeMediaError(err) {
  switch (err.name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
      return 'Camera/microphone access was blocked. Check two places: ' +
        '(1) the camera icon in your browser\'s address bar — click it and make ' +
        'sure this site is set to "Allow", then reload; ' +
        '(2) your Mac\'s System Settings → Privacy & Security → Camera (and ' +
        'Microphone) — make sure your browser (Chrome/Edge/etc.) is checked there.';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return 'No camera or microphone was found on this computer. ' +
        'If you have an external webcam, make sure it\'s connected.';
    case 'NotReadableError':
    case 'TrackStartError':
      return 'The camera or microphone seems to be in use by another app ' +
        '(Zoom, FaceTime, Photo Booth, another browser tab, etc.). ' +
        'Close other apps using the camera and try again.';
    case 'OverconstrainedError':
      return 'No camera/microphone on this computer matched what was requested. ' +
        'This is unusual — try a different device if you have one.';
    default:
      return 'Could not access the camera/microphone (' + (err.message || 'unknown error') + '). ' +
        'Make sure you\'re on http://127.0.0.1:5000 or http://localhost:5000.';
  }
}

function playClip(url) {
  const audio = new Audio(url);
  audio.addEventListener('play', () => setAvatarTalking(true));
  audio.addEventListener('ended', () => setAvatarTalking(false));
  audio.addEventListener('error', () => setAvatarTalking(false));
  audio.play().catch((err) => {
    setAvatarTalking(false);
    console.warn('Could not play alert clip, falling back to nothing:', err);
  });
}

function speak(text) {
  if (!('speechSynthesis' in window)) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.98;
  utterance.onstart = () => setAvatarTalking(true);
  utterance.onend = () => setAvatarTalking(false);
  utterance.onerror = () => setAvatarTalking(false);
  window.speechSynthesis.speak(utterance);
}

function startSpeechRecognition(language) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    console.warn('Speech recognition not supported in this browser; dialogue logging disabled.');
    return;
  }
  recognizer = new SpeechRecognition();
  recognizer.continuous = true;
  recognizer.interimResults = false;
  recognizer.lang = language;

  recognizer.onresult = (event) => {
    const last = event.results[event.results.length - 1];
    if (last.isFinal) {
      pendingTranscript += ' ' + last[0].transcript;
    }
  };
  recognizer.onerror = () => {};
  recognizer.onend = () => {
    // Browsers auto-stop recognition periodically; restart while session is live.
    if (!livePanel.hidden) recognizer.start();
  };
  recognizer.start();
}

function addTranscriptLine(text) {
  if (transcriptList.querySelector('.muted')) transcriptList.innerHTML = '';
  const li = document.createElement('li');
  const time = new Date().toLocaleTimeString();
  li.textContent = `${time} — “${text}”`;
  transcriptList.prepend(li);
}

async function stopSession() {
  clearInterval(tickTimer);
  if (recognizer) recognizer.stop();
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());

  const res = await fetch('/api/stop', { method: 'POST' });
  const data = await res.json();

  livePanel.hidden = true;
  if (data.ok) {
    reportText.textContent = data.report;
    lastReportText = data.report;
    reportPanel.hidden = false;
  } else {
    setupPanel.hidden = false;
  }
}

function downloadReport() {
  const blob = new Blob([lastReportText], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `focusbuddy-report-${Date.now()}.md`;
  a.click();
  URL.revokeObjectURL(url);
}

async function loadHistory() {
  const res = await fetch('/api/history');
  const sessions = await res.json();
  const list = document.getElementById('history-list');

  if (!sessions.length) {
    list.innerHTML = '<li class="muted">No sessions yet.</li>';
    return;
  }

  list.innerHTML = '';
  sessions.forEach((s) => {
    const li = document.createElement('li');
    const date = new Date(s.started_at * 1000).toLocaleString();
    const total = s.focus_seconds + s.distraction_seconds + s.disruption_seconds;
    const focusPct = total > 0 ? Math.round((s.focus_seconds / total) * 100) : 0;
    li.innerHTML = `<span>${s.child_name} — ${date}</span><span>${focusPct}% focused</span>`;
    list.appendChild(li);
  });
}

// ---------- Voice clip recording (onboarding) ----------

function updateScripts() {
  const name = document.getElementById('child-name').value.trim() || 'Alex';
  document.querySelectorAll('.clip-script').forEach((container) => {
    const situation = container.dataset.situation;
    const textarea = container.querySelector('.script-text');
    if (textarea.dataset.userEdited !== 'true') {
      textarea.value = BANGLA_SCRIPTS[situation](name);
    }
  });
}

on('child-name', 'input', updateScripts);
document.querySelectorAll('.script-text').forEach((textarea) => {
  textarea.addEventListener('input', () => {
    textarea.dataset.userEdited = 'true';
  });
});

document.querySelectorAll('.record-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    const situation = btn.dataset.situation;
    if (activeRecorders[situation]) {
      stopRecording(situation, btn);
    } else {
      startRecording(situation, btn);
    }
  });
});

async function startRecording(situation, button) {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    console.error('getUserMedia (mic) failed:', err.name, err.message);
    alert(describeMediaError(err) + '\n\n(Technical detail: ' + err.name + ')');
    return;
  }

  const recorder = new MediaRecorder(stream);
  const chunks = [];
  recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
  recorder.onstop = () => {
    const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' });
    const container = button.closest('.clip-script');
    const preview = container.querySelector('.preview-audio');
    preview.src = URL.createObjectURL(blob);
    preview.hidden = false;

    const saveBtn = container.querySelector('.save-clip-btn');
    const againBtn = container.querySelector('.record-again-btn');
    saveBtn.hidden = false;
    againBtn.hidden = false;
    saveBtn._pendingBlob = blob;

    stream.getTracks().forEach((t) => t.stop());
  };

  recorder.start();
  activeRecorders[situation] = recorder;
  button.textContent = '■ Stop';
  button.classList.add('recording');
}

function stopRecording(situation, button) {
  const recorder = activeRecorders[situation];
  if (recorder && recorder.state !== 'inactive') recorder.stop();
  delete activeRecorders[situation];
  button.textContent = '● Record';
  button.classList.remove('recording');
}

function resetTakeUI(container) {
  const preview = container.querySelector('.preview-audio');
  const saveBtn = container.querySelector('.save-clip-btn');
  const againBtn = container.querySelector('.record-again-btn');
  preview.hidden = true;
  preview.removeAttribute('src');
  saveBtn.hidden = true;
  saveBtn._pendingBlob = null;
  againBtn.hidden = true;
}

document.querySelectorAll('.record-again-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    const situation = btn.dataset.situation;
    const container = btn.closest('.clip-script');
    resetTakeUI(container);
    const recordBtn = container.querySelector('.record-btn');
    startRecording(situation, recordBtn);
  });
});

document.querySelectorAll('.save-clip-btn').forEach((btn) => {
  btn.addEventListener('click', async () => {
    const situation = btn.dataset.situation;
    const blob = btn._pendingBlob;
    if (!blob) return;

    const formData = new FormData();
    formData.append('situation', situation);
    formData.append('audio', blob, `parent_${Date.now()}.webm`);

    const res = await fetch('/api/upload-clip', { method: 'POST', body: formData });
    const data = await res.json();

    if (data.ok) {
      resetTakeUI(btn.closest('.clip-script'));
      refreshClipCounts();
    } else {
      alert('Could not save the clip: ' + (data.error || 'unknown error'));
    }
  });
});

async function refreshClipCounts() {
  const res = await fetch('/api/clips');
  const data = await res.json();

  document.querySelectorAll('.saved-clips-list').forEach((list) => {
    const situation = list.dataset.situation;
    const filenames = data[situation] || [];
    list.innerHTML = '';

    if (!filenames.length) {
      const li = document.createElement('li');
      li.className = 'muted';
      li.textContent = 'No clips saved yet.';
      list.appendChild(li);
      return;
    }

    filenames.forEach((filename) => {
      const li = document.createElement('li');

      const audio = document.createElement('audio');
      audio.controls = true;
      audio.src = `/static/audio/${situation}/${filename}`;
      li.appendChild(audio);

      const removeBtn = document.createElement('button');
      removeBtn.className = 'remove-clip-btn';
      removeBtn.textContent = 'Remove';
      removeBtn.addEventListener('click', async () => {
        const res2 = await fetch(`/api/clips/${situation}/${encodeURIComponent(filename)}`, {
          method: 'DELETE',
        });
        const data2 = await res2.json();
        if (data2.ok) {
          refreshClipCounts();
        } else {
          alert('Could not remove the clip: ' + (data2.error || 'unknown error'));
        }
      });
      li.appendChild(removeBtn);

      list.appendChild(li);
    });
  });
}

updateScripts();
refreshClipCounts();
loadHistory();
refreshCalibrationStatus();

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/static/sw.js').catch((err) => {
      console.warn('Service worker registration failed (non-critical):', err);
    });
  });
}

// ---------- Calibration ----------

const CALIBRATION_PHASES = [
  { key: 'reading', label: 'Look down at your book/notebook and stay still', duration: 5000 },
  { key: 'writing', label: 'Now write something (keep your hand moving)', duration: 5000 },
  { key: 'looking_away', label: 'Now look away from your desk', duration: 5000 },
];
const CALIBRATION_SAMPLE_INTERVAL_MS = 1000;
let calibrationStream = null;

async function refreshCalibrationStatus() {
  const res = await fetch('/api/calibrate/status');
  const data = await res.json();
  const statusText = document.getElementById('calibration-status-text');
  const resetBtn = document.getElementById('calibrate-reset-btn');
  if (!statusText) return;
  if (data.ok && data.profile) {
    statusText.textContent = `Calibrated — writing motion threshold: ${data.profile.writing_motion_threshold} ` +
      `(reading baseline: ${data.profile.reading_motion_baseline}, writing baseline: ${data.profile.writing_motion_baseline}). ` +
      `This applies automatically to your next session.`;
    if (resetBtn) resetBtn.hidden = false;
  } else {
    statusText.textContent = 'Not calibrated yet — sessions use generic default thresholds until you run this.';
    if (resetBtn) resetBtn.hidden = true;
  }
}

on('calibrate-reset-btn', 'click', async () => {
  await fetch('/api/calibrate/reset', { method: 'POST' });
  refreshCalibrationStatus();
});

on('calibrate-start-btn', 'click', async () => {
  const startBtn = document.getElementById('calibrate-start-btn');
  const statusText = document.getElementById('calibration-status-text');
  const liveBox = document.getElementById('calibration-live');
  const video = document.getElementById('calibration-video');
  const canvas = document.getElementById('calibration-canvas');
  const phaseLabel = document.getElementById('calibration-phase-label');
  const progressFill = document.getElementById('calibration-progress-fill');

  try {
    calibrationStream = await navigator.mediaDevices.getUserMedia({ video: true });
  } catch (err) {
    statusText.textContent = describeMediaError(err);
    return;
  }

  video.srcObject = calibrationStream;
  startBtn.hidden = true;
  liveBox.hidden = false;
  statusText.textContent = '';

  await fetch('/api/calibrate/start', { method: 'POST' });

  for (const phase of CALIBRATION_PHASES) {
    phaseLabel.textContent = phase.label;
    const phaseStart = Date.now();

    const sampleTimer = setInterval(() => {
      const ctx = canvas.getContext('2d');
      canvas.width = video.videoWidth || 320;
      canvas.height = video.videoHeight || 240;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const frameData = canvas.toDataURL('image/jpeg', 0.6);
      fetch('/api/calibrate/sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phase: phase.key, frame: frameData }),
      }).catch(() => {});
    }, CALIBRATION_SAMPLE_INTERVAL_MS);

    await new Promise((resolve) => {
      const progressTimer = setInterval(() => {
        const pct = Math.min(100, ((Date.now() - phaseStart) / phase.duration) * 100);
        progressFill.style.width = pct + '%';
        if (pct >= 100) {
          clearInterval(progressTimer);
          clearInterval(sampleTimer);
          resolve();
        }
      }, 100);
    });
  }

  calibrationStream.getTracks().forEach((t) => t.stop());
  liveBox.hidden = true;
  startBtn.hidden = false;
  progressFill.style.width = '0%';

  statusText.textContent = 'Finishing up…';
  const res = await fetch('/api/calibrate/finish', { method: 'POST' });
  const data = await res.json();
  if (data.ok) {
    statusText.textContent = `Calibrated! Writing motion threshold set to ${data.profile.writing_motion_threshold} ` +
      `for your setup — this applies to your next session.`;
    document.getElementById('calibrate-reset-btn').hidden = false;
  } else {
    statusText.textContent = (data.error || 'Calibration did not produce a clear result.') + ' You can try again.';
  }
});

// ---------- Training mode ----------

document.querySelectorAll('.train-label-btn').forEach((btn) => {
  btn.addEventListener('click', async () => {
    const label = btn.dataset.label;
    const res = await fetch('/api/train/label', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label }),
    });
    const data = await res.json();
    if (data.ok) {
      updateTrainingCounts(data.counts);
    } else {
      document.getElementById('train-status-message').textContent =
        'Could not log that label: ' + (data.error || 'unknown error');
    }
  });
});

on('train-fit-btn', 'click', async () => {
  const msg = document.getElementById('train-status-message');
  msg.textContent = 'Training…';
  const res = await fetch('/api/train/fit', { method: 'POST' });
  const data = await res.json();
  if (data.ok) {
    msg.textContent = `Trained on ${data.examples} examples. ` +
      `Training-set accuracy: ${(data.train_accuracy * 100).toFixed(1)}% ` +
      `(not a held-out score — end this session and start a new one to use the updated model).`;
  } else {
    msg.textContent = data.error || 'Training failed.';
    if (data.counts) updateTrainingCounts(data.counts);
  }
});

on('train-reset-btn', 'click', async () => {
  if (!confirm('This deletes all logged training examples and the trained model. Continue?')) return;
  await fetch('/api/train/reset', { method: 'POST' });
  refreshTrainingStatus();
  document.getElementById('train-status-message').textContent = 'Training data cleared.';
});

function updateTrainingCounts(counts) {
  document.querySelectorAll('.training-count').forEach((el) => {
    const label = el.dataset.label;
    el.textContent = counts[label] || 0;
  });
}

async function refreshTrainingStatus() {
  const res = await fetch('/api/train/status');
  const data = await res.json();
  if (data.ok) updateTrainingCounts(data.counts);
}

// ---------- Bulk video training upload ----------

on('bulk-upload-btn', 'click', async () => {
  const label = document.getElementById('bulk-upload-label').value;
  const fileInput = document.getElementById('bulk-upload-file');
  const statusEl = document.getElementById('bulk-upload-status');
  const file = fileInput.files[0];

  if (!file) {
    statusEl.textContent = 'Choose a video file first.';
    return;
  }

  statusEl.textContent = 'Uploading and extracting examples…';
  const formData = new FormData();
  formData.append('label', label);
  formData.append('video', file);

  try {
    const res = await fetch('/api/train/upload-video', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.ok) {
      statusEl.textContent = `Added ${data.frames_added} examples labeled ${data.label}.`;
      updateTrainingCounts(data.counts);
      fileInput.value = '';
    } else {
      statusEl.textContent = data.error || 'Upload failed.';
    }
  } catch (err) {
    statusEl.textContent = 'Upload failed: ' + err.message;
  }
});

on('drive-link-btn', 'click', async () => {
  const label = document.getElementById('bulk-upload-label').value;
  const driveUrl = document.getElementById('drive-link-input').value.trim();
  const statusEl = document.getElementById('bulk-upload-status');

  if (!driveUrl) {
    statusEl.textContent = 'Paste a Google Drive link first.';
    return;
  }

  statusEl.textContent = 'Fetching from Drive and extracting examples… this can take a moment.';
  try {
    const res = await fetch('/api/train/upload-drive-link', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label, drive_url: driveUrl }),
    });
    const data = await res.json();
    if (data.ok) {
      statusEl.textContent = `Added ${data.frames_added} examples labeled ${data.label}.`;
      updateTrainingCounts(data.counts);
      document.getElementById('drive-link-input').value = '';
    } else {
      statusEl.textContent = data.error || 'Could not fetch that link.';
    }
  } catch (err) {
    statusEl.textContent = 'Failed: ' + err.message;
  }
});
