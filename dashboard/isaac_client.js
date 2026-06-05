/**
 * ISONOME SIM CLIENT
 *
 * Connects to Isaac Sim or mock bridge via WebSocket.
 * No animations. No gradients. Real data only.
 */

const WS_URL = "ws://localhost:8765";
const MJPEG_URL = "http://localhost:8766";
const STATE_POLL_MS = 100;

const els = {
  uploadInput: document.getElementById("upload-input"),
  uploadBtn: document.getElementById("upload-btn"),
  uploadStatus: document.getElementById("upload-status"),
  streamImg: document.getElementById("stream-img"),
  streamPlaceholder: document.getElementById("stream-placeholder"),
  btnPlay: document.getElementById("btn-play"),
  btnPause: document.getElementById("btn-pause"),
  btnStep: document.getElementById("btn-step"),
  btnReset: document.getElementById("btn-reset"),
  jointList: document.getElementById("joint-list"),
  jointCount: document.getElementById("joint-count"),
  connStatus: document.getElementById("conn-status"),
  simStatus: document.getElementById("sim-status"),
  dofDisplay: document.getElementById("dof-display"),
  timeDisplay: document.getElementById("time-display"),
};

let ws = null;
let statePollTimer = null;
let joints = [];
let simState = { playing: false, timestamp: 0 };
let connected = false;
let mjpegActive = false;

// ── WebSocket ────────────────────────────────────────────────────
function connectWebSocket() {
  if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
    return;
  }
  try {
    ws = new WebSocket(WS_URL);
    ws.onopen = () => {
      connected = true;
      updateConnectionStatus(true);
      startStatePolling();
    };
    ws.onclose = () => {
      connected = false;
      updateConnectionStatus(false);
      stopStatePolling();
      setTimeout(connectWebSocket, 2000);
    };
    ws.onerror = () => {};
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleMessage(msg);
      } catch (e) {
        console.warn("Invalid WS message");
      }
    };
  } catch (e) {
    setTimeout(connectWebSocket, 2000);
  }
}

function sendCommand(cmd) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(cmd));
  }
}

function handleMessage(msg) {
  if (msg.error) {
    els.uploadStatus.textContent = "ERR: " + msg.error;
    return;
  }
  if (msg.ok && msg.state) {
    updateState(msg.state);
  }
  if (msg.ok && msg.joints) {
    joints = msg.joints;
    els.dofDisplay.textContent = "DOF: " + (msg.dof_count || joints.length);
    renderJointList(joints);
    els.uploadStatus.textContent = "LOADED: " + joints.length + " JOINTS";
  }
}

// ── State polling ────────────────────────────────────────────────
function startStatePolling() {
  if (statePollTimer) return;
  statePollTimer = setInterval(() => {
    sendCommand({ action: "get_state" });
  }, STATE_POLL_MS);
}

function stopStatePolling() {
  if (statePollTimer) {
    clearInterval(statePollTimer);
    statePollTimer = null;
  }
}

function updateState(state) {
  simState = state;
  els.simStatus.textContent = state.playing ? "SIM RUNNING" : "SIM STOPPED";
  els.simStatus.className = "indicator " + (state.playing ? "ok" : "warn");
  els.timeDisplay.textContent = "T: " + (state.timestamp ? state.timestamp.toFixed(3) + "s" : "—");

  if (state.joints) {
    updateJointValues(state.joints);
  }
}

// ── Joint UI ─────────────────────────────────────────────────────
function renderJointList(jointNames) {
  els.jointList.innerHTML = "";
  els.jointCount.textContent = jointNames.length;
  jointNames.forEach((name, idx) => {
    const row = document.createElement("div");
    row.className = "joint-row";
    row.id = "joint-row-" + idx;
    row.innerHTML = `
      <div class="joint-name" title="${name}">${name}</div>
      <div class="joint-bar">
        <div class="joint-bar-fill" id="joint-bar-${idx}"></div>
        <div class="joint-bar-zero"></div>
      </div>
      <div class="joint-pos" id="joint-pos-${idx}">0.0000</div>
      <div class="joint-vel" id="joint-vel-${idx}">0.00</div>
    `;
    els.jointList.appendChild(row);
  });
}

function updateJointValues(jointsData) {
  jointsData.forEach((j, idx) => {
    const posEl = document.getElementById("joint-pos-" + idx);
    const velEl = document.getElementById("joint-vel-" + idx);
    const barEl = document.getElementById("joint-bar-" + idx);
    if (!posEl || !velEl || !barEl) return;

    const pos = j.position || 0;
    const vel = j.velocity || 0;

    posEl.textContent = pos.toFixed(4);
    velEl.textContent = vel.toFixed(2);

    const vabs = Math.abs(vel);
    if (vabs > 1.0) velEl.style.color = "var(--accent-red)";
    else if (vabs > 0.3) velEl.style.color = "var(--accent-amber)";
    else velEl.style.color = "var(--text-dim)";

    // Map -PI..PI to 0%..100% for display
    const clamped = Math.max(-Math.PI, Math.min(Math.PI, pos));
    const pct = ((clamped + Math.PI) / (2 * Math.PI)) * 100;
    barEl.style.left = "0";
    barEl.style.width = pct.toFixed(1) + "%";
  });
}

// ── MJPEG Stream ─────────────────────────────────────────────────
function startMjpegStream() {
  if (mjpegActive) return;
  els.streamPlaceholder.style.display = "none";
  els.streamImg.style.display = "block";
  els.streamImg.src = MJPEG_URL;
  mjpegActive = true;
}

function stopMjpegStream() {
  els.streamImg.src = "";
  els.streamImg.style.display = "none";
  els.streamPlaceholder.style.display = "flex";
  mjpegActive = false;
}

// ── Upload ───────────────────────────────────────────────────────
async function uploadFile() {
  const file = els.uploadInput.files[0];
  if (!file) {
    els.uploadStatus.textContent = "ERR: NO FILE";
    return;
  }

  els.uploadStatus.textContent = "UPLOADING...";
  const form = new FormData();
  form.append("file", file);

  try {
    const resp = await fetch("/api/upload-urdf", { method: "POST", body: form });
    const data = await resp.json();
    if (data.error) {
      els.uploadStatus.textContent = "ERR: " + data.error;
      return;
    }
    sendCommand({ action: "load_urdf", path: data.path });
    setTimeout(startMjpegStream, 500);
  } catch (e) {
    els.uploadStatus.textContent = "ERR: " + e.message;
  }
}

// ── Controls ─────────────────────────────────────────────────────
function setupControls() {
  els.btnPlay.addEventListener("click", () => sendCommand({ action: "play" }));
  els.btnPause.addEventListener("click", () => sendCommand({ action: "pause" }));
  els.btnStep.addEventListener("click", () => sendCommand({ action: "step", steps: 1 }));
  els.btnReset.addEventListener("click", () => sendCommand({ action: "reset" }));
  els.uploadBtn.addEventListener("click", uploadFile);
  els.uploadInput.addEventListener("change", () => {
    const f = els.uploadInput.files[0];
    if (f) els.uploadStatus.textContent = "READY: " + f.name;
  });
}

function updateConnectionStatus(isConnected) {
  els.connStatus.textContent = isConnected ? "BRIDGE ONLINE" : "BRIDGE OFFLINE";
  els.connStatus.className = "indicator " + (isConnected ? "ok" : "err");
}

// ── Init ─────────────────────────────────────────────────────────
function init() {
  setupControls();
  connectWebSocket();
  startMjpegStream();
}

init();
