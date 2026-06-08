/**
 * ISONOME SIM CLIENT
 *
 * Connects to Isaac Sim or mock bridge via WebRTC (preferred) or WebSocket+MJPEG (fallback).
 * No animations. No gradients. Real data only.
 */

const WS_URL = "ws://localhost:8765";
const MJPEG_URL = "http://localhost:8766";
const STATE_POLL_MS = 100;

const els = {
  uploadInput: document.getElementById("upload-input"),
  uploadBtn: document.getElementById("upload-btn"),
  uploadStatus: document.getElementById("upload-status"),
  streamVideo: document.getElementById("stream-video"),
  streamImg: document.getElementById("stream-img"),
  streamPlaceholder: document.getElementById("stream-placeholder"),
  streamStatus: document.getElementById("stream-status"),
  statsOverlay: document.getElementById("stats-overlay"),
  statRes: document.getElementById("stat-res"),
  statFps: document.getElementById("stat-fps"),
  statDc: document.getElementById("stat-dc"),
  protoStatus: document.getElementById("proto-status"),
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
let protocol = "—";        // 'webrtc' | 'mjpeg' | 'ws' | '—'
let webrtcConnecting = false;

// ── WebRTC ───────────────────────────────────────────────────────
let pc = null;
let dc = null;
let webrtcActive = false;
let statsTimer = null;

function setStreamStatus(text) {
  if (els.streamStatus) els.streamStatus.textContent = text;
}

function updateProtocolStatus() {
  let label = "PROTO: " + protocol.toUpperCase();
  let cls = "indicator";
  if (protocol === "webrtc") cls += " ok";
  else if (protocol === "mjpeg") cls += " warn";
  else if (protocol === "—") cls += " err";
  if (webrtcConnecting) cls += " pulse";
  els.protoStatus.textContent = label;
  els.protoStatus.className = cls;
}

async function startWebRTC() {
  if (!window.RTCPeerConnection) {
    setStreamStatus("WebRTC not supported — falling back");
    startMjpegStream();
    return;
  }

  webrtcConnecting = true;
  updateProtocolStatus();
  setStreamStatus("Negotiating WebRTC…");

  pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });

  pc.ontrack = (event) => {
    const stream = event.streams[0];
    if (els.streamVideo) {
      els.streamVideo.srcObject = stream;
      els.streamVideo.style.display = "block";
    }
    if (els.streamImg) {
      els.streamImg.style.display = "none";
    }
    els.streamPlaceholder.style.display = "none";
    webrtcActive = true;
    webrtcConnecting = false;
    protocol = "webrtc";
    mjpegActive = false;
    updateProtocolStatus();
    setStreamStatus("");
    startStatsPolling();
  };

  pc.onconnectionstatechange = () => {
    const state = pc.connectionState;
    if (state === "connecting") {
      setStreamStatus("WebRTC connecting…");
    }
    if (state === "failed" || state === "closed") {
      webrtcActive = false;
      webrtcConnecting = false;
      stopStatsPolling();
      if (els.statsOverlay) els.statsOverlay.classList.remove("visible");
      if (dc) { dc.close(); dc = null; }
      protocol = "mjpeg";
      updateProtocolStatus();
      setStreamStatus("WebRTC failed — falling back to MJPEG");
      startMjpegStream();
    }
  };

  // Data channel for low-latency commands
  dc = pc.createDataChannel("commands", { ordered: true });
  dc.onopen = () => {
    connected = true;
    webrtcActive = true;
    updateConnectionStatus(true);
    updateProtocolStatus();
  };
  dc.onclose = () => {
    webrtcActive = false;
  };
  dc.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleMessage(msg);
    } catch (e) {
      console.warn("Invalid DC message");
    }
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  // Allow ICE candidates to gather briefly
  await new Promise((r) => setTimeout(r, 400));

  // Send offer via WebSocket (signaling)
  sendCommand({
    action: "webrtc_offer",
    sdp: pc.localDescription.sdp,
    type: pc.localDescription.type,
  });
}

// ── Stats polling ────────────────────────────────────────────────
function startStatsPolling() {
  if (statsTimer) return;
  statsTimer = setInterval(async () => {
    if (!pc || !webrtcActive) return;
    try {
      const stats = await pc.getStats();
      let fps = "—", w = 0, h = 0;
      let dcState = dc ? dc.readyState : "—";
      stats.forEach((report) => {
        if (report.type === "inbound-rtp" && report.kind === "video") {
          if (report.framesPerSecond != null) fps = report.framesPerSecond.toFixed(0);
        }
        if (report.type === "track" && report.kind === "video") {
          if (report.frameWidth) w = report.frameWidth;
          if (report.frameHeight) h = report.frameHeight;
        }
      });
      if (els.statRes) els.statRes.textContent = w && h ? `${w}×${h}` : "—";
      if (els.statFps) els.statFps.textContent = fps !== "—" ? fps + " FPS" : "—";
      if (els.statDc) els.statDc.textContent = dcState.toUpperCase();
      if (els.statsOverlay) els.statsOverlay.classList.add("visible");
    } catch (e) {
      // ignore
    }
  }, 1000);
}

function stopStatsPolling() {
  if (statsTimer) {
    clearInterval(statsTimer);
    statsTimer = null;
  }
}

// ── WebSocket ────────────────────────────────────────────────────
function connectWebSocket() {
  if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
    return;
  }
  setStreamStatus("Connecting to bridge…");
  try {
    ws = new WebSocket(WS_URL);
    ws.onopen = () => {
      connected = true;
      protocol = "ws";
      updateConnectionStatus(true);
      updateProtocolStatus();
      startStatePolling();
      startWebRTC();
    };
    ws.onclose = () => {
      connected = false;
      webrtcActive = false;
      webrtcConnecting = false;
      protocol = "—";
      updateConnectionStatus(false);
      updateProtocolStatus();
      stopStatePolling();
      stopStatsPolling();
      setStreamStatus("Disconnected — retrying…");
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
    setStreamStatus("Connection failed — retrying…");
    setTimeout(connectWebSocket, 2000);
  }
}

function sendCommand(cmd) {
  if (webrtcActive && dc && dc.readyState === "open") {
    dc.send(JSON.stringify(cmd));
  } else if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(cmd));
  }
}

function handleMessage(msg) {
  if (msg.error) {
    // If WebRTC offer was rejected, fall back gracefully
    if (msg.error.includes("WebRTC not available")) {
      webrtcConnecting = false;
      protocol = "mjpeg";
      updateProtocolStatus();
      setStreamStatus("WebRTC unavailable — using MJPEG");
      startMjpegStream();
      return;
    }
    els.uploadStatus.textContent = "ERR: " + msg.error;
    return;
  }

  // WebRTC signaling answer
  if (msg.ok && msg.webrtc_answer) {
    const ans = msg.webrtc_answer;
    pc.setRemoteDescription(new RTCSessionDescription(ans)).catch((err) => {
      console.warn("Failed to set remote desc:", err);
      webrtcActive = false;
      webrtcConnecting = false;
      protocol = "mjpeg";
      updateProtocolStatus();
      setStreamStatus("WebRTC handshake failed — falling back");
      startMjpegStream();
    });
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

// ── MJPEG Stream (fallback) ──────────────────────────────────────
function startMjpegStream() {
  if (mjpegActive || webrtcActive) return;
  if (els.streamVideo) {
    els.streamVideo.style.display = "none";
    els.streamVideo.srcObject = null;
  }
  if (els.streamImg) {
    els.streamImg.style.display = "block";
    els.streamImg.src = MJPEG_URL;
  }
  els.streamPlaceholder.style.display = "none";
  mjpegActive = true;
  protocol = "mjpeg";
  updateProtocolStatus();
  setStreamStatus("");
}

function stopMjpegStream() {
  if (els.streamImg) {
    els.streamImg.src = "";
    els.streamImg.style.display = "none";
  }
  if (!webrtcActive) {
    els.streamPlaceholder.style.display = "flex";
  }
  mjpegActive = false;
}

// ── Upload ───────────────────────────────────────────────────────
async function uploadFile() {
  const file = els.uploadInput.files[0];
  if (!file) {
    els.uploadStatus.textContent = "ERR: NO FILE";
    return;
  }

  els.uploadStatus.textContent = "UPLOADING…";
  const form = new FormData();
  form.append("file", file);

  try {
    const resp = await fetch("/api/upload-urdf", { method: "POST", body: form });
    const data = await resp.json();
    if (data.error) {
      els.uploadStatus.textContent = "ERR: " + data.error;
      return;
    }
    els.uploadStatus.textContent = "PARSING MODEL…";
    sendCommand({ action: "load_urdf", path: data.path });

    // Smooth transition: show loading, then activate stream
    setStreamStatus("Loading simulation…");
    setTimeout(() => {
      if (!webrtcActive && !mjpegActive) {
        startMjpegStream();
      }
      setStreamStatus("");
    }, 600);
  } catch (e) {
    els.uploadStatus.textContent = "ERR: " + e.message;
  }
}

// ── Controls ─────────────────────────────────────────────────────
function flashButton(btn) {
  if (!btn) return;
  btn.classList.add("flash");
  setTimeout(() => btn.classList.remove("flash"), 120);
}

function setupControls() {
  els.btnPlay.addEventListener("click", () => {
    flashButton(els.btnPlay);
    sendCommand({ action: "play" });
  });
  els.btnPause.addEventListener("click", () => {
    flashButton(els.btnPause);
    sendCommand({ action: "pause" });
  });
  els.btnStep.addEventListener("click", () => {
    flashButton(els.btnStep);
    sendCommand({ action: "step", steps: 1 });
  });
  els.btnReset.addEventListener("click", () => {
    flashButton(els.btnReset);
    sendCommand({ action: "reset" });
  });
  els.uploadBtn.addEventListener("click", () => {
    flashButton(els.uploadBtn);
    uploadFile();
  });
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
}

init();
