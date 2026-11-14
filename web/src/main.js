import './style.css';

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import {
  createVRMAnimationClip,
  VRMAnimationLoaderPlugin,
} from '@pixiv/three-vrm-animation';

import { MocapRetargeter } from './mocap.js';
import { WINDOW_TYPES, windowForCutoff } from './filters.js';

const canvas = document.getElementById('view');
const loaderEl = document.getElementById('loader');
const loaderText = document.getElementById('loader-text');
const toastEl = document.getElementById('toast');
const feedImg = document.getElementById('feed');
const feedMeta = document.getElementById('feed-meta');
const statusPill = document.getElementById('status-pill');
const statusText = document.getElementById('status-text');
const sourceSelect = document.getElementById('source-select');
const sourceMeta = document.getElementById('source-meta');
const liveStats = document.getElementById('live-stats');
const followRoot = document.getElementById('follow-root');
const footLockInput = document.getElementById('foot-lock');
const filterType = document.getElementById('filter-type');
const filterStrength = document.getElementById('filter-strength');
const filterDeadband = document.getElementById('filter-deadband');
const filterStrengthMeta = document.getElementById('filter-strength-meta');
const filterDeadbandMeta = document.getElementById('filter-deadband-meta');
const avatarMeta = document.getElementById('avatar-meta');
const btnLive = document.getElementById('btn-live');
const btnUpload = document.getElementById('btn-upload');
const fileInput = document.getElementById('file-input');
const btnAvatar = document.getElementById('btn-avatar');
const avatarInput = document.getElementById('avatar-input');
const chromaBox = document.getElementById('chroma');
const chromaColor = document.getElementById('chroma-color');
const keyButtons = [...document.querySelectorAll('.key')];
const obsMeta = document.getElementById('obs-meta');
const btnObs = document.getElementById('btn-obs');

/* --------------------------------------------------------------- scene */
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  powerPreference: 'high-performance',
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07080c);
scene.fog = new THREE.Fog(0x07080c, 5.5, 14);

const FRAME = { height: 2.0, width: 1.3, centerY: 0.92 };
const DEFAULT_DIR = new THREE.Vector3(0, 0.44, 3.05).normalize();

const camera = new THREE.PerspectiveCamera(
  32,
  window.innerWidth / window.innerHeight,
  0.1,
  60,
);
camera.position.copy(DEFAULT_DIR).multiplyScalar(3.1).add(
  new THREE.Vector3(0, FRAME.centerY, 0),
);

const controls = new OrbitControls(camera, canvas);
controls.target.set(0, FRAME.centerY, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.screenSpacePanning = true;
controls.mouseButtons = {
  LEFT: THREE.MOUSE.ROTATE,
  MIDDLE: THREE.MOUSE.DOLLY,
  RIGHT: THREE.MOUSE.PAN,
};
controls.minDistance = 1.2;
controls.maxDistance = 12;
controls.minPolarAngle = 0.55;
controls.maxPolarAngle = Math.PI / 2 - 0.02;
controls.update();

canvas.addEventListener('contextmenu', (event) => event.preventDefault());

let userMovedCamera = false;
controls.addEventListener('start', () => {
  userMovedCamera = true;
});

/** Keep the whole avatar in shot for any viewport shape. */
function fitCamera(force = false) {
  const ok = Number.isFinite(camera.position.x) &&
    Number.isFinite(controls.target.x);
  if (!ok) {
    camera.position.copy(DEFAULT_DIR).multiplyScalar(3).add(
      new THREE.Vector3(0, FRAME.centerY, 0),
    );
    controls.target.set(0, FRAME.centerY, 0);
    force = true;
  }
  if (userMovedCamera && !force) return;

  const width = Math.max(1, window.innerWidth);
  const height = Math.max(1, window.innerHeight);
  const aspect = Math.min(8, Math.max(0.15, width / height));
  camera.aspect = aspect;

  const halfFov = THREE.MathUtils.degToRad(camera.fov) / 2;
  const distance = Math.min(
    20,
    Math.max(
      FRAME.height / 2 / Math.tan(halfFov),
      FRAME.width / 2 / (Math.tan(halfFov) * aspect),
    ),
  );

  const direction = DEFAULT_DIR.clone();
  if (!force) {
    direction.copy(camera.position).sub(controls.target);
    if (direction.lengthSq() < 1e-6) direction.copy(DEFAULT_DIR);
    else direction.normalize();
  }
  controls.target.set(0, FRAME.centerY, 0);
  camera.position.copy(controls.target).addScaledVector(direction, distance);
  camera.updateProjectionMatrix();
  controls.update();
  controls.saveState();
}

fitCamera(true);

scene.add(new THREE.HemisphereLight(0x9dc8ff, 0x0a0c12, 1.15));

const keyLight = new THREE.DirectionalLight(0xffffff, 2.6);
keyLight.position.set(2.4, 4.2, 3.2);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.camera.left = -2.5;
keyLight.shadow.camera.right = 2.5;
keyLight.shadow.camera.top = 4;
keyLight.shadow.camera.bottom = -1;
keyLight.shadow.camera.near = 0.5;
keyLight.shadow.camera.far = 12;
keyLight.shadow.bias = -0.0006;
scene.add(keyLight);

const rimLight = new THREE.DirectionalLight(0x62f2d2, 1.7);
rimLight.position.set(-3.2, 2.4, -3.4);
scene.add(rimLight);

const warmLight = new THREE.PointLight(0xff7aa8, 9, 7, 2);
warmLight.position.set(1.7, 1.1, 1.4);
scene.add(warmLight);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(3.8, 96),
  new THREE.MeshStandardMaterial({
    color: 0x0a0d13,
    roughness: 0.92,
    metalness: 0.15,
  }),
);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

const grid = new THREE.GridHelper(9, 36, 0x1c2433, 0x10151e);
grid.position.y = 0.004;
grid.material.transparent = true;
grid.material.opacity = 0.55;
scene.add(grid);

const ring = new THREE.Mesh(
  new THREE.RingGeometry(1.0, 1.035, 96),
  new THREE.MeshBasicMaterial({
    color: 0x62f2d2,
    transparent: true,
    opacity: 0.22,
    side: THREE.DoubleSide,
  }),
);
ring.rotation.x = -Math.PI / 2;
ring.position.y = 0.006;
scene.add(ring);

function makeBlobShadow() {
  const size = 256;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  const grad = ctx.createRadialGradient(
    size / 2, size / 2, 0,
    size / 2, size / 2, size / 2,
  );
  grad.addColorStop(0, 'rgba(0,0,0,0.55)');
  grad.addColorStop(0.55, 'rgba(0,0,0,0.22)');
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(c);
  texture.colorSpace = THREE.SRGBColorSpace;
  return new THREE.Mesh(
    new THREE.PlaneGeometry(1.6, 1.6),
    new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      depthWrite: false,
    }),
  );
}

const blob = makeBlobShadow();
blob.rotation.x = -Math.PI / 2;
blob.position.y = 0.008;
scene.add(blob);

const avatarRoot = new THREE.Group();
scene.add(avatarRoot);

/* --------------------------------------------------------------- state */
const state = {
  vrm: null,
  mixer: null,
  retargeter: null,
  pose: null,
  poseAt: 0,
  tracking: false,
  wsOpen: false,
  wsLost: false,
  source: '',
  desired: '',
  // The source the user just picked, until the capture thread has actually
  // opened it. The server reports the old input for a moment after the switch
  // is accepted, and a poll in that window used to put the select back on the
  // camera - which looked exactly like "selecting a video switches me back".
  switching: '',
  switchAt: 0,
  sourceKnown: false,
  bootId: '',
  capture: { state: 'starting', source: '', label: '', fps: 0 },
  liveInfo: null,
  liveBusy: false,
  sources: [],
  feedAt: 0,
  avatarLabel: '',
  chroma: false,
  obs: false,
};

/* --------------------------------------------------------- browser camera */
// The container cannot reach the host webcam, so the page captures it and
// streams JPEG frames to the server. That link has to outlive everything a
// browser session outlives: a tab left open for hours, a phone that was
// locked, a camera unplugged, a container restarted underneath the page, a
// proxy that silently drops idle sockets. So the frames travel over one
// persistent websocket — with the plain POST as the fallback while it is
// reconnecting — and a watchdog rebuilds the whole pipeline when frames stop
// being accepted.
const CAMERA_FPS = 24;
const CAMERA_JPEG = 0.82;
const CAMERA_BACKOFF = [0, 400, 1000, 2000, 4000, 8000];   // ms, then capped
const CAMERA_STALE_MS = 4000;         // nothing accepted this long: rebuild
const CAMERA_BUSY_BYTES = 512 * 1024; // unsent backlog: skip this frame
const CAMERA_SOURCE = 'browser:camera';
const SOURCE_STORE = 'instanthmr.source';

const browserCam = {
  active: false,
  starting: false,
  stream: null,
  video: null,
  canvas: null,
  socket: null,
  socketOpen: false,
  sendTimer: 0,
  retryTimer: 0,
  watchdog: 0,
  run: 0,
  attempts: 0,
  sent: 0,
  lastOk: 0,
  lastError: '',
  asserted: 0,
  socketTried: false,
};

function closeCameraSocket() {
  const socket = browserCam.socket;
  browserCam.socket = null;
  browserCam.socketOpen = false;
  if (!socket) return;
  try {
    socket.close();
  } catch {
    /* already gone */
  }
}

function openCameraSocket() {
  if (!browserCam.active || browserCam.socket) return;
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  let socket;
  try {
    socket = new WebSocket(`${proto}://${window.location.host}/ws/camera`);
  } catch {
    scheduleCameraRetry('could not open the frame socket');
    return;
  }
  socket.binaryType = 'arraybuffer';
  browserCam.socket = socket;

  socket.addEventListener('open', () => {
    browserCam.socketOpen = true;
    browserCam.attempts = 0;
    browserCam.lastOk = performance.now();   // a fresh socket gets a grace
  });

  socket.addEventListener('message', (event) => {
    if (typeof event.data !== 'string') return;
    let message;
    try {
      message = JSON.parse(event.data);
    } catch {
      return;
    }
    if (message.type === 'ack' || message.type === 'ready' ||
        message.type === 'pong' || message.type === 'tick') {
      browserCam.lastOk = performance.now();
    } else if (message.type === 'error') {
      // the link is alive; the input is not. Take the camera back rather than
      // letting the page sit on a source the server has already left.
      browserCam.lastOk = performance.now();
      browserCam.lastError = message.error || 'frame rejected';
      // a frame that was already on the wire when the user picked another
      // input is not a reason to grab the camera back
      if (!state.switching) assertCameraSource();
    }
  });

  socket.addEventListener('close', () => {
    if (browserCam.socket !== socket) return;   // replaced on purpose
    browserCam.socket = null;
    browserCam.socketOpen = false;
    if (browserCam.active) scheduleCameraRetry('frame socket closed');
  });

  socket.addEventListener('error', () => {
    /* a close always follows; the retry is scheduled there */
  });
}

function scheduleCameraRetry(reason) {
  if (!browserCam.active || browserCam.retryTimer) return;
  browserCam.lastError = reason;
  const wait = CAMERA_BACKOFF[
    Math.min(browserCam.attempts, CAMERA_BACKOFF.length - 1)];
  browserCam.attempts += 1;
  browserCam.retryTimer = window.setTimeout(() => {
    browserCam.retryTimer = 0;
    if (!browserCam.active) return;
    openCameraSocket();
  }, wait);
}

/** Re-select the browser camera when the server is no longer on it. */
function assertCameraSource() {
  if (state.desired !== CAMERA_SOURCE || state.switching) return;
  const now = performance.now();
  if (now - browserCam.asserted < 3000) return;
  browserCam.asserted = now;
  fetch('/api/source', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source: CAMERA_SOURCE }),
  }).then((response) => {
    if (!response.ok) return;
    browserCam.attempts = 0;
    closeCameraSocket();
    openCameraSocket();
  }).catch(() => {
    /* the watchdog and the next frame keep trying */
  });
}

/**
 * Put the page back on the input the user had before the server restarted.
 *
 * A restarted container comes up on its default input; the page would follow
 * it and quietly lose the video or camera the user picked.
 */
async function reassertSource(source) {
  if (!source) return;
  state.switching = source;
  state.switchAt = performance.now();
  try {
    const response = await fetch('/api/source', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source }),
    });
    if (response.ok) return;
    state.switching = '';
    state.desired = state.source;   // it is gone: follow the server instead
    showToast(`Input unavailable: ${source}`, { error: true });
  } catch {
    /* offline: the next poll retries */
  }
}

function startCameraWatchdog() {
  if (browserCam.watchdog) window.clearInterval(browserCam.watchdog);
  browserCam.watchdog = window.setInterval(() => {
    if (!browserCam.active) return;
    if (performance.now() - browserCam.lastOk <= CAMERA_STALE_MS) {
      browserCam.socketTried = false;   // the transport is healthy again
      return;
    }
    if (browserCam.socketOpen && !browserCam.socketTried) {
      // the socket looks open but nothing is coming back: replace the cheap
      // half first, and only rebuild the device if that does not help
      browserCam.socketTried = true;
      browserCam.lastOk = performance.now();
      closeCameraSocket();
      openCameraSocket();
      return;
    }
    restartBrowserCamera('no frame accepted for 4 s');
  }, 1000);
}

/** Drop the device and the socket, then build both again from scratch. */
function restartBrowserCamera(reason) {
  if (!browserCam.active) return;
  browserCam.lastError = reason || 'restarting';
  browserCam.lastOk = performance.now();   // do not restart again at once
  browserCam.run += 1;                     // orphan the running send loop
  closeCameraSocket();
  if (browserCam.stream) {
    for (const track of browserCam.stream.getTracks()) {
      track.onended = null;
      track.stop();
    }
  }
  browserCam.stream = null;
  if (browserCam.video) browserCam.video.srcObject = null;
  browserCam.video = null;
  browserCam.attempts = 0;
  acquireCamera();
}

function startCameraLoop() {
  const run = browserCam.run += 1;
  const tick = async () => {
    if (!browserCam.active || browserCam.run !== run) return;
    try {
      await sendCameraFrame();
    } catch {
      /* a dropped frame never stops the stream */
    }
    if (browserCam.active && browserCam.run === run) {
      browserCam.sendTimer = window.setTimeout(tick, 1000 / CAMERA_FPS);
    }
  };
  tick();
}

async function sendCameraFrame() {
  const video = browserCam.video;
  if (!video || video.readyState < 2) return;
  if (video.paused) video.play().catch(() => {});
  const socket = browserCam.socket;
  const socketReady = !!(socket && browserCam.socketOpen &&
    socket.readyState === 1 &&
    socket.bufferedAmount < CAMERA_BUSY_BYTES);
  const width = video.videoWidth || 1280;
  const height = video.videoHeight || 720;
  const canvas = browserCam.canvas;
  if (!canvas) return;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, width, height);
  const blob = await new Promise((resolve) =>
    canvas.toBlob(resolve, 'image/jpeg', CAMERA_JPEG));
  if (!blob || !browserCam.active) return;
  if (socketReady) {
    const buffer = await blob.arrayBuffer();
    if (!browserCam.active || browserCam.socket !== socket) return;
    socket.send(buffer);
    browserCam.sent += 1;
    return;
  }
  await postCameraFrame(blob);
}

/** The fallback transport: one request per frame, bounded in time. */
async function postCameraFrame(blob) {
  const control = new AbortController();
  const abort = window.setTimeout(() => control.abort(), 2000);
  try {
    const response = await fetch('/api/browser-frame', {
      method: 'POST',
      headers: { 'Content-Type': 'image/jpeg' },
      body: blob,
      signal: control.signal,
      cache: 'no-store',
    });
    browserCam.lastOk = performance.now();   // the server answered
    if (response.ok) {
      browserCam.sent += 1;
    } else if (response.status === 409) {
      assertCameraSource();
    }
  } catch {
    /* server away or slower than 2 s: the watchdog rebuilds the link */
  } finally {
    window.clearTimeout(abort);
  }
}

async function stopBrowserCamera() {
  browserCam.active = false;
  browserCam.starting = false;
  browserCam.run += 1;
  for (const key of ['sendTimer', 'retryTimer']) {
    if (browserCam[key]) window.clearTimeout(browserCam[key]);
    browserCam[key] = 0;
  }
  if (browserCam.watchdog) window.clearInterval(browserCam.watchdog);
  browserCam.watchdog = 0;
  closeCameraSocket();
  if (browserCam.stream) {
    for (const track of browserCam.stream.getTracks()) {
      track.onended = null;
      track.stop();
    }
  }
  browserCam.stream = null;
  if (browserCam.video) browserCam.video.srcObject = null;
  browserCam.video = null;
  browserCam.canvas = null;
}

/** Ask for the device again, backing off while it is unavailable. */
async function acquireCamera() {
  if (!browserCam.active || state.desired !== CAMERA_SOURCE) return;
  browserCam.starting = true;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'user',
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
  } catch (err) {
    browserCam.starting = false;
    browserCam.lastError = (err && err.name) || String(err);
    if (browserCam.attempts === 0) {
      // say it once; a retry storm must not bury the page in toasts
      showToast(`Browser camera: ${err.message || err} - retrying`,
        { error: true });
    }
    const wait = Math.max(500, CAMERA_BACKOFF[
      Math.min(browserCam.attempts, CAMERA_BACKOFF.length - 1)]);
    browserCam.attempts += 1;
    if (browserCam.active && !browserCam.retryTimer) {
      browserCam.retryTimer = window.setTimeout(() => {
        browserCam.retryTimer = 0;
        acquireCamera();
      }, wait);
    }
    return;
  }
  browserCam.starting = false;
  if (!browserCam.active || state.desired !== CAMERA_SOURCE) {
    for (const track of stream.getTracks()) track.stop();
    return;
  }
  const video = document.createElement('video');
  video.autoplay = true;
  video.playsInline = true;
  video.muted = true;
  video.srcObject = stream;
  browserCam.stream = stream;
  browserCam.video = video;
  if (!browserCam.canvas) browserCam.canvas = document.createElement('canvas');
  for (const track of stream.getTracks()) {
    // unplugged, grabbed by another app, permission revoked, OS resumed from
    // sleep: the device is gone, so replace it instead of leaving a dead feed
    track.addEventListener('ended', () => {
      if (browserCam.active) restartBrowserCamera('camera device ended');
    });
  }
  try {
    await video.play();
  } catch {
    /* autoplay policy: the send loop nudges it back */
  }
  if (!browserCam.active) return;
  browserCam.attempts = 0;
  browserCam.lastOk = performance.now();
  openCameraSocket();
  startCameraLoop();
}

async function startBrowserCamera() {
  if (browserCam.active || state.desired !== CAMERA_SOURCE) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast('Browser camera not supported in this browser',
      { sticky: true, error: true });
    return;
  }
  browserCam.active = true;
  browserCam.attempts = 0;
  browserCam.asserted = 0;
  browserCam.lastOk = performance.now();
  startCameraWatchdog();
  await acquireCamera();
}

let blinkExpression = null;
let blinkTimer = 2;
let blinkPhase = 0;

/* ------------------------------------------------- green screen / OBS view */
// The background is a WebGL clear colour: the renderer converts it from the
// working space straight to the output space and writes it before any tone
// mapping, so the value picked in the swatch is the value a capture sees. That
// is what a chroma key needs. Everything that is not the character is switched
// off rather than painted over, so no floor edge or shadow gradient leaks into
// the key.
const CHROMA_KEY = new THREE.Color(0x00ff00);
const BASE_BG = scene.background;
const BASE_FOG = scene.fog;
const STAGE = [floor, grid, ring, blob];
const CHROMA_STORE = 'instanthmr.chroma';

/** Pick any key colour — the swatch, the panel line and the field all follow. */
function setChromaColor(hex) {
  const value = hex.toLowerCase();
  CHROMA_KEY.set(value);
  document.documentElement.style.setProperty('--chroma', value);
  chromaColor.value = value;
  obsMeta.textContent = `${value.toUpperCase()} · no floor, grid or shadow`;
  for (const key of keyButtons) {
    const on = key.dataset.color === value;
    key.classList.toggle('on', on);
    key.setAttribute('aria-pressed', on ? 'true' : 'false');
  }
  try {
    localStorage.setItem(CHROMA_STORE, value);
  } catch {
    /* private mode: the choice just does not survive a reload */
  }
}

function setChroma(on) {
  state.chroma = on;
  scene.background = on ? CHROMA_KEY : BASE_BG;
  scene.fog = on ? null : BASE_FOG;
  for (const part of STAGE) part.visible = !on;
  // Neutral light: the mint rim and the pink kicker are pretty on a dark stage
  // but leave coloured fringes along the silhouette that the key would eat.
  rimLight.color.set(on ? 0xffffff : 0x62f2d2);
  warmLight.intensity = on ? 0 : 9;
  document.body.classList.toggle('chroma', on);
  chromaBox.checked = on;
}

function setObsLayout(on) {
  state.obs = on;
  document.body.classList.toggle('obs', on);
}

try {
  const saved = localStorage.getItem(CHROMA_STORE);
  const valid = saved && /^#[0-9a-f]{6}$/i.test(saved);
  setChromaColor(valid ? saved : '#00ff00');
} catch {
  /* storage unavailable; keep the default */
}

/* --------------------------------------------------------------- helpers */
function showToast(message, { sticky = false, error = false } = {}) {
  window.clearTimeout(showToast.timer);
  toastEl.textContent = message;
  toastEl.classList.toggle('error', error);
  toastEl.classList.add('show');
  if (!sticky) {
    showToast.timer = window.setTimeout(
      () => toastEl.classList.remove('show'), 3200);
  }
}

function syncStatus(text, tone = '') {
  statusText.textContent = text;
  statusPill.classList.toggle('live', tone === 'live');
  statusPill.classList.toggle('warn', tone === 'warn');
  statusPill.classList.toggle('error', tone === 'error');
}

/* --------------------------------------------------------------- avatar */
function loadGLTF(url, plugin) {
  return new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    if (plugin) loader.register(plugin);
    loader.load(url, resolve, undefined, reject);
  });
}

async function loadIdle(vrm, url) {
  if (!url) return;
  try {
    const gltf = await loadGLTF(
      url,
      (parser) => new VRMAnimationLoaderPlugin(parser),
    );
    const animation = gltf.userData.vrmAnimations?.[0];
    if (!animation) return;
    const clip = createVRMAnimationClip(animation, vrm);
    const mixer = new THREE.AnimationMixer(vrm.scene);
    mixer.clipAction(clip).play();
    state.mixer = mixer;
  } catch (err) {
    console.warn('[idle] animation unavailable', err);
  }
}

async function loadAvatar(url, label) {
  const gltf = await loadGLTF(
    url,
    (parser) => new VRMLoaderPlugin(parser),
  );
  const vrm = gltf.userData.vrm;
  if (!vrm) throw new Error('VRM data missing from file');

  if (state.mixer) {
    state.mixer.stopAllAction();
    state.mixer = null;
  }
  if (state.vrm) {
    avatarRoot.remove(state.vrm.scene);
    VRMUtils.deepDispose(state.vrm.scene);
  }

  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.rotateVRM0(vrm);
  vrm.scene.traverse((object) => {
    if (object.isMesh) {
      object.castShadow = true;
      object.receiveShadow = false;
      object.frustumCulled = false;
    }
  });

  avatarRoot.add(vrm.scene);
  vrm.scene.updateMatrixWorld(true);

  state.vrm = vrm;
  state.retargeter = new MocapRetargeter(vrm, avatarRoot);
  state.retargeter.camera = camera;
  state.retargeter.rootFollow = followRoot.checked;
  applyFootLock({ persist: false });
  applySmoothing({ persist: false });
  if (vrm.lookAt) vrm.lookAt.target = camera;
  blinkExpression = vrm.expressionManager?.getExpression('blink') ?? null;
  blinkTimer = 2;
  blinkPhase = 0;

  const assets = await fetch('/api/avatar', { cache: 'no-store' })
    .then((r) => r.json())
    .catch(() => ({}));
  await loadIdle(vrm, assets.vrma?.url);

  state.avatarLabel = label || assets.vrm?.name || 'avatar';
  avatarMeta.textContent = state.avatarLabel;
  avatarMeta.title = state.avatarLabel;
  loaderEl.classList.add('done');
}

/* ------------------------------------------------------------- websocket */
function connect() {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(`${proto}://${window.location.host}/ws/pose`);

  socket.addEventListener('open', () => {
    const reconnected = state.wsLost;
    state.wsOpen = true;
    state.wsLost = false;
    updateStatusPill();
    // the MJPEG preview is a stream too: it died with the old server
    if (reconnected) startFeed();
  });

  socket.addEventListener('close', () => {
    state.wsOpen = false;
    state.wsLost = true;
    state.tracking = false;
    updateStatusPill();
    window.setTimeout(connect, 1500);
  });

  socket.addEventListener('error', () => socket.close());

  socket.addEventListener('message', (event) => {
    let message;
    try {
      message = JSON.parse(event.data);
    } catch {
      return;
    }
    if (message.type === 'pose') {
      state.pose = message;
      state.poseAt = performance.now();
      if (message.source) state.source = message.source;
    }
  });
}

/* ------------------------------------------------------------------- API */
function liveLabel(info) {
  if (!info) return 'not running';
  if (info.state === 'loading') {
    const seconds = Number(info.loading_s || 0);
    return seconds >= 1
      ? `loading models… ${seconds.toFixed(0)}s`
      : 'loading models…';
  }
  if (info.state === 'error') return info.error || 'InstantHMR failed';
  if (info.state !== 'running') {
    return info.available && !info.available.ok
      ? info.available.reason : 'not running';
  }
  const dropped = info.dropped ? ` · ${info.dropped} dropped` : '';
  return `${info.fps ?? 0} fps · ${info.ms ?? 0} ms · ${info.frames ?? 0} frames${dropped}`;
}

function syncLiveButton() {
  const info = state.liveInfo;
  const available = !info || !info.available || info.available.ok !== false;
  const alive = info && (info.state === 'loading' || info.state === 'running');
  btnLive.disabled = state.liveBusy || !available;
  btnLive.classList.toggle('active', !!alive);
  btnLive.textContent = state.liveBusy
    ? 'Starting…'
    : (alive ? '3D live on' : '3D live off');
  btnLive.title = available
    ? (alive ? 'Stop the live InstantHMR worker'
             : 'Start live InstantHMR 3D reconstruction')
    : (info.available.reason || 'InstantHMR is not available');
}

function updateStatusPill() {
  const cap = state.capture || {};
  const info = state.liveInfo;
  const fresh = state.pose && performance.now() - state.poseAt < 800;
  state.tracking = !!(fresh && state.pose.joints3d);

  if (!state.wsOpen) {
    syncStatus('connecting', 'warn');
  } else if (cap.state === 'error') {
    syncStatus(cap.error || 'input error', 'error');
  } else if (cap.source === CAMERA_SOURCE && cap.stale) {
    // the page's own link is down, not the person: the watchdog is rebuilding
    syncStatus('browser camera reconnecting', 'warn');
  } else if (!cap.source) {
    syncStatus('no input', 'warn');
  } else if (info && info.state === 'error') {
    syncStatus('InstantHMR error', 'error');
  } else if (info && info.state === 'loading') {
    const seconds = Number(info.loading_s || 0);
    syncStatus(seconds >= 1
      ? `loading InstantHMR ${seconds.toFixed(0)}s`
      : 'loading InstantHMR', 'warn');
  } else if (!info || info.state !== 'running') {
    syncStatus('3D live off', '');
  } else if (state.tracking) {
    syncStatus('3D tracking', 'live');
  } else {
    syncStatus('searching for person', 'warn');
  }
}

function renderStats() {
  const cap = state.capture || {};
  const info = state.liveInfo;
  sourceMeta.textContent = cap.source
    ? `${cap.label || cap.source} · ${cap.width || 0}×${cap.height || 0} · ${cap.fps || 0} fps`
    : 'no input';
  sourceMeta.title = sourceMeta.textContent;

  const label = liveLabel(info);
  liveStats.textContent = label;
  liveStats.title = info && info.log ? `${label}\n${info.log}` : label;
  feedMeta.textContent = state.tracking
    ? `3D tracking · ${info?.fps ?? 0} fps`
    : (cap.source ? 'input' : 'no input');

  syncLiveButton();
  updateStatusPill();
}

async function poll() {
  try {
    const [status, sources] = await Promise.all([
      fetch('/api/status', { cache: 'no-store' }).then((r) => r.json()),
      fetch('/api/sources', { cache: 'no-store' }).then((r) => r.json()),
    ]);
    state.capture = status.capture || {};
    state.liveInfo = status.live || null;
    const boot = status.boot || '';
    const restarted = !!boot && !!state.bootId && boot !== state.bootId;
    if (boot) state.bootId = boot;

    const nextSource = status.capture?.source || state.source;
    const changed = nextSource !== state.source;
    state.source = nextSource;
    if (!state.sourceKnown) {
      state.sourceKnown = true;
      if (!state.desired) state.desired = nextSource;
    }
    if (state.switching) {
      // the capture thread applies a switch a moment after it is accepted, so
      // keep showing the user's pick until the server reports it, then hand
      // the select back to the server
      if (nextSource === state.switching ||
          performance.now() - state.switchAt > 6000) {
        state.switching = '';
      }
    } else if (restarted && state.desired && nextSource !== state.desired) {
      // the container restarted and came back on its default input: take back
      // whatever the user had picked, video included
      reassertSource(state.desired);
    } else if (changed && state.desired === CAMERA_SOURCE &&
               nextSource !== CAMERA_SOURCE) {
      // another tab or a --source changed the input: follow it, do not fight
      state.desired = nextSource;
    }
    if (state.desired === CAMERA_SOURCE) {
      if (!browserCam.active) startBrowserCamera();
    } else if (browserCam.active) {
      stopBrowserCamera();
    }
    renderSources(sources);
    renderStats();
  } catch {
    syncStatus('server offline', 'error');
  }
}

/**
 * Bring back the input the page was on, after a reload.
 *
 * A container restart (or a fresh page) puts the server back on its default
 * input; without this the page silently loses the camera or the video the user
 * had picked. A stored source that no longer exists is dropped and reported.
 */
async function restoreSource() {
  let stored = '';
  try {
    stored = localStorage.getItem(SOURCE_STORE) || '';
  } catch {
    return;
  }
  if (!stored) return;
  state.desired = stored;
  state.switching = stored;
  state.switchAt = performance.now();
  try {
    const response = await fetch('/api/source', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: stored }),
    });
    if (response.ok) {
      if (stored === CAMERA_SOURCE) showToast('Browser camera restored');
      return;
    }
    state.switching = '';
    state.desired = '';
    try {
      localStorage.removeItem(SOURCE_STORE);
    } catch {
      /* private mode: nothing to forget */
    }
    showToast(`Input unavailable: ${stored}`, { error: true });
  } catch {
    state.switching = '';
  }
}

async function refreshLive() {
  try {
    const info = await fetch('/api/live', { cache: 'no-store' })
      .then((r) => r.json());
    state.liveInfo = info;
    renderStats();
  } catch {
    /* the regular poll reports the outage */
  }
}

/* --------------------------------------------------------------- sources */
function renderSources(data) {
  state.sources = data;
  const current = data.current || '';
  // The user's pick wins over the server's report while a switch is in flight,
  // and a source the server is still on keeps its option: an <option> that
  // disappears sets the select to "" and the browser then paints the first
  // entry - the camera - which read as the app switching back by itself.
  const wanted = state.switching || current;
  const items = (data.sources || []).slice();
  for (const id of [wanted, current]) {
    if (id && !items.some((item) => item.id === id)) {
      items.push({
        id,
        label: id.startsWith('video:') ? `${id.slice(6)} (missing)` : id,
        kind: 'custom',
      });
    }
  }
  const signature = items.map((item) => `${item.id}=${item.label}`).join('|');
  if (signature !== renderSources.signature) {
    renderSources.signature = signature;
    const cameras = items.filter((item) => item.kind === 'camera');
    const videos = items.filter((item) => item.kind !== 'camera');
    sourceSelect.innerHTML = '';
    for (const [label, group] of [['Cameras', cameras],
      ['Videos', videos]]) {
      if (!group.length) continue;
      const parent = document.createElement('optgroup');
      parent.label = label;
      for (const item of group) {
        const option = document.createElement('option');
        option.value = item.id;
        option.textContent = item.label;
        parent.appendChild(option);
      }
      sourceSelect.appendChild(parent);
    }
  }
  if (wanted && sourceSelect.value !== wanted) sourceSelect.value = wanted;
}

/** Read the source list again, e.g. after a switch the server refused. */
async function refreshSources() {
  try {
    renderSources(await fetch('/api/sources', { cache: 'no-store' })
      .then((r) => r.json()));
  } catch {
    /* the poll retries */
  }
}

function startFeed() {
  feedImg.src = `/video?t=${Date.now()}`;
  state.feedAt = Date.now();
}

feedImg.addEventListener('error', () => {
  // the MJPEG stream ends when the server restarts: reconnect
  window.setTimeout(() => {
    if (Date.now() - state.feedAt > 1200) startFeed();
  }, 1200);
  state.feedAt = 0;
});

sourceSelect.addEventListener('change', async () => {
  const source = sourceSelect.value;
  if (!source) {
    // a rebuilt list with no matching option: put the select back on the
    // server's input instead of POSTing an empty source
    renderSources(state.sources || {});
    return;
  }
  const previous = state.source;
  const label = sourceSelect.selectedOptions[0]?.textContent || source;
  // Claim the input before the request. The camera watchdog, the socket error
  // handler and the frame already on the wire all re-assert the camera while
  // it is "desired", and any of them landing after this POST flips the server
  // back to the camera - the reported "selecting a video switches me back".
  state.desired = source;
  state.switching = source;
  state.switchAt = performance.now();
  if (source !== CAMERA_SOURCE) await stopBrowserCamera();
  try {
    const response = await fetch('/api/source', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || 'could not switch input');
    state.pose = null;
    state.tracking = false;
    state.source = source;
    state.sourceKnown = true;
    if (state.retargeter) state.retargeter.reset();
    try {
      localStorage.setItem(SOURCE_STORE, source);
    } catch {
      /* private mode: the choice just does not survive a reload */
    }
    if (source === CAMERA_SOURCE) startBrowserCamera();
    showToast(`Input: ${label}`);
  } catch (err) {
    state.switching = '';
    state.desired = previous;
    if (previous === CAMERA_SOURCE) startBrowserCamera();
    showToast(`Input failed: ${err.message}`, { sticky: true, error: true });
    await refreshSources();   // the list may have lost the entry
  }
});

window.addEventListener('beforeunload', () => { stopBrowserCamera(); });

/* ------------------------------------------------------------ live toggle */
async function setLive(on) {
  if (state.liveBusy) return;
  state.liveBusy = true;
  syncLiveButton();
  try {
    const response = await fetch('/api/live', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: on ? 'start' : 'stop' }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    if (data.state === 'error') {
      throw new Error(data.error || 'InstantHMR failed to start');
    }
    state.liveInfo = data;
    if (on) {
      state.pose = null;
      if (state.retargeter) state.retargeter.reset();
      showToast('InstantHMR starting - loading models…');
    } else {
      showToast('Live 3D stopped');
    }
  } catch (err) {
    showToast(`3D live failed: ${err.message}`,
      { sticky: true, error: true });
  } finally {
    state.liveBusy = false;
    renderStats();
    refreshLive();
  }
}

btnLive.addEventListener('click', () => {
  const alive = state.liveInfo &&
    (state.liveInfo.state === 'running' ||
     state.liveInfo.state === 'loading');
  setLive(!alive);
});

followRoot.addEventListener('change', () => {
  if (state.retargeter) state.retargeter.setRootFollow(followRoot.checked);
});

/* --------------------------------------------------------------- foot lock */
// The choice is remembered, like the smoothing and the chroma key: a stream
// should not need the box ticking again after a reload.
const FOOT_LOCK_KEY = 'instanthmr.footlock';

function applyFootLock({ persist = true } = {}) {
  const on = footLockInput.checked;
  if (state.retargeter) state.retargeter.footLock.enabled = on;
  if (persist) localStorage.setItem(FOOT_LOCK_KEY, on ? '1' : '0');
}

footLockInput.addEventListener('change', () => applyFootLock());

{
  const stored = localStorage.getItem(FOOT_LOCK_KEY);
  if (stored !== null) footLockInput.checked = stored === '1';
}

/* -------------------------------------------------------------- smoothing */
// One global filter bank for the whole rig (see filters.js). The slider is
// strength, not Hz: 0 barely touches the motion, 100 is a heavy low-pass that
// floats behind the person. The mapping is geometric so the useful range of
// the slider is spread over its whole travel instead of bunched at one end.
const SMOOTH_STORE = 'instanthmr.smoothing';
const FILTER_OFF = 'off';

function strengthToCutoff(strength) {
  const s = Math.min(1, Math.max(0, strength / 100));
  return 20 * Math.pow(0.06, s);
}

function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

/** Push the panel state into the retargeter and label the sliders. */
function applySmoothing({ persist = true } = {}) {
  const type = filterType.value;
  const strength = Number(filterStrength.value) || 0;
  const deadband = clamp01((Number(filterDeadband.value) || 0) / 30) * 3;
  const cutoff = strengthToCutoff(strength);
  const off = type === FILTER_OFF;

  filterStrength.disabled = off;
  filterDeadband.disabled = off;
  // the windowed methods are tuned in frames, everything else in Hz
  filterStrengthMeta.textContent = off
    ? 'off'
    : (WINDOW_TYPES.includes(type)
      ? `${windowForCutoff(cutoff)} frames`
      : `${cutoff.toFixed(1)} Hz`);
  filterDeadbandMeta.textContent = off ? 'off' : `${deadband.toFixed(1)}°`;

  if (state.retargeter) {
    state.retargeter.smoothing.configure({ type, cutoff, deadband });
  }

  if (!persist) return;
  try {
    localStorage.setItem(SMOOTH_STORE, JSON.stringify({
      type,
      strength,
      deadband: Number(filterDeadband.value) || 0,
    }));
  } catch {
    /* private mode: the choice just does not survive a reload */
  }
}

for (const el of [filterType, filterStrength, filterDeadband]) {
  el.addEventListener('input', () => applySmoothing());
}

try {
  const saved = JSON.parse(localStorage.getItem(SMOOTH_STORE) || 'null');
  if (saved && typeof saved === 'object') {
    if (typeof saved.type === 'string') filterType.value = saved.type;
    if (Number.isFinite(saved.strength)) {
      filterStrength.value = String(saved.strength);
    }
    if (Number.isFinite(saved.deadband)) {
      filterDeadband.value = String(saved.deadband);
    }
  }
} catch {
  /* storage unavailable; keep the defaults */
}
applySmoothing({ persist: false });

/* ------------------------------------------------- green screen / OBS wiring */
chromaBox.addEventListener('change', () => setChroma(chromaBox.checked));
chromaColor.addEventListener('input', () => setChromaColor(chromaColor.value));

for (const key of keyButtons) {
  key.addEventListener('click', () => {
    setChromaColor(key.dataset.color);
    // picking a key is a request to see it
    if (!state.chroma) setChroma(true);
  });
}

btnObs.addEventListener('click', () => {
  const url = new URL(window.location.href);
  url.searchParams.set('obs', '1');
  window.open(url.toString(), '_blank', 'noopener');
});

// ?obs=1 is the capture layout: green field, no controls. ?green=1 keeps the
// controls and only drops the stage. G and H flip the same two things live, so
// a window capture can be previewed without reloading.
const query = new URLSearchParams(window.location.search);
if (query.has('obs')) {
  setChroma(true);
  setObsLayout(true);
} else if (query.has('green') || query.has('chroma')) {
  setChroma(true);
}

window.addEventListener('keydown', (event) => {
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const tag = event.target?.tagName;
  const type = event.target?.type;
  const typing = tag === 'TEXTAREA' || tag === 'SELECT' ||
    (tag === 'INPUT' && !['checkbox', 'radio', 'button', 'range'].includes(type));

  if (event.key === 'Escape') {
    setObsLayout(!state.obs);
  } else if (!typing && event.key.toLowerCase() === 'g') {
    setChroma(!state.chroma);
  } else if (!typing && event.key.toLowerCase() === 'h') {
    setObsLayout(!state.obs);
  } else {
    return;
  }
  showToast(`${state.chroma ? 'Chroma key' : 'Stage'} · ` +
    `${state.obs ? 'controls hidden' : 'controls shown'}`);
});

/* ---------------------------------------------------------------- uploads */
btnUpload.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async () => {
  const file = fileInput.files?.[0];
  fileInput.value = '';
  if (!file) return;
  btnUpload.disabled = true;
  btnUpload.textContent = 'Uploading…';
  try {
    const response = await fetch(
      `/api/upload?name=${encodeURIComponent(file.name)}`,
      { method: 'POST', body: file },
    );
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || 'upload failed');
    renderSources.signature = '';      // force the list to rebuild
    await poll();
    await fetch('/api/source', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: data.source }),
    });
    state.pose = null;
    state.tracking = false;
    if (state.retargeter) state.retargeter.reset();
    showToast(`Added ${file.name}`);
  } catch (err) {
    showToast(`Upload failed: ${err.message}`,
      { sticky: true, error: true });
  } finally {
    btnUpload.disabled = false;
    btnUpload.textContent = 'Add video…';
  }
});

btnAvatar.addEventListener('click', () => avatarInput.click());

avatarInput.addEventListener('change', async () => {
  const file = avatarInput.files?.[0];
  avatarInput.value = '';
  if (!file) return;
  const was = btnAvatar.textContent;
  btnAvatar.disabled = true;
  btnAvatar.textContent = 'Loading VRM…';
  let local = '';
  try {
    // Upload before loading. A picked file used to be shown from a blob URL,
    // which dies with the window: the pick looked right until a clean window
    // (or a reload, or OBS) went back to the avatar that was already on disk.
    // Storing it makes the pick the server's answer for every window.
    let url = '';
    let label = file.name;
    try {
      const response = await fetch(
        `/api/avatar?name=${encodeURIComponent(file.name)}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/octet-stream' },
          body: file,
        },
      );
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || 'upload failed');
      url = data.avatar?.url || '';
      label = data.avatar?.name || file.name;
    } catch (err) {
      // offline, or the server refused it: still show it here, and say so
      local = URL.createObjectURL(file);
      url = local;
      showToast(`Avatar in this window only: ${err.message}`,
        { error: true });
    }
    await loadAvatar(url, label);
    if (!local) showToast(`Avatar: ${label}`);
  } catch (err) {
    showToast(`Avatar failed: ${err.message}`,
      { sticky: true, error: true });
  } finally {
    if (local) URL.revokeObjectURL(local);
    btnAvatar.disabled = false;
    btnAvatar.textContent = was;
  }
});

/* ------------------------------------------------------------------ loop */
function updateBlink(dt) {
  if (!state.vrm?.expressionManager || !blinkExpression) return;
  blinkTimer -= dt;
  if (blinkTimer <= 0) {
    blinkPhase = 0.15;
    blinkTimer = 2 + Math.random() * 4;
  }
  let value = 0;
  if (blinkPhase > 0) {
    blinkPhase = Math.max(0, blinkPhase - dt);
    value = Math.sin((blinkPhase / 0.15) * Math.PI);
  }
  state.vrm.expressionManager.setValue('blink', value);
}

let lastFrame = performance.now();

function animate(now) {
  requestAnimationFrame(animate);
  const dt = Math.min(0.1, Math.max(0.001, (now - lastFrame) / 1000));
  lastFrame = now;

  if (state.mixer) state.mixer.update(dt);

  if (state.retargeter) {
    const fresh = state.pose && performance.now() - state.poseAt < 800;
    state.tracking = !!(fresh && state.pose.joints3d);
    state.retargeter.update(fresh ? state.pose : null, dt);
  }

  updateBlink(dt);
  if (state.vrm) state.vrm.update(dt);

  const tracked = state.tracking;
  blob.material.opacity = tracked ? 1 : 0.35;
  blob.position.x = avatarRoot.position.x;

  controls.update();
  renderer.render(scene, camera);
}

/* ------------------------------------------------------------------ init */
window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  fitCamera();
});

window.mocap = {
  state, scene, camera, controls, avatarRoot, renderer,
  browserCamera: browserCam,
  setChroma, setChromaColor, setObsLayout, applySmoothing,
};

requestAnimationFrame(animate);

(async () => {
  startFeed();
  connect();
  try {
    const assets = await fetch('/api/avatar', { cache: 'no-store' })
      .then((r) => r.json());
    if (!assets.vrm) throw new Error('no .vrm in vrm_models/');
    await loadAvatar(assets.vrm.url, assets.vrm.name);
  } catch (err) {
    console.error(err);
    loaderText.textContent = 'Could not load the VRM avatar';
    showToast(`VRM load failed: ${err.message}`,
      { sticky: true, error: true });
  }
  await restoreSource();
  await poll();
  window.setInterval(poll, 1000);
})();
