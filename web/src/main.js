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
  source: '',
  capture: { state: 'starting', source: '', label: '', fps: 0 },
  liveInfo: null,
  liveBusy: false,
  sources: [],
  feedAt: 0,
  avatarLabel: '',
  chroma: false,
  obs: false,
};

let browserCam = {
  stream: null,
  video: null,
  canvas: null,
  timer: 0,
  sending: false,
  active: false,
};

async function stopBrowserCamera() {
  browserCam.active = false;
  if (browserCam.timer) {
    window.clearInterval(browserCam.timer);
    browserCam.timer = 0;
  }
  if (browserCam.stream) {
    for (const track of browserCam.stream.getTracks()) track.stop();
  }
  browserCam.stream = null;
  if (browserCam.video) {
    browserCam.video.srcObject = null;
  }
  browserCam.video = null;
  browserCam.canvas = null;
  browserCam.sending = false;
}

async function startBrowserCamera() {
  if (browserCam.active || state.source !== 'browser:camera') return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast('Browser camera not supported in this browser',
      { sticky: true, error: true });
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    if (state.source !== 'browser:camera') {
      for (const track of stream.getTracks()) track.stop();
      return;
    }
    const video = document.createElement('video');
    video.autoplay = true;
    video.playsInline = true;
    video.muted = true;
    video.srcObject = stream;
    await video.play().catch(() => {});
    const canvas = document.createElement('canvas');
    browserCam = { stream, video, canvas, timer: 0, sending: false, active: true };

    const sendFrame = async () => {
      if (!browserCam.active || browserCam.sending) return;
      const v = browserCam.video;
      if (!v || v.readyState < 2) return;
      const w = v.videoWidth || 1280;
      const h = v.videoHeight || 720;
      if (browserCam.canvas.width !== w || browserCam.canvas.height !== h) {
        browserCam.canvas.width = w;
        browserCam.canvas.height = h;
      }
      const ctx = browserCam.canvas.getContext('2d');
      ctx.drawImage(v, 0, 0, w, h);
      browserCam.sending = true;
      try {
        const blob = await new Promise((resolve) =>
          browserCam.canvas.toBlob(resolve, 'image/jpeg', 0.82));
        if (!blob || !browserCam.active) return;
        await fetch('/api/browser-frame', {
          method: 'POST',
          headers: { 'Content-Type': 'image/jpeg' },
          body: blob,
          keepalive: true,
        });
      } catch {
        /* dropped frame; keep streaming */
      } finally {
        browserCam.sending = false;
      }
    };
    browserCam.timer = window.setInterval(sendFrame, 1000 / 24);
    sendFrame();
  } catch (err) {
    showToast(`Browser camera failed: ${err.message || err}`,
      { sticky: true, error: true });
  }
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
    state.wsOpen = true;
    updateStatusPill();
  });

  socket.addEventListener('close', () => {
    state.wsOpen = false;
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
    const nextSource = status.capture?.source || state.source;
    if (nextSource !== state.source) {
      state.source = nextSource;
      if (state.source === 'browser:camera') startBrowserCamera();
      else if (browserCam.active) stopBrowserCamera();
    } else {
      state.source = nextSource;
      if (state.source === 'browser:camera' && !browserCam.active) startBrowserCamera();
    }
    renderSources(sources);
    renderStats();
  } catch {
    syncStatus('server offline', 'error');
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
  const items = data.sources || [];
  const signature = items.map((item) => item.id).join('|');
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
  const current = data.current || '';
  if (current && sourceSelect.value !== current) {
    sourceSelect.value = current;
    if (sourceSelect.value !== current && current.startsWith('video:')) {
      // the select is rebuilt before a just-started source appears in it
      const option = document.createElement('option');
      option.value = current;
      option.textContent = current.slice(6);
      sourceSelect.appendChild(option);
      sourceSelect.value = current;
    }
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
  try {
    const response = await fetch('/api/source', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'could not switch input');
    state.pose = null;
    state.tracking = false;
    state.source = source;
    if (state.retargeter) state.retargeter.reset();
    await stopBrowserCamera();
    if (source === 'browser:camera') startBrowserCamera();
    showToast(`Input: ${sourceSelect.selectedOptions[0]?.textContent || source}`);
  } catch (err) {
    showToast(`Input failed: ${err.message}`, { sticky: true, error: true });
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
  const url = URL.createObjectURL(file);
  try {
    await loadAvatar(url, file.name);
    showToast(`Avatar: ${file.name}`);
  } catch (err) {
    showToast(`Avatar failed: ${err.message}`,
      { sticky: true, error: true });
  } finally {
    URL.revokeObjectURL(url);
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
  await poll();
  window.setInterval(poll, 1000);
})();
