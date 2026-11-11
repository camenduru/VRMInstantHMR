"""InstantHMR live 3D reconstruction for the standalone VRM app.

InstantHMR cannot run inside this app's interpreter: it needs Python 3.10-3.12
with PyTorch for the RF-DETR person detector plus its own onnxruntime, while
the web server runs on a newer Python with onnxruntime only. So live mode keeps
one child process alive in InstantHMR's Python (``scripts/instanthmr_live.py``)
and streams the newest capture frame into it.

Loading the graphs takes seconds, so the child is reused instead of started per
frame. Only one frame is in flight; while the child is busy the newest capture
replaces the queued one, which keeps latency at one inference instead of letting
a backlog build up.

The child replies with the 70 MHR landmarks (3D in camera axes, 2D in submitted
pixels). ``mhr_to_smpl`` maps them onto the 22-joint SMPL layout the browser
retargeter reads, and ``LiveInstantHMR`` publishes that as the live pose.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INSTANTHMR = Path(os.environ.get('INSTANTHMR_DIR') or (ROOT.parent / 'InstantHMR'))
LIVE_RUNNER = ROOT / 'scripts' / 'instanthmr_live.py'

# The live worker gets a downscaled frame: the HMR crops are ~256 px, so a
# 1280-wide capture buys nothing but pipe bandwidth. The 2D preview joints come
# back in this frame's coordinates and are normalised, so the size is free.
LIVE_WIDTH = 640

# The MHR70 landmark ordering shipped with InstantHMR (instanthmr.JOINT_NAMES).
# Only body landmarks are used; hands stay out of the payload because the VRM
# humanoid rig has no finger bones.
MHR = {
    'nose': 0,
    'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_hip': 9, 'right_hip': 10,
    'left_knee': 11, 'right_knee': 12,
    'left_ankle': 13, 'right_ankle': 14,
    'left_big_toe': 15, 'left_small_toe': 16,
    'right_big_toe': 18, 'right_small_toe': 19,
    'right_wrist': 41,
    'left_wrist': 62,
    'neck': 69,
}

MHR_JOINTS = 70
BODY_JOINTS = 22

# Auxiliary MHR70 landmarks that describe the roll of a bone. The 22-joint
# skeleton fixes where a bone points but not how it is twisted, so the browser
# needs these extra anatomical references:
#   nose/ears          head yaw
#   olecranon/cubital  back of the elbow (upper arm + forearm twist)
#   middle knuckle     wrist flex direction
#   heel               sole normal (foot roll)
LANDMARKS = {
    'nose': 0,
    'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_heel': 17, 'right_heel': 20,
    'right_middle': 30, 'left_middle': 51,
    'left_olecranon': 63, 'right_olecranon': 64,
    'left_cubital': 65, 'right_cubital': 66,
}


def interpreter() -> Optional[Path]:
    """The Python that has InstantHMR's torch/ONNX dependencies."""
    env = os.environ.get('INSTANTHMR_PYTHON')
    if env and Path(env).exists():
        return Path(env)
    marker = INSTANTHMR / '.instanthmr_python'
    if marker.exists():
        found = Path(marker.read_text(encoding='utf-8').strip())
        if found.exists():
            return found
    for candidate in (INSTANTHMR / '.venv' / 'Scripts' / 'python.exe',
                      INSTANTHMR / '.venv' / 'bin' / 'python'):
        if candidate.exists():
            return candidate
    return None


def availability() -> dict:
    """What the UI needs to know before offering live 3D."""
    if not (INSTANTHMR / 'instanthmr' / 'pipeline.py').exists():
        return {'ok': False,
                'reason': f'InstantHMR checkout not found in {INSTANTHMR}',
                'missing': []}
    if not LIVE_RUNNER.exists():
        return {'ok': False,
                'reason': f'live runner not found at {LIVE_RUNNER}',
                'missing': []}
    python = interpreter()
    if python is None:
        return {'ok': False,
                'reason': 'InstantHMR Python not configured - create its '
                          'environment (InstantHMR/README.md) and set '
                          'INSTANTHMR_PYTHON to that interpreter',
                'missing': []}
    return {'ok': True, 'reason': 'ready', 'python': str(python),
            'missing': []}


def mhr_to_smpl(points) -> np.ndarray:
    """Map one frame of MHR70 landmarks to the 22 SMPL-ordered body joints.

    ``points`` is (70, 2) or (70, 3); the mapping is a set of linear
    combinations, so the same code serves the preview skeleton (pixels) and the
    retargeter (metres). The synthetic pelvis/spine joints inherit the measured
    points' frame.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < MHR_JOINTS or pts.shape[1] not in (2, 3):
        raise ValueError(f'expected ({MHR_JOINTS}, 2 or 3) MHR joints, got '
                         f'{pts.shape}')

    left_hip = pts[MHR['left_hip']]
    right_hip = pts[MHR['right_hip']]
    neck = pts[MHR['neck']]
    pelvis = (left_hip + right_hip) * 0.5
    # the spine chain follows the torso (hips -> shoulders), not the neck:
    # a tilted head must not bend the whole back
    chest = (pts[MHR['left_shoulder']] + pts[MHR['right_shoulder']]) * 0.5

    out = np.empty((BODY_JOINTS, pts.shape[1]), dtype=np.float32)
    out[0] = pelvis                                    # pelvis
    out[1] = left_hip                                  # left hip
    out[2] = right_hip                                 # right hip
    # spine, spine1, spine2: evenly spaced up the torso
    for index, fraction in ((3, 0.25), (6, 0.5), (9, 0.75)):
        out[index] = pelvis + (chest - pelvis) * fraction
    out[4] = pts[MHR['left_knee']]
    out[5] = pts[MHR['right_knee']]
    out[7] = pts[MHR['left_ankle']]
    out[8] = pts[MHR['right_ankle']]
    out[10] = (pts[MHR['left_big_toe']] + pts[MHR['left_small_toe']]) * 0.5
    out[11] = (pts[MHR['right_big_toe']] + pts[MHR['right_small_toe']]) * 0.5
    out[12] = neck
    out[13] = (neck + pts[MHR['left_shoulder']]) * 0.5   # left collar
    out[14] = (neck + pts[MHR['right_shoulder']]) * 0.5  # right collar
    # the head joint is the centre of the face landmarks, not the nose tip
    out[15] = pts[0:5].mean(axis=0)
    out[16] = pts[MHR['left_shoulder']]
    out[17] = pts[MHR['right_shoulder']]
    out[18] = pts[MHR['left_elbow']]
    out[19] = pts[MHR['right_elbow']]
    out[20] = pts[MHR['left_wrist']]
    out[21] = pts[MHR['right_wrist']]
    return out


def mhr_landmarks(points) -> dict:
    """Named auxiliary landmarks, in the same frame as the 22 body joints."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < MHR_JOINTS or pts.shape[1] != 3:
        return {}
    return {name: np.round(pts[index], 4).tolist()
            for name, index in LANDMARKS.items()}


class LiveInstantHMR:
    """A persistent InstantHMR process fed the live capture frames.

    ``status()`` is the JSON the UI polls, ``pose()`` the newest mapped frame
    (``None`` until the first detected person), and ``submit()`` is the capture
    frame sink: it only ever queues, so it never stalls the capture thread.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._proc = None
        self._threads: list = []
        self._pending = None              # newest unsent (frame, wall time)
        self._inflight = 0                # seq written, reply not seen yet
        self._sent: dict = {}             # seq -> (perf_counter, (w, h))
        self._next_seq = 1
        self._pose = None
        self._pose_seq = 0
        self._replies: deque = deque(maxlen=30)
        # Last non-protocol lines the worker printed. Loading the graphs takes
        # tens of seconds and a fatal import/driver error shows up as one of
        # these, so they are worth showing in the UI instead of swallowing.
        self._tail: deque = deque(maxlen=6)
        self._started = 0.0
        self._status = {
            'state': 'off',
            'ready': False,
            'error': '',
            'fps': 0.0,
            'ms': 0.0,
            'dropped': 0,
            'frames': 0,
            'model': '',
            'detector': '',
            'stride': 0,
            'fit': 0,
            'provider': '',
        }

    # ------------------------------------------------------------------ state
    def status(self) -> dict:
        with self._lock:
            data = dict(self._status)
            if data['state'] == 'loading' and self._started:
                data['loading_s'] = round(time.time() - self._started, 1)
            if self._tail:
                data['log'] = ' | '.join(self._tail)
            return data

    def pose(self) -> Optional[dict]:
        """The newest 3D pose, or None while the worker is not tracking."""
        with self._lock:
            return self._pose

    def running(self) -> bool:
        with self._lock:
            return self._status['state'] in ('loading', 'running')

    # ------------------------------------------------------------------ start
    def start(self, model: str = 'medium', fit: int = 2, device: str = '',
              detector: str = 'medium', detector_stride: int = 2,
              pose_filter: str = 'one-euro') -> dict:
        check = availability()
        if not check['ok']:
            raise RuntimeError(check['reason'])
        python = interpreter()
        if python is None:
            raise RuntimeError('InstantHMR Python not configured')

        if self.running():
            self.stop()

        fit = int(fit)
        if fit not in (0, 1, 2):
            raise ValueError('fit must be 0, 1 or 2')
        detector_stride = max(1, int(detector_stride))
        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        env['PYTHONPATH'] = str(INSTANTHMR) + os.pathsep + \
            env.get('PYTHONPATH', '')
        cmd = [str(python), str(LIVE_RUNNER),
               '--model', str(model), '--fit', str(fit),
               '--detector', str(detector),
               '--detector-stride', str(detector_stride),
               '--pose-filter', str(pose_filter)]
        device = device or os.environ.get('INSTANTHMR_DEVICE', 'auto')
        if device:
            cmd.extend(['--device', device])

        self._stop = threading.Event()
        with self._lock:
            self._pose = None
            self._pose_seq = 0
            self._pending = None
            self._inflight = 0
            self._sent.clear()
            self._replies.clear()
            self._next_seq = 1
            self._tail.clear()
            self._started = time.time()
            self._status.update(state='loading', ready=False, error='',
                                fps=0.0, ms=0.0, dropped=0, frames=0,
                                model=str(model), detector=str(detector),
                                stride=detector_stride, fit=fit, provider='')
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(INSTANTHMR), env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)
        except OSError as exc:
            with self._lock:
                self._status.update(state='error',
                                    error=f'could not start InstantHMR: {exc}')
            return self.status()
        self._proc = proc
        self._threads = [
            threading.Thread(target=self._write_loop, args=(proc,),
                             daemon=True),
            threading.Thread(target=self._read_loop, args=(proc,),
                             daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        return self.status()

    # ------------------------------------------------------------------- pipe
    def submit(self, frame: np.ndarray, _t: float = 0.0) -> bool:
        """Queue the newest capture frame; older queued frames are dropped."""
        with self._cond:
            if self._status['state'] != 'running':
                return False
            if self._pending is not None:
                self._status['dropped'] += 1
            self._pending = (frame, time.time())
            self._cond.notify_all()
            return True

    def _write_loop(self, proc):
        try:
            while not self._stop.is_set():
                with self._cond:
                    while not self._stop.is_set() and (
                            self._pending is None or self._inflight):
                        self._cond.wait(0.5)
                    if self._stop.is_set():
                        return
                    frame, _sent_at = self._pending
                    self._pending = None
                    seq = self._next_seq
                    self._next_seq += 1
                    self._inflight = seq
                rgb = _frame_rgb(frame)
                with self._lock:
                    self._sent[seq] = (time.perf_counter(),
                                       (rgb.shape[1], rgb.shape[0]))
                header = struct.pack('<IIQ', rgb.shape[1], rgb.shape[0], seq)
                proc.stdin.write(header)
                proc.stdin.write(rgb.tobytes())
                proc.stdin.flush()
        except Exception as exc:  # noqa: BLE001
            self._fail(f'live InstantHMR pipe failed: {exc}')

    def _read_loop(self, proc):
        for raw in proc.stdout:
            line = raw.decode('utf-8', 'replace').strip()
            if not line:
                continue
            if not line.startswith('{'):
                # loading chatter and library warnings are not protocol, but a
                # driver/import failure appears here and is the only clue the
                # user would otherwise never see
                with self._lock:
                    self._tail.append(line[:300])
                    state = self._status['state']
                if state == 'loading':
                    print(f'[instanthmr] {line}', flush=True)
                continue
            try:
                message = json.loads(line)
            except ValueError:
                continue
            self._handle(message)
        with self._lock:
            if self._stop.is_set():
                return
            state = self._status['state']
            tail = ' | '.join(self._tail)
        if state in ('loading', 'running'):
            detail = f' - last output: {tail}' if tail else ''
            message = ('InstantHMR exited before it became ready'
                       if state == 'loading'
                       else 'InstantHMR stopped unexpectedly')
            self._fail(message + detail)

    def _handle(self, message: dict):
        if message.get('fatal'):
            self._fail(str(message['fatal']))
            return
        if message.get('ready'):
            with self._lock:
                self._status.update(state='running', ready=True, error='',
                                    provider=str(message.get('provider', '')))
            return
        try:
            seq = int(message.get('seq', 0))
        except (TypeError, ValueError):
            seq = 0
        now = time.perf_counter()
        with self._lock:
            sent = self._sent.pop(seq, None)
            latency = (now - sent[0]) * 1000.0 if sent is not None else 0.0
            if self._inflight == seq:
                self._inflight = 0
            self._replies.append(now)
            self._status['frames'] += 1
            self._status['ms'] = round(latency, 1)
            if len(self._replies) > 1:
                span = self._replies[-1] - self._replies[0]
                if span > 0:
                    self._status['fps'] = round(
                        (len(self._replies) - 1) / span, 1)
            if 'error' in message:
                self._status['error'] = str(message['error'])
                self._cond.notify_all()
                return
            self._status['error'] = ''
            joints = message.get('joints')
            detected = bool(message.get('detected')) and bool(joints)
            self._pose_seq += 1
            pose = {'seq': self._pose_seq, 'detected': detected,
                    'latency': latency, 'fps': self._status['fps']}
            if detected:
                joints = np.asarray(joints, np.float32)
                body = mhr_to_smpl(joints)
                pose['joints3d'] = np.round(body, 4).tolist()
                landmarks = mhr_landmarks(joints)
                if landmarks:
                    pose['landmarks'] = landmarks
                # preview skeleton: SMPL22 in normalised, unmirrored frame
                # coordinates (the capture mirrors its preview itself)
                screen = mhr_to_smpl(np.asarray(message.get('joints2d'),
                                                np.float32)) \
                    if message.get('joints2d') else None
                size = sent[1] if sent is not None else (640, 360)
                if screen is not None:
                    pose['skeleton2d'] = np.round(
                        screen / np.asarray(size, np.float32), 4).tolist()
                # detector box in normalised coordinates: whether the feet are
                # inside the shot is the only clue that a close-up crop has
                # hallucinated legs, and the box bottom shows where the frame
                # cuts the body
                box = message.get('box')
                if box:
                    box = np.asarray(box, np.float32) / np.asarray(
                        (size[0], size[1], size[0], size[1]), np.float32)
                    pose['box2d'] = np.round(box, 4).tolist()
            self._pose = pose
            self._cond.notify_all()

    # ------------------------------------------------------------------- stop
    def stop(self) -> dict:
        with self._cond:
            was_running = self._status['state'] in ('loading', 'running')
            self._stop.set()
            self._pending = None
            self._cond.notify_all()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.write(struct.pack('<IIQ', 0, 0, 0))
                proc.stdin.flush()
            except Exception:  # noqa: BLE001
                pass
            try:
                proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                proc.kill()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads = []
        if proc is not None:
            for stream in (proc.stdin, proc.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:  # noqa: BLE001
                    pass
        self._proc = None
        with self._lock:
            self._pose = None
            self._inflight = 0
            self._sent.clear()
            self._status.update(state='off', ready=False, fps=0.0, ms=0.0)
            if not was_running:
                self._status['error'] = ''
        return self.status()

    def _fail(self, message: str):
        with self._cond:
            self._status.update(state='error', ready=False, error=str(message))
            self._cond.notify_all()
        proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass


def _frame_rgb(frame: np.ndarray) -> np.ndarray:
    """BGR capture frame -> contiguous RGB at most LIVE_WIDTH wide."""
    height, width = frame.shape[:2]
    if width > LIVE_WIDTH:
        frame = cv2.resize(
            frame, (LIVE_WIDTH, max(1, round(height * LIVE_WIDTH / width))),
            interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
