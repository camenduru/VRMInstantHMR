"""Video/camera capture, MJPEG preview and the live 3D frame sink.

The capture owns exactly one input at a time: a video file (looped) or a
camera index. Every frame goes two ways:

* the unmirrored BGR frame is handed to ``frame_sink`` (the InstantHMR worker),
  because the 3D model expects the real orientation;
* a mirrored copy is resized, annotated with the newest 3D skeleton and JPEG
  encoded for the browser preview, so the preview reads like a selfie.

No pose model lives here: the preview skeleton is drawn from the 3D worker's
2D landmarks, so the app has a single source of motion.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

VIDEO_EXT = {'.mp4', '.webm', '.mov', '.mkv', '.avi', '.m4v'}
PREVIEW_WIDTH = 640
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERAS = 4
BROWSER_SOURCE = 'browser:camera'
# A pushed-frame source that stops sending is a dead link, not a still person:
# the status reports it so the page can tear the transport down and rebuild it.
BROWSER_STALE_S = 2.5

# SMPL22 body edges, drawn between the mapped MHR landmarks.
SMPL_EDGES = (
    (0, 1), (0, 2), (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (9, 13), (13, 16),
    (16, 18), (18, 20), (9, 14), (14, 17), (17, 19), (19, 21),
    (12, 15),
)
SKELETON_COLOR = (210, 242, 98)      # the UI's mint accent, as BGR
JOINT_COLOR = (255, 255, 255)


class Capture:
    """A threaded video/camera source with a shared JPEG preview."""

    def __init__(self, video_dirs: list[Path]):
        self.video_dirs = [Path(p) for p in video_dirs]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pending = None
        self.source = ''
        self._jpeg: Optional[bytes] = None
        self._seq = 0
        self._status = {
            'state': 'starting',
            'source': '',
            'label': '',
            'kind': '',
            'error': '',
            'fps': 0.0,
            'width': 0,
            'height': 0,
            'frames': 0,
            'media_t': None,
        }
        # ``(frame_bgr, wall time)`` hook for the 3D worker; always the
        # unmirrored capture frame
        self.frame_sink = None
        # ``() -> live pose dict`` used only to draw the preview skeleton
        self.pose_provider = None
        # timestamps for pushed (browser) frame FPS
        self._push_times: list[float] = []
        # wall time of the newest accepted pushed frame (browser camera)
        self._last_push = 0.0

    # ---------------------------------------------------------------- sources
    def _videos(self) -> list[dict]:
        seen = set()
        items = []
        for folder in self.video_dirs:
            if not folder.exists():
                continue
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in VIDEO_EXT or path.name in seen:
                    continue
                seen.add(path.name)
                items.append({'id': f'video:{path.name}',
                              'label': path.stem.replace('_', ' '),
                              'kind': 'video'})
        return items

    def available_sources(self) -> list[dict]:
        items = [{'id': BROWSER_SOURCE, 'label': 'Browser camera',
                  'kind': 'camera'}]
        items.extend({'id': f'camera:{i}', 'label': f'Camera {i}',
                      'kind': 'camera'} for i in range(CAMERAS))
        items.extend(self._videos())
        if self.source and not any(i['id'] == self.source for i in items):
            kind, _, value = self.source.partition(':')
            items.insert(0, {'id': self.source,
                             'label': (f'{value} (missing)' if kind == 'video'
                                       else self.source),
                             'kind': 'custom'})
        return items

    def resolve(self, source: str) -> Optional[Path]:
        kind, _, value = source.partition(':')
        if kind != 'video':
            return None
        for folder in self.video_dirs:
            path = folder / value
            try:
                path = path.resolve()
            except OSError:
                continue
            if path.parent in {f.resolve() for f in self.video_dirs} \
                    and path.suffix.lower() in VIDEO_EXT and path.exists():
                return path
        return None

    def set_source(self, source: str) -> None:
        """Switch input; ``camera:N``, ``browser:camera`` or ``video:<name>``."""
        kind, _, value = source.partition(':')
        if source == BROWSER_SOURCE:
            pass
        elif kind == 'camera':
            if not value.isdigit() or int(value) >= CAMERAS:
                raise ValueError(f'unknown camera {source!r}')
        elif kind == 'video':
            if self.resolve(source) is None:
                raise ValueError(f'unknown video {source!r}')
        else:
            raise ValueError(f'unknown source {source!r}')
        with self._lock:
            if source == self.source and self._pending is None:
                # already on this input: a re-assert (the page taking its
                # camera back after a restart) must not wedge the status in
                # ``switching`` for an input that never changes
                return
            self._pending = source
            self._status.update(state='switching', error='')

    # ------------------------------------------------------------------ thread
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _open(self, source: str):
        if source == BROWSER_SOURCE:
            return None, None, 'Browser camera'
        kind, _, value = source.partition(':')
        if kind == 'camera':
            index = int(value)
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) \
                if hasattr(cv2, 'CAP_DSHOW') else cv2.VideoCapture(index)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                raise RuntimeError(f'could not open camera {index}')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            return cap, None, None
        path = self.resolve(source)
        if path is None:
            raise RuntimeError(f'video not found: {source}')
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f'could not open video {path.name}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if 1.0 <= fps <= 240.0 else 30.0
        label = path.stem.replace('_', ' ')
        return cap, 1.0 / fps, label

    def _loop(self) -> None:
        cap = None
        current = ''
        frame_interval = 0.0
        due = 0.0
        media_frame = 0
        times: list[float] = []

        while not self._stop.is_set():
            with self._lock:
                if self._pending is not None:
                    pending, self._pending = self._pending, None
                else:
                    pending = None
            if pending is not None and pending != current:
                if cap is not None:
                    cap.release()
                    cap = None
                current = pending
                media_frame = 0
                times.clear()
                self._push_times.clear()
                self._last_push = 0.0
                try:
                    cap, frame_interval, label = self._open(current)
                except Exception as exc:  # noqa: BLE001
                    with self._lock:
                        self._status.update(state='error', source=current,
                                            label=current, kind='',
                                            error=str(exc), fps=0.0,
                                            width=0, height=0)
                    current = ''
                    continue
                due = time.perf_counter()
                with self._lock:
                    self.source = current
                    self._status.update(state='running', source=current,
                                        label=label or current,
                                        kind=current.partition(':')[0],
                                        error='', fps=0.0, frames=0,
                                        media_t=None)
                continue

            if pending is not None:
                # a queued switch to the input already in use: nothing to
                # reopen, but the status must not stay on ``switching``
                with self._lock:
                    if self._status.get('state') == 'switching':
                        self._status.update(state='running', error='')

            # Browser camera: frames are pushed externally via push_frame().
            if current == BROWSER_SOURCE:
                time.sleep(0.02)
                continue

            if cap is None or not current:
                time.sleep(0.05)
                continue

            if frame_interval:
                wait = due - time.perf_counter()
                if wait > 0:
                    self._stop.wait(min(wait, 0.1))
                    if self._stop.is_set():
                        break
                    continue
                due += frame_interval
                if due < time.perf_counter() - frame_interval:
                    due = time.perf_counter()      # a stall must not burst

            ok, frame = cap.read()
            if not ok or frame is None:
                if frame_interval:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop the clip
                    media_frame = 0
                    due = time.perf_counter()
                    continue
                time.sleep(0.1)                       # camera hiccup
                continue
            media_frame += 1

            sink = self.frame_sink
            if sink is not None:
                try:
                    sink(frame, time.time())
                except Exception as exc:  # noqa: BLE001
                    print(f'[live3d] frame sink failed: {exc}', flush=True)
                    self.frame_sink = None

            pose = None
            provider = self.pose_provider
            if provider is not None:
                try:
                    pose = provider()
                except Exception:  # noqa: BLE001
                    pose = None
            preview = self._render(frame, pose)

            now = time.perf_counter()
            times.append(now)
            while times and now - times[0] > 1.5:
                times.pop(0)
            fps = (len(times) - 1) / (times[-1] - times[0]) \
                if len(times) > 1 else 0.0

            ok, encoded = cv2.imencode('.jpg', preview,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
            h, w = frame.shape[:2]
            with self._lock:
                if ok:
                    self._jpeg = encoded.tobytes()
                    self._seq += 1
                self._status.update(width=w, height=h,
                                    fps=round(fps, 1),
                                    frames=self._status['frames'] + 1,
                                    media_t=(round(media_frame * frame_interval,
                                                   3) if frame_interval
                                             else None))

        if cap is not None:
            cap.release()

    # ---------------------------------------------------------------- preview
    @staticmethod
    def _render(frame: np.ndarray, pose: Optional[dict]) -> np.ndarray:
        preview = cv2.flip(frame, 1)
        h, w = preview.shape[:2]
        if pose and pose.get('detected') and pose.get('skeleton2d'):
            # the joints are normalised in the unmirrored frame; mirror x to
            # match the preview
            points = np.asarray(pose['skeleton2d'], np.float32)
            points[:, 0] = 1.0 - points[:, 0]
            points[:, 0] *= w
            points[:, 1] *= h
            for a, b in SMPL_EDGES:
                p, q = points[a], points[b]
                cv2.line(preview, (int(p[0]), int(p[1])),
                         (int(q[0]), int(q[1])), SKELETON_COLOR, 3,
                         cv2.LINE_AA)
            for x, y in points:
                cv2.circle(preview, (int(x), int(y)), 4, JOINT_COLOR, -1,
                           cv2.LINE_AA)
        out_w = min(PREVIEW_WIDTH, w)
        out_h = max(1, round(h * out_w / w))
        return cv2.resize(preview, (out_w, out_h),
                          interpolation=cv2.INTER_AREA)

    # ----------------------------------------------------------- pushed frames
    def push_frame(self, frame: np.ndarray, wall_time: Optional[float] = None) -> bool:
        """Accept a frame for the browser camera source.

        Returns True if the frame was consumed.
        """
        if wall_time is None:
            wall_time = time.time()
        with self._lock:
            # Only the source matters here. The state flips to ``switching``
            # for a moment on every input change and rejecting frames for that
            # is right, but a stale browser link must never be able to wedge
            # the transport: it is reported through ``status()`` instead.
            if self.source != BROWSER_SOURCE:
                return False

        sink = self.frame_sink
        if sink is not None:
            try:
                sink(frame, wall_time)
            except Exception as exc:  # noqa: BLE001
                print(f'[live3d] frame sink failed: {exc}', flush=True)
                self.frame_sink = None

        pose = None
        provider = self.pose_provider
        if provider is not None:
            try:
                pose = provider()
            except Exception:  # noqa: BLE001
                pose = None
        preview = self._render(frame, pose)

        now = time.perf_counter()
        with self._lock:
            times = self._push_times
            times.append(now)
            while times and now - times[0] > 1.5:
                times.pop(0)
            fps = (len(times) - 1) / (times[-1] - times[0]) \
                if len(times) > 1 else 0.0

        ok, encoded = cv2.imencode('.jpg', preview,
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
        h, w = frame.shape[:2]
        with self._lock:
            if ok:
                self._jpeg = encoded.tobytes()
                self._seq += 1
            self._status.update(width=w, height=h,
                                fps=round(fps, 1),
                                frames=self._status.get('frames', 0) + 1,
                                media_t=None)
            self._last_push = wall_time
        return True

    # --------------------------------------------------------------- snapshots
    def status(self) -> dict:
        with self._lock:
            status = dict(self._status)
            if self.source == BROWSER_SOURCE and status.get('state') == 'running':
                age = (time.time() - self._last_push) if self._last_push else None
                status['push_age'] = None if age is None else round(age, 2)
                status['stale'] = age is None or age > BROWSER_STALE_S
            return status

    def snapshot(self):
        with self._lock:
            return self._jpeg, self._seq
