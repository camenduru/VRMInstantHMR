"""Checks for the standalone InstantHMR -> VRM app.

Plain asserts, no pytest - run with the app's interpreter:

    ..\\.venv\\Scripts\\python.exe tests\\test_app.py

InstantHMR itself is a separate environment that loads its graphs on first
use, so these cover the parts that must be right regardless: the MHR->SMPL
mapping (3D for the retargeter, 2D for the preview), the frame pipe, source
resolution, the live worker's reply handling and the HTTP surface.
"""
from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import live as live_mod  # noqa: E402
from app.capture import Capture  # noqa: E402

results = []


def check(name):
    def wrap(fn):
        try:
            fn()
            results.append(f'  PASS  {name}')
        except AssertionError as exc:
            results.append(f'  FAIL  {name}\n        {exc}')
            globals()['FAILED'] = True
        return fn
    return wrap


FAILED = False


def t_pose(dims: int = 3) -> np.ndarray:
    """A (70, dims) MHR frame with every landmark at a distinguishable place."""
    points = np.zeros((live_mod.MHR_JOINTS, dims), np.float32)
    points[0] = [0.0, 2.3, 0.1][:dims]            # nose
    points[1] = [0.1, 2.35, 0.1][:dims]           # left eye
    points[2] = [-0.1, 2.35, 0.1][:dims]          # right eye
    points[3] = [0.15, 2.3, 0.0][:dims]           # left ear
    points[4] = [-0.15, 2.3, 0.0][:dims]          # right ear
    points[live_mod.MHR['left_hip']] = [1.0, 0.0, 0.0][:dims]
    points[live_mod.MHR['right_hip']] = [-1.0, 0.0, 0.0][:dims]
    points[live_mod.MHR['neck']] = [0.0, 2.0, 0.0][:dims]
    points[live_mod.MHR['left_shoulder']] = [1.2, 1.8, 0.0][:dims]
    points[live_mod.MHR['right_shoulder']] = [-1.2, 1.8, 0.0][:dims]
    points[live_mod.MHR['left_elbow']] = [1.6, 1.2, 0.0][:dims]
    points[live_mod.MHR['right_elbow']] = [-1.6, 1.2, 0.0][:dims]
    points[live_mod.MHR['left_wrist']] = [2.0, 0.6, 0.0][:dims]
    points[live_mod.MHR['right_wrist']] = [-2.0, 0.6, 0.0][:dims]
    points[live_mod.MHR['left_knee']] = [0.9, -1.0, 0.0][:dims]
    points[live_mod.MHR['right_knee']] = [-0.9, -1.0, 0.0][:dims]
    points[live_mod.MHR['left_ankle']] = [0.8, -2.0, 0.0][:dims]
    points[live_mod.MHR['right_ankle']] = [-0.8, -2.0, 0.0][:dims]
    points[live_mod.MHR['left_big_toe']] = [0.8, -2.1, 0.4][:dims]
    points[live_mod.MHR['left_small_toe']] = [0.7, -2.1, 0.5][:dims]
    points[live_mod.MHR['right_big_toe']] = [-0.8, -2.1, 0.4][:dims]
    points[live_mod.MHR['right_small_toe']] = [-0.7, -2.1, 0.5][:dims]
    return points


@check('MHR70 landmarks land on the 22 SMPL-ordered body joints')
def _():
    body = live_mod.mhr_to_smpl(t_pose())
    assert body.shape == (22, 3), body.shape
    assert np.allclose(body[0], [0.0, 0.0, 0.0]), 'pelvis is the hip midpoint'
    assert np.allclose(body[1], [1.0, 0.0, 0.0])
    assert np.allclose(body[2], [-1.0, 0.0, 0.0])
    assert np.allclose(body[12], [0.0, 2.0, 0.0]), 'neck is measured'
    assert np.allclose(body[16], [1.2, 1.8, 0.0])
    assert np.allclose(body[20], [2.0, 0.6, 0.0])
    assert np.allclose(body[10][:2], [0.75, -2.1]), 'toes are averaged'
    assert abs(body[15][1] - 2.32) < 1e-4, 'head is the face-landmark centre'
    # the spine chain walks hips -> shoulders, evenly
    assert np.allclose(body[3], [0.0, 0.45, 0.0]), body[3]
    assert np.allclose(body[6], [0.0, 0.9, 0.0]), body[6]
    assert np.allclose(body[9], [0.0, 1.35, 0.0]), body[9]


@check('the same mapping serves the 2D preview skeleton')
def _():
    screen = live_mod.mhr_to_smpl(t_pose(2))
    assert screen.shape == (22, 2), screen.shape
    assert np.allclose(screen[16], [1.2, 1.8]), screen[16]


@check('a wrong MHR shape is rejected')
def _():
    for bad in (np.zeros((10, 3), np.float32), np.zeros((70, 4), np.float32),
                np.zeros(70, np.float32)):
        try:
            live_mod.mhr_to_smpl(bad)
        except ValueError:
            continue
        raise AssertionError(f'{bad.shape} was accepted')


@check('auxiliary landmarks carry the roll references')
def _():
    points = t_pose()
    points[63] = [0.4, 1.2, -0.2]           # left olecranon
    points[64] = [0.1, 1.2, -0.2]           # right olecranon
    points[65] = [0.5, 1.2, 0.1]            # left cubital fossa
    points[66] = [0.2, 1.2, 0.1]            # right cubital fossa
    points[17] = [0.9, -2.2, -0.3]          # left heel
    marks = live_mod.mhr_landmarks(points)
    assert set(marks) == set(live_mod.LANDMARKS), set(marks)
    assert np.allclose(marks['right_olecranon'], [0.1, 1.2, -0.2])
    assert np.allclose(marks['right_cubital'], [0.2, 1.2, 0.1])
    assert np.allclose(marks['left_heel'], [0.9, -2.2, -0.3])
    assert all(0 <= i < live_mod.MHR_JOINTS
               for i in live_mod.LANDMARKS.values())
    # the MHR70 order is left, right, left, right for the elbow landmarks
    assert (live_mod.LANDMARKS['left_olecranon'],
            live_mod.LANDMARKS['right_olecranon'],
            live_mod.LANDMARKS['left_cubital'],
            live_mod.LANDMARKS['right_cubital']) == (63, 64, 65, 66)
    assert (live_mod.LANDMARKS['left_heel'],
            live_mod.LANDMARKS['right_heel']) == (17, 20)
    assert live_mod.mhr_landmarks(np.zeros((70, 2), np.float32)) == {}, \
        '2D landmarks have no roll information'

    live = live_mod.LiveInstantHMR()
    live._handle({'ready': True, 'provider': 'CUDAExecutionProvider'})
    live._sent[1] = (live_mod.time.perf_counter(), (640, 360))
    live._inflight = 1
    live._handle({'seq': 1, 'detected': True, 'ms': 1.0, 'w': 640, 'h': 360,
                  'joints': points.tolist(),
                  'joints2d': np.zeros((70, 2), np.float32).tolist()})
    assert 'landmarks' in live.pose(), 'the retargeter needs the roll refs'


@check('a capture frame reaches the worker as a small RGB image')
def _():
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[:] = (10, 20, 30)                     # BGR
    rgb = live_mod._frame_rgb(frame)
    assert rgb.shape == (360, 640, 3), rgb.shape
    assert rgb.flags['C_CONTIGUOUS']
    assert rgb[0, 0].tolist() == [30, 20, 10], 'BGR must arrive as RGB'


@check('a live reply becomes the retargeter payload and a preview skeleton')
def _():
    live = live_mod.LiveInstantHMR()
    assert live.status()['state'] == 'off'
    assert live.submit(np.zeros((4, 4, 3), np.uint8)) is False, \
        'frames are not queued before the worker is running'

    live._handle({'ready': True, 'provider': 'CUDAExecutionProvider'})
    status = live.status()
    assert status['state'] == 'running' and status['ready'] is True
    assert status['provider'] == 'CUDAExecutionProvider'

    joints2d = np.zeros((70, 2), np.float32)
    joints2d[:] = (320.0, 180.0)
    live._sent[1] = (live_mod.time.perf_counter(), (640, 360))
    live._inflight = 1
    live._handle({'seq': 1, 'detected': True, 'ms': 1.0, 'w': 640,
                  'h': 360, 'joints': t_pose().tolist(),
                  'joints2d': joints2d.tolist(),
                  'box': [64.0, 72.0, 576.0, 288.0]})
    pose = live.pose()
    assert pose['detected'] is True
    assert pose['joints3d'] == np.round(
        live_mod.mhr_to_smpl(t_pose()), 4).tolist()
    skeleton = np.asarray(pose['skeleton2d'])
    assert skeleton.shape == (22, 2), skeleton.shape
    assert np.allclose(skeleton, 0.5, atol=1e-4), skeleton[:3]
    box = np.asarray(pose['box2d'])
    assert box.shape == (4,), box.shape
    assert np.allclose(box, (0.1, 0.2, 0.9, 0.8), atol=1e-4), box
    assert live._inflight == 0, 'the reply frees the pipe for the next frame'
    assert live.status()['frames'] == 1

    live._handle({'seq': 2, 'detected': False, 'ms': 1.0})
    pose = live.pose()
    assert pose['detected'] is False
    assert 'joints3d' not in pose, 'a miss must not publish stale joints'
    assert 'skeleton2d' not in pose
    assert 'box2d' not in pose


@check('a worker that dies while loading reports the failure instead of hanging')
def _():
    live = live_mod.LiveInstantHMR()
    live._status['state'] = 'loading'
    live._started = live_mod.time.time()

    class FakeProc:
        def __init__(self, lines):
            self.stdout = io.BytesIO(lines)

        def kill(self):
            pass

    live._read_loop(FakeProc(
        b'ImportError: DLL load failed while importing onnxruntime\n'))
    status = live.status()
    assert status['state'] == 'error', status['state']
    assert 'exited before it became ready' in status['error'], status['error']
    assert 'DLL load failed' in status['error'], status['error']

    live = live_mod.LiveInstantHMR()
    live._status['state'] = 'loading'
    live._started = live_mod.time.time()
    assert live.status().get('loading_s') is not None


@check('the live wire format is the one both sides pack')
def _():
    spec = importlib.util.spec_from_file_location(
        'instanthmr_live', ROOT / 'scripts' / 'instanthmr_live.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    header = worker.HEADER.pack(640, 360, 7)
    assert worker.HEADER.size == 16, worker.HEADER.size
    assert live_mod.struct.unpack('<IIQ', header) == (640, 360, 7)
    assert worker.read_exact(io.BytesIO(b'abc'), 3) == b'abc'
    assert worker.read_exact(io.BytesIO(b''), 1) is None


@check('sources list cameras and videos and refuse path escapes')
def _():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / 'clip one.mp4').write_bytes(b'')
        (folder / 'notes.txt').write_text('x')
        (folder.parent / 'outside.mp4').write_bytes(b'')
        capture = Capture([folder])
        items = capture.available_sources()
        ids = [item['id'] for item in items]
        assert 'camera:0' in ids and 'camera:3' in ids
        assert 'video:clip one.mp4' in ids, ids
        assert 'video:notes.txt' not in ids
        assert capture.resolve('video:clip one.mp4') == (
            folder / 'clip one.mp4').resolve()
        assert capture.resolve('video:../outside.mp4') is None
        assert capture.resolve('camera:0') is None
        try:
            capture.set_source('video:missing.mp4')
        except ValueError:
            pass
        else:
            raise AssertionError('a missing video was accepted')


@check('the preview shows the raw frame and puts the joints back on it')
def _():
    # the frame keeps its own left and right: right half bright, left half dark
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[:, 640:] = 255
    preview = Capture._render(frame, None)
    assert preview.shape == (360, 640, 3), preview.shape
    assert preview[:, :300].max() < 40, preview[:, :300].max()
    assert preview[:, 340:].min() > 200, preview[:, 340:].min()

    # the joints were measured in the mirrored frame the model saw, so a joint
    # at x=0.25 there belongs at x=0.75 of the raw preview
    dark = np.zeros((720, 1280, 3), np.uint8)
    joints = [[0.25, 0.5]] * 22
    preview = Capture._render(
        dark, {'detected': True, 'skeleton2d': joints})
    spot = preview[176:185, 476:485].astype(np.int32).sum(axis=2)
    assert spot.max() > 500, spot.max()
    other = preview[176:185, 156:165].astype(np.int32).sum(axis=2)
    assert other.max() == 0, 'the skeleton must not stay in the model frame'

    flat = Capture._render(dark, None)
    assert flat.shape == (360, 640, 3)
    assert flat.max() == 0, 'no pose must leave the frame alone'


@check('the model gets a mirrored frame while the preview does not')
def _():
    capture = Capture([])
    with capture._lock:
        capture.source = 'browser:camera'
    seen = []
    capture.frame_sink = lambda f, wall: seen.append(f.copy())

    # left half bright, right half dark
    frame = np.zeros((48, 64, 3), np.uint8)
    frame[:, :32] = 255
    assert capture.push_frame(frame) is True
    assert len(seen) == 1, 'the sink must be called once'
    fed = seen[0]
    assert fed.shape == frame.shape
    assert fed[:, :24].max() == 0, 'the model frame is not mirrored'
    assert fed[:, 40:].min() == 255, 'the model frame is not mirrored'

    # the browser preview is encoded from the untouched frame
    jpeg, _ = capture.snapshot()
    assert jpeg
    shown = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert shown is not None
    assert shown[:, :20].min() > 200, 'the preview got mirrored too'
    assert shown[:, 44:].max() < 40, 'the preview got mirrored too'


@check('the websocket carries the joints and their roll landmarks')
def _():
    from fastapi.testclient import TestClient
    from app.server import create_app

    with tempfile.TemporaryDirectory() as tmp:
        capture = Capture([Path(tmp)])
        live = live_mod.LiveInstantHMR()
        app = create_app(capture, live, start_live=False)
        points = t_pose()
        live._pose = {
            'seq': 3, 'detected': True, 'fps': 12.0, 'latency': 20.0,
            'joints3d': live_mod.mhr_to_smpl(points).round(4).tolist(),
            'landmarks': live_mod.mhr_landmarks(points),
            'skeleton2d': np.full((22, 2), 0.5, np.float32).tolist(),
            'box2d': [0.1, 0.2, 0.9, 0.8],
        }
        with TestClient(app) as client:
            with client.websocket_connect('/ws/pose') as ws:
                assert ws.receive_json()['type'] == 'hello'
                message = ws.receive_json()
                assert message['type'] == 'pose', message
                assert message['joints3d'], 'the avatar needs the joints'
                assert message['landmarks'], \
                    'without the landmarks the bones get no roll'
                assert len(message['skeleton2d']) == 22, \
                    'the frame crop is judged from the 2D joints'
                assert message['box2d'] == [0.1, 0.2, 0.9, 0.8], \
                    'the box tells the retargeter where the frame cuts the body'


@check('the HTTP surface serves status, sources and the avatar')
def _():
    from fastapi.testclient import TestClient
    from app.server import create_app

    with tempfile.TemporaryDirectory() as tmp:
        capture = Capture([Path(tmp)])
        live = live_mod.LiveInstantHMR()
        app = create_app(capture, live, start_live=False)
        with TestClient(app) as client:
            status = client.get('/api/status').json()
            assert status['live']['state'] == 'off', status['live']
            assert 'capture' in status
            sources = client.get('/api/sources').json()
            assert any(s['id'] == 'camera:0' for s in sources['sources'])
            bad = client.post('/api/source', json={'source': 'nope'})
            assert bad.status_code == 400, bad.status_code
            avatar = client.get('/api/avatar').json()
            assert avatar['vrm']['name'] == 'AvatarSample_B.vrm', avatar
            assert client.get('/api/live').json()['state'] == 'off'


@check('the page offers a green-screen capture layout for OBS')
def _():
    from fastapi.testclient import TestClient
    from app.server import create_app

    with tempfile.TemporaryDirectory() as tmp:
        capture = Capture([Path(tmp)])
        live = live_mod.LiveInstantHMR()
        app = create_app(capture, live, start_live=False)
        with TestClient(app) as client:
            page = client.get('/')
            assert page.status_code == 200, page.status_code
            html = page.text
            assert 'id="chroma"' in html, 'no green-screen toggle'
            assert 'id="btn-obs"' in html, 'no clean-window link'
            assert 'id="key-green"' in html and 'id="key-blue"' in html, \
                'no green/blue key presets'
            assert 'id="chroma-color"' in html, 'no custom key colour'
            assert '/assets/index-' in html and '.js' in html, \
                'the page is not the built bundle'


@check('a waist-up shot pins the legs instead of crouching the avatar')
def _():
    from fastapi.testclient import TestClient
    from app.server import create_app

    with tempfile.TemporaryDirectory() as tmp:
        capture = Capture([Path(tmp)])
        live = live_mod.LiveInstantHMR()
        app = create_app(capture, live, start_live=False)
        with TestClient(app) as client:
            html = client.get('/').text
            ref = html.split('/assets/')[1].split('"')[0]
            js = client.get(f'/assets/{ref}').text
            assert '_cropWeights' in js, \
                'the bundle predates the close-up leg gate'
            assert '_legSeen' in js, \
                'nothing carries the per-leg visibility into the retargeter'
            assert '_armSeen' in js, \
                'nothing carries the per-arm visibility into the retargeter'


print('\n'.join(results))
if FAILED:
    sys.exit(1)
print('\nInstantHMR app: ok')
