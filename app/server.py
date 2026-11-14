"""FastAPI server for the standalone InstantHMR -> VRM live app.

Serves the built three.js frontend, the VRM assets, an MJPEG preview of the
selected video/camera and a websocket that pushes the newest 22-joint 3D pose.
The only motion source is InstantHMR: one persistent worker reads the capture
and reconstructs the person; the browser retargets those joints onto the VRM.

Run:
    python -m app.server                     # first video, or camera 0
    python -m app.server --source camera:0
    python -m app.server --source "video:clip.mp4" --no-live
"""
from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import live as live_mod
from .capture import BROWSER_SOURCE, Capture

ROOT = Path(__file__).resolve().parent.parent
WEB_DIST = ROOT / 'web' / 'dist'
VRM_DIR = ROOT / 'vrm_models'
VIDEOS_DIR = ROOT / 'videos'
UPLOAD_LIMIT = 512 * 1024 * 1024
VRM_LIMIT = 256 * 1024 * 1024
_boundary = 'frame'
# Changes on every process start: how the page tells "the server restarted and
# forgot my input" apart from "someone else picked another input".
_BOOT_ID = f'{time.time_ns():x}'

# Windows' mimetypes database does not always know .wasm; three.js does not
# need it, but the preview/VRM fetch should not be mislabelled either.
mimetypes.add_type('application/wasm', '.wasm')


def _vrm_assets() -> dict:
    vrm_files = sorted(VRM_DIR.glob('*.vrm'))
    vrma_files = sorted(VRM_DIR.glob('*.vrma'))
    # The newest file wins. A picked avatar is written here with a fresh mtime,
    # so it is what every window loads afterwards - a window opened after the
    # pick included - and a .vrm copied in by hand behaves the same way. The
    # bundled sample is only the default because it is all the image ships.
    preferred = (max(vrm_files, key=lambda p: (p.stat().st_mtime, p.name))
                 if vrm_files else None)

    def entry(path: Path) -> dict:
        stat = path.stat()
        # size as well as mtime: two uploads of the same name inside one
        # second must not share a URL, or the window keeps the old avatar
        return {'name': path.name,
                'url': f'/vrm/{path.name}?v={int(stat.st_mtime)}-{stat.st_size}'}

    return {
        'vrm': entry(preferred) if preferred else None,
        'vrma': entry(vrma_files[0]) if vrma_files else None,
        'vrms': [entry(p) for p in vrm_files],
    }


def _take_browser_frame(capture: Capture, body: bytes) -> str:
    """Decode one pushed JPEG and hand it to the capture.

    Runs off the event loop: a 720p decode plus the preview render and encode
    is milliseconds of CPU, and doing that inline would stall the MJPEG preview
    and the pose websocket for every frame the page sends.

    Returns ``ok``, ``bad`` (not a JPEG) or ``source`` (the input is no longer
    the browser camera, so the page has to re-assert it).
    """
    if not body:
        return 'bad'
    frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return 'bad'
    return 'ok' if capture.push_frame(frame) else 'source'


def create_app(capture: Capture, live: live_mod.LiveInstantHMR,
               start_live: bool = True,
               live_options: dict | None = None) -> FastAPI:
    options = dict(live_options or {})

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        capture.pose_provider = live.pose
        capture.start()
        if start_live:
            ok = live_mod.availability()
            if ok['ok']:
                try:
                    live.start(**options)
                    capture.frame_sink = live.submit
                except Exception as exc:  # noqa: BLE001
                    print(f'[live3d] could not start: {exc}', flush=True)
        try:
            yield
        finally:
            capture.frame_sink = None
            live.stop()
            capture.stop()

    app = FastAPI(title='InstantHMR VRM Live', lifespan=lifespan)

    @app.middleware('http')
    async def revalidate_assets(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == '/' or path.startswith('/vrm/') or path.endswith('.html'):
            # a rebuilt bundle or a swapped avatar must never come from cache
            response.headers['Cache-Control'] = 'no-store'
        return response

    # ------------------------------------------------------------------- state
    @app.get('/api/status')
    async def status():
        return {'capture': capture.status(),
                'live': live.status(),
                'avatar': _vrm_assets(),
                'boot': _BOOT_ID}

    @app.get('/api/sources')
    async def sources():
        return {'current': capture.source,
                'sources': capture.available_sources()}

    @app.post('/api/source')
    async def set_source(payload: dict):
        source = str(payload.get('source', '')).strip()
        try:
            capture.set_source(source)
        except ValueError as exc:
            return JSONResponse({'error': str(exc)}, status_code=400)
        return {'current': capture.source, 'pending': source}

    # ------------------------------------------------------------------- video
    @app.get('/video')
    async def video():
        async def stream():
            last = -1
            idle = 0
            while True:
                jpeg, seq = capture.snapshot()
                if jpeg is not None and seq != last:
                    last = seq
                    idle = 0
                    yield (b'--' + _boundary.encode() + b'\r\n'
                           b'Content-Type: image/jpeg\r\n'
                           b'Content-Length: ' + str(len(jpeg)).encode() +
                           b'\r\n\r\n' + jpeg + b'\r\n')
                else:
                    idle += 1
                await asyncio.sleep(1 / 60 if idle < 120 else 0.25)

        return StreamingResponse(stream(),
                                 media_type='multipart/x-mixed-replace; '
                                            'boundary=' + _boundary)

    @app.post('/api/upload')
    async def upload(request: Request, name: str = 'clip.mp4'):
        """Store a video the user picked, so it appears in the source list."""
        clean = re.sub(r'[^A-Za-z0-9._ -]+', '_', Path(name).name).strip(' .')
        if not clean or Path(clean).suffix.lower() not in {
                '.mp4', '.webm', '.mov', '.mkv', '.avi', '.m4v'}:
            return JSONResponse({'error': 'expected a video file'},
                                status_code=400)
        body = await request.body()
        if not body:
            return JSONResponse({'error': 'empty upload'}, status_code=400)
        if len(body) > UPLOAD_LIMIT:
            return JSONResponse({'error': 'video is too large'},
                                status_code=413)
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        target = VIDEOS_DIR / clean
        target.write_bytes(body)
        return {'source': f'video:{clean}', 'name': clean}

    @app.post('/api/browser-frame')
    async def browser_frame(request: Request):
        """One JPEG frame from the page's webcam (fallback transport).

        The websocket below is what the page prefers; this stays because a
        proxy that blocks websockets must not cost the user the camera.
        """
        if capture.source != BROWSER_SOURCE:
            return JSONResponse({'error': 'browser camera source not active',
                                 'source': capture.source}, status_code=409)
        body = await request.body()
        if not body:
            return JSONResponse({'error': 'empty frame'}, status_code=400)
        result = await asyncio.to_thread(_take_browser_frame, capture, body)
        if result == 'bad':
            return JSONResponse({'error': 'could not decode frame'},
                                status_code=400)
        if result == 'source':
            return JSONResponse({'error': 'browser camera source not active',
                                 'source': capture.source}, status_code=409)
        return {'ok': True}

    @app.websocket('/ws/camera')
    async def ws_camera(ws: WebSocket):
        """The page's webcam as a stream of binary JPEG frames.

        One socket for the whole session instead of a request per frame: no
        per-frame handshake, no half-dead request left behind by a network
        blip, and the idle ``ping`` below is how both ends notice that the
        other one is gone (a restarted container, a dropped proxy, a laptop
        that went to sleep).
        """
        await ws.accept()
        frames = 0
        last_ack = 0.0
        try:
            await ws.send_json({'type': 'ready',
                                'source': capture.source,
                                'active': capture.source == BROWSER_SOURCE})
            while True:
                try:
                    message = await asyncio.wait_for(ws.receive(), timeout=5.0)
                except asyncio.TimeoutError:
                    await ws.send_json({'type': 'ping', 'frames': frames})
                    continue
                if message['type'] == 'websocket.disconnect':
                    return
                body = message.get('bytes')
                if body:
                    if capture.source != BROWSER_SOURCE:
                        await ws.send_json(
                            {'type': 'error',
                             'error': 'browser camera source not active',
                             'source': capture.source})
                        continue
                    result = await asyncio.to_thread(
                        _take_browser_frame, capture, body)
                    frames += 1
                    now = time.monotonic()
                    if result != 'ok':
                        await ws.send_json(
                            {'type': 'error', 'error': result,
                             'source': capture.source})
                    elif now - last_ack >= 0.5:
                        # a cheap ack is what the page's watchdog listens to
                        last_ack = now
                        await ws.send_json({'type': 'ack', 'frames': frames})
                    continue
                text = message.get('text') or ''
                if text == 'ping':
                    await ws.send_json({'type': 'pong', 'frames': frames})
                    continue
                try:
                    payload = json.loads(text) if text else {}
                except ValueError:
                    payload = {}
                if payload.get('type') == 'source':
                    await ws.send_json({'type': 'state',
                                        'source': capture.source,
                                        'active': capture.source == BROWSER_SOURCE,
                                        'frames': frames})
        except WebSocketDisconnect:
            return
        except Exception:  # noqa: BLE001
            return

    # -------------------------------------------------------------- live 3D
    @app.get('/api/live')
    async def live_state():
        data = live.status()
        data['available'] = live_mod.availability()
        data['source'] = capture.source
        return data

    @app.post('/api/live')
    async def live_control(payload: dict):
        action = str(payload.get('action', 'start')).strip().lower()
        if action == 'stop':
            capture.frame_sink = None
            return live.stop()
        if action != 'start':
            return JSONResponse({'error': f'unknown action {action!r}'},
                                status_code=400)
        fit = payload.get('fit', 2)
        stride = payload.get('detector_stride', 2)
        try:
            live.start(
                model=str(payload.get('model', 'medium') or 'medium'),
                fit=2 if fit is None else int(fit),
                device=str(payload.get('device', '') or ''),
                detector=str(payload.get('detector', 'medium') or 'medium'),
                detector_stride=2 if stride is None else int(stride),
                pose_filter=str(payload.get('pose_filter', 'one-euro')
                                or 'one-euro'))
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({'error': str(exc)}, status_code=409)
        capture.frame_sink = live.submit
        return live.status()

    # ------------------------------------------------------------------- avatar
    @app.get('/api/avatar')
    async def avatar():
        assets = _vrm_assets()
        if assets['vrm'] is None:
            return JSONResponse({'error': 'no .vrm in vrm_models'},
                                status_code=404)
        return assets

    @app.post('/api/avatar')
    async def upload_avatar(request: Request, name: str = 'avatar.vrm'):
        """Store the VRM the user picked, so every window loads it from now on.

        The page used to load a picked file from a blob URL: it looked right in
        the window that picked it, and the next window - or a reload - fell back
        to whatever was already on disk. Writing it here makes the pick the
        server's answer, so a clean window, OBS and a second browser agree.
        """
        clean = re.sub(r'[^A-Za-z0-9._ -]+', '_', Path(name).name).strip(' .')
        if not clean or Path(clean).suffix.lower() != '.vrm':
            return JSONResponse({'error': 'expected a .vrm file'},
                                status_code=400)
        body = await request.body()
        if not body:
            return JSONResponse({'error': 'empty upload'}, status_code=400)
        if len(body) > VRM_LIMIT:
            return JSONResponse({'error': 'file too large'}, status_code=413)
        # glTF binary magic: a renamed .png must not become the avatar and then
        # break every window that tries to load it
        if body[:4] != b'glTF':
            return JSONResponse({'error': 'not a VRM/GLB file'},
                                status_code=400)
        VRM_DIR.mkdir(parents=True, exist_ok=True)
        (VRM_DIR / clean).write_bytes(body)
        return {'ok': True, 'avatar': _vrm_assets()['vrm']}

    # --------------------------------------------------------------- websocket
    @app.websocket('/ws/pose')
    async def ws_pose(ws: WebSocket):
        await ws.accept()
        await ws.send_json({'type': 'hello', 'source': capture.source})
        last_pose = -1
        last_frame = -1
        last_sent = time.monotonic()
        try:
            while True:
                pose = live.pose()
                pose_seq = pose['seq'] if pose else 0
                _, frame_seq = capture.snapshot()
                if pose_seq != last_pose or frame_seq != last_frame:
                    last_pose = pose_seq
                    last_frame = frame_seq
                    state = capture.status()
                    message = {'type': 'pose', 'seq': frame_seq,
                               'source': state.get('source', ''),
                               'width': state.get('width', 0),
                               'height': state.get('height', 0)}
                    if pose is not None:
                        message['detected'] = bool(pose.get('detected'))
                        message['live3d'] = {
                            'fps': round(float(pose.get('fps', 0.0)), 1),
                            'ms': round(float(pose.get('latency', 0.0)), 1)}
                        if pose.get('detected') and pose.get('joints3d'):
                            message['joints3d'] = pose['joints3d']
                            # the auxiliary landmarks (nose/ears, elbows, heels,
                            # knuckles) are the only source of bone roll, so they
                            # have to travel with the joints
                            if pose.get('landmarks'):
                                message['landmarks'] = pose['landmarks']
                            # 2D joints + detector box: how much of the body is
                            # inside the frame, which is how a close-up is told
                            # apart from a full-body shot
                            if pose.get('skeleton2d'):
                                message['skeleton2d'] = pose['skeleton2d']
                            if pose.get('box2d'):
                                message['box2d'] = pose['box2d']
                    await ws.send_json(message)
                    last_sent = time.monotonic()
                elif time.monotonic() - last_sent > 5.0:
                    # a quiet source must not let a proxy (or a slept laptop)
                    # drop this socket without either end noticing
                    await ws.send_json({'type': 'tick'})
                    last_sent = time.monotonic()
                await asyncio.sleep(1 / 90)
        except WebSocketDisconnect:
            return
        except Exception:  # noqa: BLE001
            return

    # ------------------------------------------------------------------ static
    VRM_DIR.mkdir(exist_ok=True)
    VIDEOS_DIR.mkdir(exist_ok=True)
    app.mount('/vrm', StaticFiles(directory=VRM_DIR), name='vrm')
    app.mount('/videos', StaticFiles(directory=VIDEOS_DIR), name='videos')

    assets = WEB_DIST / 'assets'
    if assets.exists():
        app.mount('/assets', StaticFiles(directory=assets), name='assets')

    @app.get('/')
    async def index():
        page = WEB_DIST / 'index.html'
        if not page.exists():
            return JSONResponse(
                {'error': 'web/dist is missing - run "npm run build" in web'},
                status_code=503)
        return FileResponse(page)

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='',
                        help='camera:0 or video:<file name> (default: first '
                             'video in videos/, else camera:0)')
    parser.add_argument('--videos', action='append', default=[],
                        metavar='DIR',
                        help='extra folder to list videos from (repeatable); '
                             'the sibling rtmlib/videos is added automatically '
                             'when present')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8010)
    parser.add_argument('--no-live', action='store_true',
                        help='do not start InstantHMR on launch')
    parser.add_argument('--model', default='medium')
    parser.add_argument('--fit', type=int, default=2, choices=(0, 1, 2))
    parser.add_argument('--detector', default='medium')
    parser.add_argument('--detector-stride', type=int, default=2)
    args = parser.parse_args()

    dirs = [VIDEOS_DIR] + [Path(p) for p in args.videos]
    sibling = ROOT.parent / 'videos'
    if sibling.exists() and sibling not in dirs:
        dirs.append(sibling)

    capture = Capture(dirs)
    live = live_mod.LiveInstantHMR()

    chosen = args.source.strip()
    if not chosen:
        videos = [s['id'] for s in capture.available_sources()
                  if s['kind'] == 'video']
        chosen = videos[0] if videos else 'camera:0'
    capture.set_source(chosen)

    app = create_app(capture, live, start_live=not args.no_live,
                     live_options={
                         'model': args.model, 'fit': args.fit,
                         'detector': args.detector,
                         'detector_stride': args.detector_stride,
                     })
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')


if __name__ == '__main__':
    main()
