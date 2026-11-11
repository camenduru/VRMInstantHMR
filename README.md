# InstantHMR → VRM live

A small standalone app that turns a **video or a camera** into a **VRM avatar
driven in real time** by [InstantHMR](https://github.com/momolesang/InstantHMR)
3D body reconstruction. There is no bake step, no audio and no 2D fallback
tracker: the only motion source is the live 3D worker.

This is the focused sibling of the full `rtmlib` app: same browser retargeter
and avatar look, but only input → realtime VRM.

## What you need

* Windows, Python 3.10+ with `fastapi`, `uvicorn`, `numpy`, `opencv-python`.
  The parent repository's `.venv` already has them, so the launcher reuses it.
* An InstantHMR checkout with its own environment (`InstantHMR/.venv`), as
  described in `InstantHMR/README.md`. Its ONNX graphs download from Hugging
  Face on first use.
* Node 18+ only if you rebuild the web frontend (a built `web/dist` ships
  alongside it).

The InstantHMR location is configurable:

| Variable | Default |
| --- | --- |
| `INSTANTHMR_DIR` | `../InstantHMR` (sibling of this app) |
| `INSTANTHMR_PYTHON` | `InstantHMR/.venv` interpreter, or the `.instanthmr_python` marker |

## Run

```bat
cd InstantHMRApp
run.bat
```

or explicitly:

```bat
cd InstantHMRApp
..\.venv\Scripts\python.exe -m app.server
```

Open **http://127.0.0.1:8010**.

The app starts on the first video it knows about (or camera 0) and launches
InstantHMR automatically; the avatar idles until the first reconstructed frame
arrives (model loading takes a few seconds, longer on a cold cache).

### Options

```
--source camera:0            camera 0 (or camera:1 …)
--source "video:clip.mp4"    a clip from videos/ (see below)
--no-live                    open the UI without starting InstantHMR
--videos DIR                 extra folder to list clips from (repeatable)
--host / --port              default 127.0.0.1:8010
--model / --fit              InstantHMR model (default medium / fit 2)
--detector / --detector-stride
                             person detector (default medium, every 2nd frame)
```

Videos come from `InstantHMRApp/videos/` (uploads land there) and from any
`--videos` folder. When this app sits next to the full `rtmlib` checkout, its
`videos/` folder is listed too, so the same test clips are available.

### In the page

* **Input** – pick a camera or a video. **Add video…** uploads one and switches
  to it. A clip loops.
* **InstantHMR 3D** – starts/stops the live worker; the button shows the
  measured rate once it is running. **Follow movement** is off by default so
  the avatar dances on its mark; enable it to let the avatar step with the
  person, bounded to a small cone.
* **Avatar** – **Load VRM…** swaps the avatar from disk, locally.
* Drag to orbit, right-drag to pan, wheel to zoom. The bottom-left preview
  shows the input with the reconstructed 22-joint skeleton on it.

## How it works

```
video/camera ──> app/capture.py ──> LiveInstantHMR (child process) ──> /ws/pose ──> three.js VRM
                      │                    │
                      └── MJPEG preview <──┘  (2D landmark overlay)
```

* `app/capture.py` owns one input at a time and runs in a thread. It hands the
  **unmirrored** frame to the worker (the 3D model expects the real
  orientation) and builds the mirrored MJPEG preview.
* `app/live.py` keeps one InstantHMR process alive (`scripts/instanthmr_live.py`
  in InstantHMR's Python) and streams frames over a small binary protocol.
  Only one frame is in flight; the newest capture replaces the queued one, so
  a slow inference drops frames instead of adding latency. `mhr_to_smpl` maps
  the 70 MHR landmarks onto the 22-joint SMPL layout the browser retargets.
* `web/src/mocap.js` retargets the reconstructed skeleton bone by bone. The
  22 joints go through one **fixed** camera → avatar mapping (camera
  x-right / y-down / z-forward become the avatar's mirrored x / up / facing),
  so when the person turns the avatar turns with them and faces the same way;
  the frame is never rebuilt from the body, which would cancel every turn.
  Each bone is then solved in full: its measured direction *and* an anatomical
  roll reference — the shoulder line against the hip line for the torso twist,
  nose vs ears for the head, the point of the elbow vs the elbow crease for arm
  twist, the heel for the sole — so roll, twist and facing transfer instead of
  just an aim.
* The elbow landmarks are only ~6 cm apart, so reconstruction depth error can
  leave part of that line along the arm, which an elbow never does. The measured
  arm twist is weighted by how much of the line is perpendicular to the forearm
  and otherwise eases back to "the back of the arm faces away from the chest",
  so a noisy frame steadies the arm instead of shaking it.
* The root carries the centre of gravity. Vertically it always follows the
  person's hips: the legs are aimed independently, so a root that kept its
  height would let a crouch fold the legs and leave the feet hanging in the
  air. A crouch is measured as the hips dropping towards the support foot (the
  lower ankle, so a kick never lifts the body), expressed against the person's
  own standing height, and scaled to this avatar's leg; a slow trim then sinks
  the root until the support ankle really is back at its bind height, so the
  soles stay on the floor. Horizontal travel is the optional **Follow
  movement** checkbox and is confined to a small cone around the spot where
  tracking began, so reconstruction depth noise can never walk the avatar off
  its mark.
* The preview skeleton is drawn from the worker's 2D landmarks, so no second
  pose model is running.

The worker is independent of the input: switching videos or cameras keeps it
alive and it simply follows the new frames. Stopping **3D live** (or closing
the page) stops it; the server also stops it on shutdown.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/status` | capture + worker state, avatar URLs |
| `GET` | `/api/sources` | cameras and videos available |
| `POST` | `/api/source` | `{"source": "video:clip.mp4"}` / `{"source": "camera:0"}` |
| `GET` | `/api/live` | worker state, fps, inference ms, dropped frames |
| `POST` | `/api/live` | `{"action": "start"}` / `{"action": "stop"}` (optional `model`, `fit`, `detector`, `detector_stride`, `device`) |
| `POST` | `/api/upload?name=clip.mp4` | store a video in `videos/` (raw body) |
| `GET` | `/api/avatar` | default `.vrm` / idle `.vrma` URLs |
| `GET` | `/video` | MJPEG preview with the live skeleton |
| `WS` | `/ws/pose` | `{type: "pose", joints3d: [[x,y,z] × 22], landmarks: {nose, ears, elbows, heels, knuckles}, detected, live3d}` |

## Rebuilding the frontend

```bat
cd InstantHMRApp\web
npm install
npm run build
```

## Tests

```bat
cd InstantHMRApp
..\.venv\Scripts\python.exe tests\test_app.py
```

Covers the MHR→SMPL mapping (3D and 2D), the frame pipe, source resolution and
path-escape refusal, the worker's reply handling, the mirroring preview and the
HTTP surface. InstantHMR's own inference needs its environment and models, so
that part is exercised by running the app.
