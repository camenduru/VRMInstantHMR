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
* **Smoothing** – one global filter bank over the whole rig. **Butterworth**
  (2nd-order, the default: the flattest passband for the lag it costs),
  **One Euro** (adaptive: steady while a limb is still, quick when it moves) or
  **Exponential** (plain single-pole), or **Off**. **Smoothing** sets the
  cutoff, **De-jitter** holds movement smaller than a threshold as tracking
  noise, which is what removes the last stand-still shiver. The choice is
  remembered in the browser.
* **Avatar** – **Load VRM…** swaps the avatar from disk, locally.
* **OBS output** – **Chroma key** puts a flat key colour behind the avatar and
  takes the floor, grid, ring and blob shadow out of the shot. The two swatches
  are the usual keys — green `#00FF00` and blue `#0000FF` — and the picker
  beside them takes any colour; the choice is remembered. **Clean window**
  opens the same view with all the controls hidden.
* Drag to orbit, right-drag to pan, wheel to zoom. The bottom-right preview
  shows the input with the reconstructed 22-joint skeleton on it.

### OBS / green screen

For a stream or a recording you usually want only the character, on a field a
chroma key can remove:

* **Browser source (cleanest).** Add `http://127.0.0.1:8010/?obs=1` as an OBS
  browser source at your stream resolution, then add a **Chroma Key** filter
  whose key colour matches the swatch — the default `#00FF00`, or `#0000FF` if
  the character itself wears green — with similarity ~400 and smoothness ~80.
  The canvas follows the source size, so the avatar reframes itself if you
  change the resolution. The chosen colour is remembered in the browser, so a
  browser source can be set up once.
* **Window capture.** Click **Clean window** (or open `?obs=1` in a tab) and
  capture that window: no panels, no preview, no furniture.

In a normal window `G` toggles the key field and `H` (or `Esc`) hides and
shows the controls, so the exact capture layout can be previewed without
reloading. `?green=1` turns the key field on and keeps the controls; the
**Chroma key** checkbox does the same thing.

The background is a WebGL clear colour, which is written to the framebuffer
after the output colour-space conversion but before tone mapping, so the pixels
really are the picked colour rather than a tone-mapped approximation. The mint
rim light turns neutral white and the pink kicker goes off in this mode: they
look good on a dark stage but leave coloured fringes along the silhouette that
the key would eat.

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
* **Close-ups and shoulder-height shots** are handled separately, because a
  camera that cuts the body at the waist leaves the reconstruction no legs and
  no floor. It fills the gap with a guess — hips dropped, thighs swung forward,
  torso pitched away from the viewer — which used to retarget as a crouching
  avatar whose feet hovered in mid air, however upright the person actually
  was. The worker already sends the 2D projection of every joint, so the page
  can tell what the camera really saw: a leg whose knee, ankle and toes all
  project outside the image was never observed. Such a leg is blended to a
  neutral straight stance under its hip and the root stays at standing height,
  and the invented depth pitch of the torso is removed by rotating the
  skeleton about the pelvis until the shoulder line is upright in the image
  (left-right lean, which the image *can* show, is kept). The weights are
  smoothed, so a leg walking back into shot takes over again over a fraction of
  a second. Nothing of this runs while both feet are in frame, so a full-body
  dance keeps its real lean and crouches.
* **Arms cut by the frame** get the same treatment, segment by segment. A
  waist-up shot also hides the arms, and the reconstruction does not leave them
  out — it invents an elbow and a wrist that wander from frame to frame, which
  reads as the model's arms shaking. Shoulder + elbow must both project inside
  the image for an upper arm to count, elbow + wrist for a forearm; a hidden
  upper arm is blended to a relaxed hang under its own shoulder, a hidden
  forearm continues the upper arm, and the hand continues the forearm. The
  elbow-crease landmarks that carry the arm's roll are also ignored when the
  arm they belong to is not in shot, so the roll cannot spin either. Unlike the
  legs, this is per arm: one arm in frame is trusted while the other hangs.
* `web/src/filters.js` is the global smoothing pass, and it runs in two places
  for one reason: noise that has already become a bone rotation cannot be
  removed without also lagging the pose. The reconstructed joints and the
  auxiliary landmarks are low-passed in camera space *before* any bone is
  solved, the root translation with them, and every bone's finished local
  rotation once afterwards — so a bad frame is quietened on the way in and the
  residual is quietened on the way out. Three filters share the cutoff knob: a
  2nd-order Butterworth biquad (Q = 1/√2), One Euro, and a plain exponential.
  Quaternions are filtered component-wise on one hemisphere and renormalised,
  which is stable and needs no wrap-around bookkeeping. On top of the filter, a
  deadband holds sub-threshold movement: a low-pass decays towards a still pose
  but never quite reaches one, so the last shiver needs a floor, not a
  smaller cutoff.
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
