# InstantHMR → VRM live

https://github.com/user-attachments/assets/dcb5da61-9789-4b1e-8c17-a1adc4c90960

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
  Face on first use — the [Docker](#docker) image bakes them in instead, so
  that path never touches the network.
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

## Docker

The image is published on Docker Hub as **`camenduru/vrminstanthmr`**. It is
self-contained: **nothing is downloaded when the container runs**. The build
bakes in:

| What | Where |
| --- | --- |
| the app + the three.js bundle | compiled from `web/` in a Node stage |
| InstantHMR checkout | `/opt/InstantHMR`, pinned commit, training trees dropped |
| InstantHMR ONNX graphs | `/opt/hf-cache` — every variant × fit 0/1/2, plus `mhr-body.onnx` |
| RF-DETR detector weights | `/opt/roboflow` — all five variants |
| avatar + idle animation | `/app/vrm_models` |

### Run it from Docker Hub

```bash
docker pull camenduru/vrminstanthmr:latest

# GPU (needs the NVIDIA Container Toolkit)
docker run --rm -it --gpus all -p 8010:8010 --name vrminstanthmr \
  camenduru/vrminstanthmr:latest
```

Open **http://localhost:8010**. The container listens on `0.0.0.0` and shows
`(healthy)` in `docker ps` once `/api/status` answers. Add `-d` to run detached,
then `docker logs -f vrminstanthmr` for the worker's own output; the page's
**InstantHMR 3D** line reads `CUDAExecutionProvider` when the GPU was picked up.
Without a GPU drop `--gpus all` — the same image falls back to the CPU, just
slower.

**A fresh container has no clips.** `/app/videos` ships empty, so the app starts
on `camera:0`; with no `/dev/video0` passed through that camera cannot open and
the bottom-right preview stays blank. Either add a clip in the page
(**Add video…** uploads into the container's `/app/videos` and switches to it,
but `--rm` discards it on exit) or mount a folder:

| Mount | Effect |
| --- | --- |
| `-v "$PWD/videos:/app/videos"` | clips are listed at start, the first is selected, uploads survive |
| `-v "$PWD/vrm_models:/app/vrm_models"` | your own `.vrm` / `.vrma` appear in **Load VRM…** |
| `--device /dev/video0` | a real camera as `camera:0` |

On Windows quote the host path instead: `-v "%cd%\videos:/app/videos"`.

The image's `CMD` already carries `--host 0.0.0.0 --port 8010`, and `docker run`
*replaces* a `CMD` rather than appending to it, so pinning an input at startup
means repeating the command:

```bash
docker run --rm -it --gpus all -p 8010:8010 \
  -v "$PWD/videos:/app/videos" camenduru/vrminstanthmr:latest \
  /opt/app/bin/python -m app.server --host 0.0.0.0 --port 8010 \
  --source "video:clip.mp4"
```

Otherwise just pick the clip in the **Input** menu.

### Build it yourself

```bash
docker build -t camenduru/vrminstanthmr:latest .
docker run --rm -it --gpus all -p 8010:8010 camenduru/vrminstanthmr:latest
```

The name must be lowercase — Docker Hub rejects `camenduru/VRMInstantHMR` as an
invalid reference, which is why every command here uses
`camenduru/vrminstanthmr`.

A full build is **16.2 GB** (context ~250 MB) and takes roughly ten minutes on a
warm cache: `COPY . /app/` sits above the Python environments, the InstantHMR
checkout and the weight downloads, so any source edit redoes those layers. To
trim it, keep only what you will actually select in the UI — the app defaults to
`--model medium --fit 2 --detector medium`, so keep those:

```bash
docker build -t camenduru/vrminstanthmr \
  --build-arg INSTANTHMR_VARIANTS="medium" \
  --build-arg RFDETR_VARIANTS="medium" .
```

### Push

```bash
docker login
docker build -t camenduru/vrminstanthmr:latest .
docker push camenduru/vrminstanthmr:latest
```

Only layers the Hub does not already have are uploaded, so a rebuild that
touches the app code pushes a few layers and a new manifest rather than all
16 GB. Tag a version as well if you want a fixed reference:
`docker tag camenduru/vrminstanthmr:latest camenduru/vrminstanthmr:1.0` and push
both tags.

`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set in the image, so a
weight that failed to bake raises instead of being fetched — the "no runtime
download" guarantee is enforced, not just intended. The build ends with checks
that resolve `medium`/fit2 and construct `RFDETRMedium()` with the network
switched off.

`feet/` and `filters/` are the upstream reference implementations the
retargeter was ported from. They are documentation, never imported at runtime,
and `.dockerignore` keeps their ~212 MB out of the build context.

### In the page

* **Input** – pick a camera or a video. **Add video…** uploads one and switches
  to it. A clip loops.
* **InstantHMR 3D** – starts/stops the live worker; the button shows the
  measured rate once it is running. **Follow movement** is off by default so
  the avatar dances on its mark; enable it to let the avatar step with the
  person, bounded to a small cone. **Stick feet to floor** finds a foot that is
  down and barely moving and holds that sole where it landed with a two-bone
  leg IK, so it stops skating sideways without dragging the whole avatar.
  Height is a separate floor constraint; the global smoothing pass runs
  *after* the plant so the corrected leg cannot shake (see
  [Filters](#filters)).
* **Smoothing** – one global filter bank over the whole rig, built from the
  real-time methods of the reference projects in `filters/` (see
  [Filters](#filters)): **Butterworth**, **One Euro**, **Exponential**,
  **moving average**, **weighted average**, **HDE**, **Kalman (velocity)**,
  **Kalman (adaptive)**, or **Off**. **Smoothing** sets the cutoff — or, for the three windowed methods,
  the window length in frames — and **De-jitter** holds movement smaller than a
  threshold as tracking noise, which is what removes the last stand-still
  shiver. The choice is remembered in the browser.
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
  own standing height, and scaled to this avatar's leg; a slow trim removes the
  remaining bias against the measured **sole** height. The crouch term is only
  a feed-forward estimate, though, and the legs are aimed with fixed bone
  lengths, so the two disagree by centimetres on a fast step. The posed soles
  get the last word: `grounding.js` is a one-sided solve that lifts the root
  whenever a sole would end up below the bind floor, and leaves a real hop
  above it alone. Horizontal travel is the optional **Follow movement**
  checkbox and is confined to a small cone around the spot where tracking
  began, so reconstruction depth noise can never walk the avatar off its mark.
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
  residual is quietened on the way out. That second pass is deliberately AFTER
  the foot plant: the plant's leg IK rewrites rotations on the way out, so a
  filter that ran first would leave every IK correction unsmoothed and the
  corrected legs would shake. Ten filters share the cutoff knob, ported from
  the reference projects in `filters/` (see [Filters](#filters)).
  Quaternions are filtered component-wise on one hemisphere and renormalised,
  which is stable and needs no wrap-around bookkeeping. On top of the filter, a
  deadband holds sub-threshold movement: a low-pass decays towards a still pose
  but never quite reaches one, so the last shiver needs a floor, not a
  smaller cutoff.
* **Foot planting** is split in two: `web/src/footlock.js` only DETECTS
  contact, `web/src/footik.js` holds it. Both take their structure from the
  `feet/` reference projects: `UnderPressure` labels a foot from its height
  over the ground *and* its speed - height alone catches a foot mid-step,
  speed alone catches a foot parked in mid air - with hysteresis so it cannot
  flicker; FootMR cancels the displacement of the joints it believes are
  static and drops the sequence onto its lowest point. Here a foot that is
  down and slow gets a root-relative XZ anchor that is **frozen for the whole
  stance**, eased in and out over 40/120 ms. Three details decide whether that
  does anything at all, and each one on its own measured as a no-op: an anchor
  leashed to the sole leaves the IK nothing to correct, so the foot slides with
  the reconstruction (a 6 cm leash still let a planted foot travel 45 cm);
  clearing the anchor on the first missed frame re-seeds it wherever the foot
  already is, so a 250 ms debounce carries it across the one-frame speed
  spikes; and because the dance clip contains 10-16 cm single-frame tracking
  jumps, the anchor and the step test read a low-passed position while the
  speed gate keeps reading the raw one. A stance ends once the sole has walked
  20 cm from its anchor, which is also what stops a genuine step from being
  fought. Detection reads the NATURAL pose before any IK has touched a bone, so
  a held foot can never hide its own release from the detector.
  `footik.js` then solves the analytic two-bone triangle for that leg and
  holds the sole on the anchor, leaving the shoe's world orientation and the
  root completely alone. That split is the point: the first versions tried to
  cancel skate with a root offset, but with **Follow movement** off the root
  barely moves while the soles still slide ~14 cm per stance - the drift comes
  from reconstructed leg rotations, and moving the one transform shared by
  both feet can only trade foot skate for whole-body sliding. The IK
  correction is low-passed and slew-limited to 0.3 m/s, the knee pole is
  smoothed, and the global filter runs after it; without those the corrected
  legs glitch. The slew limit is what keeps a release from snapping: a 15 cm
  hold that vanished in one frame read as a pop more than twice the foot's own
  motion, and it now unwinds at walking speed instead.
  **Stick feet to floor** in the panel switches it off, and the choice is
  remembered.
* Grounding is on the **soles**, not on the ankle bones. An ankle is not a
  sole: point the toes and the ankle rises while the shoe goes through the
  floor, and every bone still reports a sensible height — which is what "the
  mesh sinks but the bones are fine" looks like. `mapSoles()` samples the
  avatar's skin once, in its bind pose, and keeps the lowest vertex per small
  XZ cell for each foot (attached to the foot and toe bones) - a sole
  *outline*, not just the eight lowest points clustered around one spot. A
  heel or toe that was not lowest in the bind pose can easily be lowest in a
  pitched one, so sampling only the bottom of the bind sole misses the very
  skin that goes through the floor. `grounding.js` measures the lowest posed
  sole and lifts the root by exactly the amount that puts it back on the bind
  floor, never lowering it, so a hop survives. It is a solve rather than an
  integrator: the slow trim arrived a second late and let the skin sit up to
  9 cm under the floor. The constraint runs once on the root target and once
  more as the FINAL write of the frame - after the plant and the output
  filter, since both can rotate a leg and drop a sole by a centimetre or two.
* The preview skeleton is drawn from the worker's 2D landmarks, so no second
  pose model is running.

The worker is independent of the input: switching videos or cameras keeps it
alive and it simply follows the new frames. Stopping **3D live** (or closing
the page) stops it; the server also stops it on shutdown.

## Filters

`web/src/filters.js` is one filter bank applied to the whole rig: the
reconstructed joints and landmarks on the way in, and every bone's finished
local rotation on the way out. The output pass runs after the foot plant's leg
IK so its corrections are filtered too, and the frame closes with the
one-sided floor constraint. In both the 2D and 3D update paths the order is
`floor clamp → plant → global smoothing → floor clamp`; a hop above the floor
is never pulled down. The methods are the real-time ones from the reference
projects in `filters/`:

| Panel | Method | Ported from | Note |
| --- | --- | --- | --- |
| Butterworth | 2nd-order biquad, Q = 1/√2 | `mocap-refinement/filters/BF.py` | the reference runs `filtfilt` over a whole clip, which a live feed cannot do; this is the causal equivalent |
| Exponential | one-pole | `mocap-refinement/filters/EMA.py` | alpha derived from the cutoff rather than a constant |
| One Euro | adaptive one-pole | — | steady while a limb is still, quick when it moves |
| Moving average | uniform window | `mocap-refinement/filters/SMA.py` | |
| Weighted average | triangular window, newest heaviest | `mocap-refinement/filters/WMA.py` | a causal window cannot be symmetric |
| HDE | smoothing plus a trend term | `mocap-refinement/filters/HDE.py` | keeps the reference's trend rate, takes alpha from the cutoff |
| Kalman (velocity) | constant-velocity Kalman | `mocap-refinement/filters/KF1.py` | steady-state gains: β = α²/(2−α) |
| Kalman (adaptive) | velocity Kalman, gain falls on a jump | `FLK/FLK/AKF.py` | AKF's low-confidence branch is a gain floor here, not a hard gate: a limb re-entering frame steps a long way in one sample, and a hard gate would freeze the bone |

One-state Kalman (`KF0.py`) is an exponential filter, so it is not listed
twice. The three-state one (`KF2.py`) is left out because its acceleration
gain grows without bound as the cutoff nears Nyquist at this frame rate: part
of the slider would diverge rather than smooth. `SG.py` smooths a *centred*
window; run causally it has to extrapolate the polynomial to the newest sample,
which measured a 1.1× gain on 15 Hz noise — it would add the shake it exists to
remove. `INT.py` fills missing samples, which this app does not need: a frame
with a missing joint is rejected upstream. `TSVD`, `LRMC`, `NLRMC` and `RPCA` are batch methods that
need the finished clip, and `ODAE`, `UAE`, `UDAE`, `RNN`, `N-euro` and
`motion_rolling_prediction` are trained networks with PyTorch checkpoints.
None of those can run per frame in this page.

The windowed methods are tuned in frames rather than hertz, but the same
**Smoothing** slider drives them: it maps a cutoff to the equivalent window
length, so the panel means one thing either way.

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
