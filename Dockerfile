# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# camenduru/vrminstanthmr - fully self-contained image.
# Published on Docker Hub; the name must be lowercase.
#
# Everything is baked in; nothing is downloaded at runtime:
#   * the app + the three.js bundle, compiled here from web/
#   * InstantHMR checkout (pinned commit) + its Python 3.12 environment
#   * InstantHMR ONNX graphs, every variant and every kinematic-fit graph,
#     pre-pulled into the Hugging Face cache                (~1.0 GB)
#   * RF-DETR detector weights, all five variants          (~1.8 GB)
#   * vrm_models/  avatar + idle animation
#   * two Python environments: /opt/app (fastapi) and /opt/instanthmr (torch)
#
# HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are set in the final image, so a
# missing weight fails loudly instead of silently reaching for the network.
#
# Run the published image (UI on http://localhost:8010):
#   docker pull camenduru/vrminstanthmr:latest
#   docker run --rm -it --gpus all -p 8010:8010 \
#     -v "$PWD/videos:/app/videos" camenduru/vrminstanthmr:latest
#
# Build (from this repository's root):
#   docker build -t camenduru/vrminstanthmr:latest .
#
# Smaller build, default detector + default model only:
#   docker build -t camenduru/vrminstanthmr \
#     --build-arg INSTANTHMR_VARIANTS="medium" \
#     --build-arg RFDETR_VARIANTS="medium" .
#
# Run (GPU, UI on http://localhost:8010):
#   docker run --rm -it --gpus all -p 8010:8010 camenduru/vrminstanthmr:latest
#
# Run on CPU (slower, no --gpus):
#   docker run --rm -it -p 8010:8010 camenduru/vrminstanthmr:latest
#
# Camera input needs a device passthrough:
#   docker run --rm -it --gpus all -p 8010:8010 --device /dev/video0 \
#     camenduru/vrminstanthmr:latest
#
# Push:
#   docker push camenduru/vrminstanthmr:latest
#
# Expect a 16.2 GB image with every model; the trimmed build is smaller.
# ---------------------------------------------------------------------------

# ---- frontend: build the three.js bundle from source ----------------------
FROM node:22-alpine AS frontend
WORKDIR /src
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---- runtime --------------------------------------------------------------
# CUDA 12.8 + cuDNN 9 matches onnxruntime-gpu 1.26 and the cu126 torch wheel;
# Ubuntu 24.04 ships Python 3.12, which is what InstantHMR's env was built on.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG INSTANTHMR_REPO=https://github.com/Naturalpad-studio/InstantHMR
# Pinned to the commit this app was developed and measured against.
ARG INSTANTHMR_REF=6b187eada5cf1deac6a2eaa1a85edcad7bfba660
# A space-separated list trims the image; the app's default model is `medium`
# and its default detector is `medium`, so keep those in any trimmed build.
ARG INSTANTHMR_VARIANTS="nano tiny small medium medium-commercial"
ARG RFDETR_VARIANTS="nano small medium base large"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    INSTANTHMR_DIR=/opt/InstantHMR \
    INSTANTHMR_PYTHON=/opt/instanthmr/bin/python \
    INSTANTHMR_VARIANTS=${INSTANTHMR_VARIANTS} \
    RFDETR_VARIANTS=${RFDETR_VARIANTS} \
    PYTHONPATH=/opt/InstantHMR \
    HF_HOME=/opt/hf-cache \
    ROBOFLOW_HOME=/opt/roboflow

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
        ffmpeg \
        libgl1 libglib2.0-0 \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- InstantHMR checkout --------------------------------------------------
# It lives in its own repository, so it is fetched here and pinned. Fetching
# just the one commit keeps the layer small; the training and dataset trees are
# not needed to run live inference, so they are dropped.
RUN git init -q /opt/InstantHMR \
    && cd /opt/InstantHMR \
    && git remote add origin "$INSTANTHMR_REPO" \
    && { git fetch -q --depth 1 origin "$INSTANTHMR_REF" \
         && git checkout -q FETCH_HEAD \
         || { echo "pinned commit unavailable, falling back to the default branch"; \
              git fetch -q --depth 1 origin HEAD && git checkout -q FETCH_HEAD; }; } \
    && rm -rf .git datasets_pipeline training benchmark tools \
    && test -f instanthmr/pipeline.py

# ---- app code -------------------------------------------------------------
WORKDIR /app
COPY . /app/
# always use the bundle built in the stage above
COPY --from=frontend /src/dist /app/web/dist
RUN mkdir -p /app/videos

# ---- app environment ------------------------------------------------------
# The web server only needs fastapi + opencv. requirements.txt asks for the
# GUI opencv build; the headless one is the same library without an X11
# dependency, so it is swapped in.
RUN python3.12 -m venv /opt/app \
    && /opt/app/bin/pip install --upgrade pip wheel \
    && /opt/app/bin/pip install -r requirements.txt \
    && /opt/app/bin/pip uninstall -y opencv-python \
    && /opt/app/bin/pip install opencv-python-headless

# ---- InstantHMR environment ----------------------------------------------
# Separate interpreter: InstantHMR needs torch (RF-DETR) and the CUDA
# onnxruntime, while the web server runs fine on the CPU-only build.
RUN python3.12 -m venv /opt/instanthmr \
    && /opt/instanthmr/bin/pip install --upgrade pip wheel \
    && /opt/instanthmr/bin/pip install \
         "onnxruntime-gpu[cuda,cudnn]>=1.26,<1.27" \
         numpy opencv-python-headless pillow huggingface_hub \
    && /opt/instanthmr/bin/pip install \
         torch torchvision \
         --index-url https://download.pytorch.org/whl/cu126 \
    && /opt/instanthmr/bin/pip install "rfdetr>=1.6"

# ---- InstantHMR graphs, baked into the HF cache ---------------------------
# models.resolve_model() calls hf_hub_download, which with HF_HUB_OFFLINE=1
# (set below) reads this cache and fails instead of downloading.
RUN /opt/instanthmr/bin/python - <<'PY'
import os
from huggingface_hub import hf_hub_download

REPO = "momolesang/InstantHMR"
variants = os.environ["INSTANTHMR_VARIANTS"].split()
files = ["mhr-body.onnx"]
for name in variants:
    files += [f"instanthmr-{name}.onnx",          # network only
              f"instanthmr-{name}-fit1.onnx",     # 1 crop + kinematic fit
              f"instanthmr-{name}-fit2.onnx"]     # 2 crops + kinematic fit
total = 0
for name in files:
    path = hf_hub_download(REPO, name)
    size = os.path.getsize(path)
    total += size
    print(f"{name:44s} {size / 1e6:7.1f} MB")
print(f"{'total':44s} {total / 1e9:7.2f} GB")
PY

# ---- RF-DETR detector weights --------------------------------------------
# PersonDetector builds e.g. rfdetr.RFDETRMedium(), which fetches its .pth into
# ROBOFLOW_HOME on first use. Constructing each variant once bakes them all.
RUN /opt/instanthmr/bin/python - <<'PY'
import gc
import os
import rfdetr

for name in os.environ["RFDETR_VARIANTS"].split():
    cls = getattr(rfdetr, f"RFDETR{name.capitalize()}", None)
    if cls is None:
        print(f"{name}: not shipped by rfdetr "
              f"{getattr(rfdetr, '__version__', '?')}, skipped")
        continue
    try:
        model = cls()
    except Exception as exc:                      # e.g. a plus-only variant
        print(f"{name}: unavailable ({exc}), skipped")
        continue
    del model
    gc.collect()
    print(f"{name}: weights cached")
PY

# ---- offline from here on ------------------------------------------------
# After this point nothing may reach the network. A missing weight now raises
# instead of being fetched, which is what makes the image self-contained.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

# ---- baked-in checks -----------------------------------------------------
# Proves the weights are really present, that the graphs resolve while offline,
# and that the app can find the InstantHMR interpreter and checkout.
RUN /opt/instanthmr/bin/python - <<'PY'
import os
from pathlib import Path

from instanthmr.models import resolve_body, resolve_model

variants = os.environ["INSTANTHMR_VARIANTS"].split()
name = "medium" if "medium" in variants else variants[0]
path = resolve_model(name, fit=2)
assert path.exists(), path
body = resolve_body(path)
assert body.exists(), body
print(f"InstantHMR graphs offline: {path.name} + {body.name}")

import rfdetr
model = rfdetr.RFDETRMedium()
print("RF-DETR offline:", type(model).__name__)

weights = list(Path(os.environ["ROBOFLOW_HOME"]).rglob("*.pth"))
assert weights, f"no detector weights under {os.environ['ROBOFLOW_HOME']}"
print("detector weights:", sorted(p.name for p in weights))
PY

RUN /opt/app/bin/python - <<'PY'
from pathlib import Path

import app.server          # noqa: F401  (fastapi, cv2)
from app import live

root = Path("/app")
assert (root / "web" / "dist" / "index.html").exists(), "bundle missing"
avatars = sorted(p.name for p in (root / "vrm_models").glob("*.vrm"))
assert avatars, "no avatar baked in"
assert (root / "vrm_models" / "idle_loop.vrma").exists(), "no idle clip"

info = live.availability()
assert info["ok"], info
print("app ok |", info["python"], "| avatar:", avatars)
PY

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8010/api/status > /dev/null || exit 1

# 0.0.0.0 is required: the default 127.0.0.1 would only listen on the
# container's loopback and the published port would answer nothing.
CMD ["/opt/app/bin/python", "-m", "app.server", \
     "--host", "0.0.0.0", "--port", "8010"]
