#!/usr/bin/env python3
"""Persistent InstantHMR worker for the standalone live VRM app.

Launched by ``app/live.py`` with InstantHMR's own Python and kept alive across
frames: loading the graphs costs seconds, so a live session must pay that once
instead of per frame.

Wire protocol, over stdin/stdout:

    parent -> child   ``<IIQ`` (width, height, seq) then the RGB bytes
                      width * height * 3.  A zero width means stop.
    child  -> parent  one JSON object per line:
                      {"ready": true, "provider": "..."} once the graphs are up
                      {"seq": n, "detected": bool, "ms": ..,
                       "w": .., "h": .., "joints": [[..70 x 3..]],
                       "joints2d": [[..70 x 2..]]}
                      {"seq": n, "error": "..."} for a frame that failed

``joints`` is the (70, 3) MHR70 set in camera axes, exactly like the offline
bake's ``joints_cam``; the app maps it onto the 22-joint skeleton the browser
retargets. ``joints2d`` is the same landmark set in pixels of the submitted
frame, used for the preview skeleton. At most one frame is in flight, so a slow
inference never builds a backlog of stale poses.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time

import numpy as np

HEADER = struct.Struct('<IIQ')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='medium',
                   help='nano, tiny, small, medium, medium-commercial, or an .onnx path')
    p.add_argument('--fit', type=int, default=2, choices=(0, 1, 2),
                   help='kinematic fit on 1 or 2 crops per person (default 2)')
    p.add_argument('--device', default='auto',
                   choices=('auto', 'cuda', 'tensorrt', 'cpu', 'coreml'))
    p.add_argument('--detector', default='medium',
                   choices=('nano', 'small', 'medium', 'base', 'large'))
    p.add_argument('--detector-stride', type=int, default=2, metavar='N')
    p.add_argument('--pose-filter', default='one-euro',
                   choices=('one-euro', 'alpha-beta', 'none'))
    return p.parse_args()


def pick_device(requested: str) -> str:
    """Resolve ``auto`` and fall back to the CPU when the GPU provider is absent."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if requested == 'auto':
        requested = 'cuda' if 'CUDAExecutionProvider' in available else 'cpu'
    if requested == 'cuda' and 'CUDAExecutionProvider' not in available:
        print('[loading] CUDA is not available in this onnxruntime build; '
              'using the CPU', flush=True)
        return 'cpu'
    if requested == 'tensorrt' and 'TensorrtExecutionProvider' not in available:
        print('[loading] TensorRT is not available in this onnxruntime build; '
              'using CUDA', flush=True)
        return 'cuda' if 'CUDAExecutionProvider' in available else 'cpu'
    return requested


def box_center(box: np.ndarray) -> tuple[float, float]:
    return (float(box[0] + box[2]) * 0.5, float(box[1] + box[3]) * 0.5)


def box_area(box: np.ndarray) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def box_diagonal(box: np.ndarray) -> float:
    return float(np.hypot(box[2] - box[0], box[3] - box[1]))


def choose_person(persons: list, previous: tuple[float, float] | None,
                  previous_size: float) -> tuple[int, np.ndarray]:
    """Index of the person to follow and their box.

    Continuity wins when a candidate is close to the previous box's centre;
    otherwise (first frame, a cut, or a person who left) the largest person is
    the subject. This is how a two-person clip stays on one body instead of
    alternating between them every frame.
    """
    if not persons:
        raise ValueError('no persons')
    if previous is not None:
        best_index, distance = 0, float('inf')
        for i, person in enumerate(persons):
            centre = box_center(person.bbox)
            gap = float(np.hypot(centre[0] - previous[0],
                                 centre[1] - previous[1]))
            if gap < distance:
                best_index, distance = i, gap
        limit = max(previous_size,
                    box_diagonal(persons[best_index].bbox)) * 0.75
        if distance <= limit:
            return best_index, persons[best_index].bbox
    best_index = max(range(len(persons)),
                     key=lambda i: box_area(persons[i].bbox))
    return best_index, persons[best_index].bbox


def read_exact(stream, size: int):
    """Read exactly ``size`` bytes, or None at a clean EOF."""
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def main() -> int:
    args = parse_args()
    from instanthmr import PosePipeline

    device = pick_device(args.device)
    pose_filter = None if args.pose_filter == 'none' else args.pose_filter
    try:
        pipeline = PosePipeline(args.model, fit=args.fit, device=device,
                                detector=args.detector, max_persons=1,
                                detector_stride=max(1, args.detector_stride),
                                pose_filter=pose_filter)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({'fatal': f'InstantHMR failed to load: {exc}'}),
              flush=True)
        return 1

    print(json.dumps({'ready': True, 'provider': pipeline.hmr.provider}),
          flush=True)

    stdin = sys.stdin.buffer
    previous_centre: tuple[float, float] | None = None
    previous_size = 0.0
    while True:
        header = read_exact(stdin, HEADER.size)
        if header is None:
            break
        width, height, seq = HEADER.unpack(header)
        if width == 0 or height == 0:
            break
        payload = read_exact(stdin, width * height * 3)
        if payload is None:
            break
        rgb = np.frombuffer(payload, np.uint8).reshape(height, width, 3)
        started = time.perf_counter()
        try:
            result = pipeline(rgb, time.perf_counter())
            person = None
            if result.persons:
                which, box = choose_person(result.persons, previous_centre,
                                           previous_size)
                person = result.persons[which]
                previous_centre = box_center(box)
                previous_size = box_diagonal(box)
            reply = {'seq': seq, 'detected': person is not None,
                     'w': width, 'h': height,
                     'ms': round((time.perf_counter() - started) * 1000.0, 1)}
            if person is not None:
                reply['joints'] = np.round(person.joints_3d_cam, 4).tolist()
                reply['joints2d'] = np.round(person.joints_2d, 1).tolist()
                reply['box'] = np.round(box, 1).tolist()
        except Exception as exc:  # noqa: BLE001
            reply = {'seq': seq, 'error': str(exc)}
        print(json.dumps(reply, separators=(',', ':')), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
