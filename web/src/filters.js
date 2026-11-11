import * as THREE from 'three';

/**
 * Global motion filtering for the retargeted avatar.
 *
 * The retargeter already fights jitter limb by limb, but every one of those
 * fixes is a *decision* about one bone: crop weights, foreshortening
 * references, the anatomical eases. Reconstruction noise that survives all of
 * them shows up as the whole avatar shivering, and shivering is a frequency
 * problem - it lives in the top band of the signal - so the cure is one
 * filter over the finished motion rather than another per-limb rule.
 *
 * Two kinds of signal are filtered, both in one place:
 *
 *   positions   - the camera-space joints and auxiliary landmarks, before a
 *                 single bone is solved, so nothing downstream inherits the
 *                 shake; plus the root translation.
 *   rotations   - every bone's finished local quaternion, so what the per-limb
 *                 logic *did* apply is smoothed as well.
 *
 * The methods are the real-time ones from the reference projects kept in
 * `filters/`, ported to run per frame in the browser:
 *
 *   butterworth     mocap-refinement BF.py    2nd-order Butterworth biquad.
 *   exponential     mocap-refinement EMA.py   one-pole smoothing.
 *   oneeuro         (adaptive one-pole)       smooth while still, quick when
 *                                             moving.
 *   sma             mocap-refinement SMA.py   moving average.
 *   wma             mocap-refinement WMA.py   weighted moving average.
 *   hde             mocap-refinement HDE.py   smoothing plus a trend term.
 *   kalman          mocap-refinement KF1.py   constant-velocity Kalman.
 *   adaptiveKalman  FLK AKF.py                Kalman whose measurement gain
 *                                             falls when a sample jumps.
 *
 * Not ported, on purpose:
 *   KF0.py           a one-state Kalman is an exponential filter, already here;
 *   KF2.py           the constant-acceleration filter's third state is
 *                    ill-conditioned at this frame rate - its acceleration
 *                    gain grows without bound as the cutoff nears Nyquist,
 *                    so part of the slider would simply diverge;
 *   SG.py            Savitzky-Golay is a *centred* smoother; run causally it
 *                    has to extrapolate the polynomial to the newest sample,
 *                    which measured a 1.1x gain on 15 Hz noise - it would add
 *                    the shake it exists to remove;
 *   INT.py           fills *missing* samples, and the worker refuses a frame
 *                    with a missing joint before it ever reaches the page;
 *   TSVD/LRMC/NLRMC/RPCA   batch methods: they need the finished clip;
 *   ODAE/UAE/UDAE/RNN, N-euro, motion_rolling_prediction   trained networks and
 *                    PyTorch checkpoints, which cannot run in this page.
 *
 * The Kalman ports keep the reference's model order but not its hard-coded
 * Q/R: the panel has one cutoff knob, so the gains are the steady-state ones
 * that knob implies (an optimal alpha-beta filter *is* the steady-state KF1),
 * which is one multiply per sample instead of a covariance recursion. The
 * adaptive variant mirrors AKF's confidence weighting, but its "treat a very
 * low-confidence sample as missing" branch becomes a gain *floor*: missing
 * data never reaches this layer (a frame with a missing joint is rejected
 * upstream), and a hard gate would freeze a limb that re-enters the shot with
 * one large step.
 *
 * On top of whichever filter runs, there is a deadband: a movement smaller
 * than the threshold is treated as noise and held, which kills the last of the
 * stand-still shiver that no low-pass fully removes (a filter never reaches
 * zero, it only decays).
 */

// Butterworth's Q: the pole pair that makes the response maximally flat.
const Q_BUTTERWORTH = Math.SQRT1_2;
const MIN_CUTOFF = 0.2;
const MAX_CUTOFF = 24;
// A frame slower than this is a stall, not a frame: clamping keeps the
// coefficients (and therefore the filter) stable.
const MIN_DT = 1 / 240;
const MAX_DT = 1 / 10;
// Window filters look at at most this many frames back.
const MAX_WINDOW = 31;
const MIN_WINDOW = 3;

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const step = (dt) => clamp(Number.isFinite(dt) ? dt : 1 / 60, MIN_DT, MAX_DT);
/** A usable rotation: finite, and not the zero quaternion. */
const isRotation = (q) => Number.isFinite(q.x) && Number.isFinite(q.y) &&
  Number.isFinite(q.z) && Number.isFinite(q.w) &&
  q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w > 1e-12;

/** Filter ids, in the order the panel lists them. */
export const FILTER_TYPES = Object.freeze([
  'off',
  'butterworth', 'oneeuro', 'exponential',
  'sma', 'wma',
  'hde', 'kalman', 'adaptiveKalman',
]);

/** The filters whose knob is a window length rather than a cutoff. */
export const WINDOW_TYPES = Object.freeze(['sma', 'wma']);

/**
 * Window length, in frames, that a cutoff asks for: the same amount of
 * smoothing an N-frame average gives is the same amount a low-pass at roughly
 * fs/(2.2N) gives, so the one slider drives both kinds sensibly.
 */
export function windowForCutoff(cutoff, fps = 60) {
  const want = Math.round(fps / (2.2 * Math.max(MIN_CUTOFF, cutoff)));
  const odd = want % 2 === 0 ? want + 1 : want;
  return clamp(odd, MIN_WINDOW, MAX_WINDOW);
}

/**
 * One second-order Butterworth low-pass section (RBJ cookbook, transposed
 * direct form II). A biquad is the whole filter: two poles and two zeros,
 * which is exactly a 2nd-order Butterworth at Q = 1/sqrt2.
 */
class Biquad {
  constructor() {
    this.b0 = 1;
    this.b1 = 0;
    this.b2 = 0;
    this.a1 = 0;
    this.a2 = 0;
    this.x1 = 0;
    this.x2 = 0;
    this.y1 = 0;
    this.y2 = 0;
    this.fc = 0;
    this.fs = 0;
  }

  /** Coefficients for cutoff `fc` at sample rate `fs`; cached per pair. */
  tune(fc, fs) {
    if (this.fc === fc && this.fs === fs) return;
    this.fc = fc;
    this.fs = fs;
    // never let the cutoff reach Nyquist: the bilinear warp would fold the
    // passband and the section would stop being a low-pass at all
    const f = Math.min(fc, fs * 0.45);
    const w0 = (2 * Math.PI * f) / fs;
    const cos = Math.cos(w0);
    const alpha = Math.sin(w0) / (2 * Q_BUTTERWORTH);
    const a0 = 1 + alpha;
    this.b0 = (1 - cos) / 2 / a0;
    this.b1 = (1 - cos) / a0;
    this.b2 = this.b0;
    this.a1 = (-2 * cos) / a0;
    this.a2 = (1 - alpha) / a0;
  }

  reset(value = 0) {
    this.x1 = this.x2 = value;
    this.y1 = this.y2 = value;
  }

  next(x) {
    const y = this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2 -
      this.a1 * this.y1 - this.a2 * this.y2;
    this.x2 = this.x1;
    this.x1 = x;
    this.y2 = this.y1;
    this.y1 = y;
    // a non-finite sample must not poison the state for every frame after it
    return Number.isFinite(y) ? y : (Number.isFinite(x) ? x : 0);
  }
}

/* ------------------------------------------------------------ the filters --
 * Every filter is a per-component scalar state machine with the same shape:
 *
 *   push(x, h, ctx) -> y     one sample in, the filtered sample out
 *   reset(x)                 restart from `x`
 *
 * `ctx` carries the shared configuration: `cutoff` (Hz), `speed` (how fast
 * this channel is moving, units per second) and `window` (frames, for the
 * windowed methods). Keeping the filters scalar is what lets a Vector3 be
 * three of them and a quaternion four, with no filter-specific channel code.
 */

function makeButterworth() {
  const bq = new Biquad();
  return {
    push(x, h, ctx) {
      bq.tune(Math.min(ctx.cutoff, 1 / h), 1 / h);
      return bq.next(x);
    },
    reset(x) {
      bq.reset(x);
    },
  };
}

/** EMA.py: alpha fixed per frame, derived from the cutoff instead. */
function makeExponential() {
  let v = 0;
  let ready = false;
  return {
    push(x, h, ctx) {
      if (!ready) {
        v = x;
        ready = true;
        return v;
      }
      const a = 1 - Math.exp(-h * 2 * Math.PI * ctx.cutoff);
      v += (x - v) * a;
      return v;
    },
    reset(x) {
      v = x;
      ready = false;
    },
  };
}

/** An exponential whose cutoff rises with the channel's own speed. */
function makeOneEuro() {
  let v = 0;
  let ready = false;
  return {
    push(x, h, ctx) {
      if (!ready) {
        v = x;
        ready = true;
        return v;
      }
      const cutoff = ctx.cutoff + 2.5 * ctx.speed;
      const a = 1 - Math.exp(-h * 2 * Math.PI * cutoff);
      v += (x - v) * a;
      return v;
    },
    reset(x) {
      v = x;
      ready = false;
    },
  };
}

/**
 * HDE.py: the reference smooths towards a *trend-extrapolated* previous value
 * and adds the trend back on the way out, so the filter spends less of its lag
 * catching up with a limb that is already moving. It keeps the reference's
 * trend rate (beta) and takes alpha from the cutoff.
 */
function makeHde() {
  const beta = 0.1;
  let p = 0;
  let trend = 0;
  let ready = false;
  return {
    push(x, h, ctx) {
      if (!ready) {
        p = x;
        trend = 0;
        ready = true;
        return x;
      }
      const a = 1 - Math.exp(-h * 2 * Math.PI * ctx.cutoff);
      const predicted = p + trend;
      const next = a * x + (1 - a) * predicted;
      trend = beta * (next - p) + (1 - beta) * trend;
      p = next;
      return p + trend;
    },
    reset(x) {
      p = x;
      trend = 0;
      ready = false;
    },
  };
}

/**
 * KF1.py as a steady-state Kalman filter: a constant-velocity model whose
 * optimal position/velocity gains obey beta = alpha^2 / (2 - alpha). Keeping
 * the velocity is what makes it lead an EMA at the same cutoff - the price is
 * a little overshoot when the person reverses.
 */
function makeKalman() {
  let p = 0;
  let v = 0;
  let ready = false;
  return {
    push(x, h, ctx) {
      if (!ready) {
        p = x;
        v = 0;
        ready = true;
        return x;
      }
      const a = 1 - Math.exp(-h * 2 * Math.PI * ctx.cutoff);
      const b = (a * a) / (2 - a);
      const predicted = p + v * h;
      const err = x - predicted;
      p = predicted + a * err;
      v += (b / h) * err;
      return p;
    },
    reset(x) {
      p = x;
      v = 0;
      ready = false;
    },
  };
}

/**
 * AKF.py: the adaptive Kalman filter used by FLK. The reference turns its
 * measurement noise up with the size of the jump (`R = alpha * v^2`), so a
 * sample that leaps away from where the model expected it is trusted less and
 * a spike is smoothed out instead of being followed. That confidence is
 * reproduced here; the reference's "confidence below theta means the sample is
 * missing" branch is not needed (see the header).
 */
function makeAdaptiveKalman() {
  const noise = 0.01;
  // AKF replaces a sample whose confidence falls under a threshold with the
  // model's prediction. A floor on the gain does the same job - a jump is
  // mostly resisted - without ever freezing the channel.
  const floor = 0.25;
  let p = 0;
  let v = 0;
  let ready = false;
  return {
    push(x, h, ctx) {
      if (!ready) {
        p = x;
        v = 0;
        ready = true;
        return x;
      }
      const a = 1 - Math.exp(-h * 2 * Math.PI * ctx.cutoff);
      const b = (a * a) / (2 - a);
      const predicted = p + v * h;
      const err = x - predicted;
      const confidence = clamp(1 / (noise * ctx.speed * ctx.speed + 1),
        floor, 1);
      const k = a * confidence;
      p = predicted + k * err;
      v += (b / h) * k * err;
      return p;
    },
    reset(x) {
      p = x;
      v = 0;
      ready = false;
    },
  };
}

/* -------------------------------------------------------- windowed methods */

const WEIGHTS = new Map();

function windowWeights(kind, n) {
  const key = `${kind}:${n}`;
  const cached = WEIGHTS.get(key);
  if (cached) return cached;
  let weights;
  if (kind === 'sma') {
    weights = new Float64Array(n);
    weights.fill(1 / n);
  } else if (kind === 'wma') {
    // WMA.py's triangular weights, newest heaviest: a causal window cannot be
    // symmetric, and weighting the newest sample highest is the point of it.
    weights = new Float64Array(n);
    const total = (n * (n + 1)) / 2;
    for (let i = 0; i < n; i++) weights[i] = (n - i) / total;
  } else {
    weights = new Float64Array(n);
    weights.fill(1 / n);
  }
  WEIGHTS.set(key, weights);
  return weights;
}

/** SMA.py / WMA.py: a ring buffer and a weight vector. */
function makeWindowed(kind) {
  const buffer = new Float64Array(MAX_WINDOW);
  let count = 0;
  let head = 0;
  let cached = 0;
  let weights = null;
  return {
    push(x, h, ctx) {
      buffer[head] = x;
      head = head + 1 === MAX_WINDOW ? 0 : head + 1;
      if (count < MAX_WINDOW) count += 1;
      const n = Math.min(ctx.window, count);
      if (n !== cached) {
        cached = n;
        weights = windowWeights(kind, n);
      }
      let sum = 0;
      for (let i = 0; i < n; i++) {
        let index = head - 1 - i;   // 0 is the newest sample
        if (index < 0) index += MAX_WINDOW;
        sum += buffer[index] * weights[i];
      }
      return sum;
    },
    reset(x) {
      buffer.fill(x);
      count = 1;
      head = 1;
      cached = 0;
      weights = null;
    },
  };
}

function make(type) {
  switch (type) {
    case 'exponential':
      return makeExponential();
    case 'oneeuro':
      return makeOneEuro();
    case 'hde':
      return makeHde();
    case 'kalman':
      return makeKalman();
    case 'adaptiveKalman':
      return makeAdaptiveKalman();
    case 'sma':
    case 'wma':
      return makeWindowed(type);
    case 'butterworth':
    default:
      return makeButterworth();
  }
}

/* ------------------------------------------------------------ the bank ---- */

export class MotionFilter {
  constructor({ type = 'off', cutoff = 5, deadband = 0 } = {}) {
    this.type = FILTER_TYPES.includes(type) ? type : 'off';
    this.cutoff = clamp(cutoff, MIN_CUTOFF, MAX_CUTOFF);
    this.deadband = 0;
    this.posDeadband = 0;
    this.setDeadband(deadband);
    // channel name -> filter state; positions and rotations are separate
    // banks because a joint channel is a vector and a bone channel a
    // quaternion (which also needs hemisphere bookkeeping)
    this._vec = new Map();
    this._quat = new Map();
    this._tmp = [0, 0, 0, 0];
    this._ctx = { cutoff: this.cutoff, speed: 0, window: MIN_WINDOW };
    this._windowLen = 0;
  }

  /**
   * Retune from the panel. Changing the filter type restarts the banks so the
   * new filter is not fed the old one's state; moving a slider does not, so
   * dragging strength does not hitch the avatar.
   */
  configure({ type, cutoff, deadband } = {}) {
    let changed = false;
    if (typeof type === 'string' && FILTER_TYPES.includes(type) &&
        type !== this.type) {
      this.type = type;
      changed = true;
    }
    if (Number.isFinite(cutoff)) this.cutoff = clamp(cutoff, MIN_CUTOFF, MAX_CUTOFF);
    if (Number.isFinite(deadband)) this.setDeadband(deadband);
    if (changed) this.reset();
    return changed;
  }

  /**
   * How far the filter is allowed to sit still, in degrees of rotation. The
   * position threshold is derived from it: the two knobs are the same kind of
   * thing (reconstruction noise) at two different scales, so one slider
   * driving both keeps the panel honest.
   */
  setDeadband(degrees) {
    const deg = clamp(Number.isFinite(degrees) ? degrees : 0, 0, 8);
    this.deadband = THREE.MathUtils.degToRad(deg);
    this.posDeadband = this.deadband * 0.2;
  }

  get active() {
    return this.type !== 'off';
  }

  /** Window length in frames, for the windowed filters (3 for the others). */
  get windowLength() {
    return this._windowLen || MIN_WINDOW;
  }

  reset() {
    this._vec.clear();
    this._quat.clear();
    this._windowLen = 0;
  }

  /**
   * The window a cutoff asks for, with hysteresis: the frame rate wobbles by a
   * sample or two, and without the 2-frame rule a window sitting on a boundary
   * would flip every frame and hitch the avatar.
   */
  _window(h) {
    const want = windowForCutoff(this.cutoff, 1 / h);
    if (!this._windowLen) this._windowLen = want;
    else if (Math.abs(want - this._windowLen) >= 2) this._windowLen = want;
    return this._windowLen;
  }

  _ctxFor(speed, h) {
    const ctx = this._ctx;
    ctx.cutoff = this.cutoff;
    ctx.speed = speed;
    ctx.window = this._window(h);
    return ctx;
  }

  /* ------------------------------------------------------------ positions */

  vector(v, key, dt) {
    if (!this.active) return v;
    if (!Number.isFinite(v.x + v.y + v.z)) return v;
    const h = step(dt);
    let ch = this._vec.get(key);
    if (!ch) {
      ch = {
        value: v.clone(),
        prev: v.clone(),
        filters: [make(this.type), make(this.type), make(this.type)],
        ready: false,
      };
      this._vec.set(key, ch);
    } else if (ch.ready && this.posDeadband > 0 &&
        v.distanceTo(ch.prev) < this.posDeadband) {
      // under the noise floor: hold, and feed the filter its own held output
      // so resuming from a still pose has no step to recover from
      v.copy(ch.value);
    }
    const speed = ch.ready ? v.distanceTo(ch.prev) / h : 0;
    ch.prev.copy(v);

    const ctx = this._ctxFor(speed, h);
    const x = v.x;
    const y = v.y;
    const z = v.z;
    ch.value.set(
      ch.filters[0].push(x, h, ctx),
      ch.filters[1].push(y, h, ctx),
      ch.filters[2].push(z, h, ctx),
    );
    if (!Number.isFinite(ch.value.x + ch.value.y + ch.value.z)) {
      // a filter that diverged is worse than no filter: rebuild this channel
      this._vec.delete(key);
      v.set(x, y, z);
      return v;
    }
    v.copy(ch.value);
    ch.ready = true;
    return v;
  }

  /** Filters the camera-space joints in place, one channel per joint. */
  joints(list, count, dt) {
    if (!this.active) return list;
    for (let i = 0; i < count; i++) this.vector(list[i], `joint${i}`, dt);
    return list;
  }

  /* ------------------------------------------------------------ rotations */

  quaternion(q, key, dt) {
    if (!this.active) return q;
    const h = step(dt);
    let ch = this._quat.get(key);
    const good = isRotation(q);
    if (!ch) {
      ch = {
        value: good ? q.clone() : new THREE.Quaternion(),
        target: new THREE.Quaternion(),
        filters: [
          make(this.type), make(this.type), make(this.type), make(this.type),
        ],
        ready: false,
      };
      this._quat.set(key, ch);
    }
    if (!good) {
      // A sample that is not a rotation cannot be filtered. Hold the last good
      // orientation instead: one broken frame must not poison the channel for
      // every frame after it.
      q.copy(ch.value);
      return q;
    }

    // q and -q are the same rotation, and the tracker's solve can land on
    // either side of that pair. Put the new sample on the filtered one's
    // hemisphere or the filter would chase a full turn that never happened.
    const target = ch.target.copy(q);
    if (target.dot(ch.value) < 0) {
      target.set(-target.x, -target.y, -target.z, -target.w);
    }

    let angle = 2 * Math.acos(clamp(Math.abs(ch.value.dot(target)), -1, 1));
    let speed = angle / h;
    if (ch.ready && this.deadband > 0 && angle < this.deadband) {
      target.copy(ch.value);
      angle = 0;
      speed = 0;
    }

    const ctx = this._ctxFor(speed, h);
    const c = this._tmp;
    c[0] = target.x;
    c[1] = target.y;
    c[2] = target.z;
    c[3] = target.w;
    const x = ch.filters[0].push(c[0], h, ctx);
    const y = ch.filters[1].push(c[1], h, ctx);
    const z = ch.filters[2].push(c[2], h, ctx);
    const w = ch.filters[3].push(c[3], h, ctx);

    // Filtering four components separately leaves the point off the unit
    // sphere; normalising is what makes the result a rotation again.
    const len = Math.hypot(x, y, z, w);
    if (!Number.isFinite(len) || len <= 1e-6) {
      this._quat.delete(key);
      q.copy(target);
      return q;
    }
    ch.value.set(x / len, y / len, z / len, w / len);
    ch.ready = true;
    q.copy(ch.value);
    return q;
  }
}
