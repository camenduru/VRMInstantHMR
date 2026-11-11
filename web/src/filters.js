import * as THREE from 'three';

/**
 * Global motion filtering for the retargeted avatar.
 *
 * The retargeter already fights jitter limb by limb, but every one of those
 * fixes is a *decision* about one bone: crop weights, foreshortening
 * references, the anatomical eases. Reconstruction noise that survives all of
 * them shows up as the whole avatar shivering, and shivering is a frequency
 * problem - it lives in the top band of the signal - so the cure is one
 * low-pass over the finished motion rather than another per-limb rule.
 *
 * Two kinds of signal are filtered, both in one place:
 *
 *   positions   - the camera-space joints and auxiliary landmarks, before a
 *                 single bone is solved, so nothing downstream inherits the
 *                 shake; plus the root translation.
 *   rotations   - every bone's finished local quaternion, so what the per-limb
 *                 logic *did* apply is smoothed as well.
 *
 * Three filters are offered, all driven by the same cutoff knob:
 *
 *   butterworth   2nd-order Butterworth (Q = 1/sqrt2): the flattest passband a
 *                 given cutoff allows, which is what makes it the default - it
 *                 removes the shiver without rounding off the poses that are
 *                 supposed to be there.
 *   oneeuro       adaptive: smooth while the limb is still, quick when it
 *                 moves. The best of both for a noisy tracker, at the cost of
 *                 not being a fixed, repeatable response.
 *   exponential   the classic single-pole smoothing. Cheap, obvious, and it
 *                 lags the most for the same amount of quiet.
 *
 * On top of the chosen filter there is a deadband: a movement smaller than the
 * threshold is treated as noise and held, which kills the last of the
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

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const step = (dt) => clamp(Number.isFinite(dt) ? dt : 1 / 60, MIN_DT, MAX_DT);
/** A usable rotation: finite, and not the zero quaternion. */
const isRotation = (q) => Number.isFinite(q.x) && Number.isFinite(q.y) &&
  Number.isFinite(q.z) && Number.isFinite(q.w) &&
  q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w > 1e-12;

/** Filter ids, in the order the panel lists them. */
export const FILTER_TYPES = Object.freeze([
  'off', 'butterworth', 'oneeuro', 'exponential',
]);

/**
 * One second-order Butterworth low-pass section (RBJ cookbook, transposed
 * direct form II). A biquad is the whole filter: two poles and two zeros,
 * which is exactly 2nd-order Butterworth at Q = 1/sqrt2.
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

  reset() {
    this._vec.clear();
    this._quat.clear();
  }

  /* ------------------------------------------------------------ positions */

  vector(v, key, dt) {
    if (!this.active) return v;
    const h = step(dt);
    let ch = this._vec.get(key);
    if (!ch) {
      ch = {
        value: v.clone(),
        prev: v.clone(),
        ema: [v.x, v.y, v.z],
        bq: [new Biquad(), new Biquad(), new Biquad()],
        ready: false,
      };
      // Start the section *at* the first sample rather than at its zero
      // state, or the first frames would mix the real motion with a ramp in
      // from 0.
      ch.bq[0].reset(v.x);
      ch.bq[1].reset(v.y);
      ch.bq[2].reset(v.z);
      this._vec.set(key, ch);
    } else if (ch.ready && this.posDeadband > 0 &&
        v.distanceTo(ch.prev) < this.posDeadband) {
      // under the noise floor: hold, and feed the filter its own held output
      // so resuming from a still pose has no step to recover from
      v.copy(ch.value);
    }
    ch.prev.copy(v);

    switch (this.type) {
      case 'butterworth': {
        const fs = 1 / h;
        ch.bq[0].tune(this.cutoff, fs);
        ch.bq[1].tune(this.cutoff, fs);
        ch.bq[2].tune(this.cutoff, fs);
        ch.value.set(
          ch.bq[0].next(v.x),
          ch.bq[1].next(v.y),
          ch.bq[2].next(v.z),
        );
        break;
      }
      case 'oneeuro': {
        const speed = Math.hypot(
          v.x - ch.ema[0], v.y - ch.ema[1], v.z - ch.ema[2]) / h;
        const fc = this.cutoff + 3 * speed;
        const a = 1 - Math.exp(-h * 2 * Math.PI * fc);
        for (let i = 0; i < 3; i++) {
          ch.ema[i] += ((i === 0 ? v.x : i === 1 ? v.y : v.z) - ch.ema[i]) * a;
        }
        ch.value.set(ch.ema[0], ch.ema[1], ch.ema[2]);
        break;
      }
      case 'exponential': {
        const a = 1 - Math.exp(-h * 2 * Math.PI * this.cutoff);
        for (let i = 0; i < 3; i++) {
          ch.ema[i] += ((i === 0 ? v.x : i === 1 ? v.y : v.z) - ch.ema[i]) * a;
        }
        ch.value.set(ch.ema[0], ch.ema[1], ch.ema[2]);
        break;
      }
      default:
        break;
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
        ema: good ? [q.x, q.y, q.z, q.w] : [0, 0, 0, 1],
        bq: [new Biquad(), new Biquad(), new Biquad(), new Biquad()],
        ready: false,
      };
      for (let i = 0; i < 4; i++) ch.bq[i].reset(ch.ema[i]);
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

    const c = this._tmp;
    c[0] = target.x;
    c[1] = target.y;
    c[2] = target.z;
    c[3] = target.w;

    switch (this.type) {
      case 'butterworth': {
        const fs = 1 / h;
        for (let i = 0; i < 4; i++) {
          ch.bq[i].tune(this.cutoff, fs);
          ch.ema[i] = ch.bq[i].next(c[i]);
        }
        break;
      }
      case 'oneeuro': {
        const fc = this.cutoff + 2.5 * speed;
        const a = 1 - Math.exp(-h * 2 * Math.PI * fc);
        for (let i = 0; i < 4; i++) ch.ema[i] += (c[i] - ch.ema[i]) * a;
        break;
      }
      case 'exponential': {
        const a = 1 - Math.exp(-h * 2 * Math.PI * this.cutoff);
        for (let i = 0; i < 4; i++) ch.ema[i] += (c[i] - ch.ema[i]) * a;
        break;
      }
      default:
        break;
    }

    // Filtering four components separately leaves the point off the unit
    // sphere; normalising is what makes the result a rotation again.
    const len = Math.hypot(ch.ema[0], ch.ema[1], ch.ema[2], ch.ema[3]);
    if (len > 1e-6) {
      ch.value.set(
        ch.ema[0] / len, ch.ema[1] / len, ch.ema[2] / len, ch.ema[3] / len);
    } else {
      ch.value.copy(target);
      ch.ema[0] = target.x;
      ch.ema[1] = target.y;
      ch.ema[2] = target.z;
      ch.ema[3] = target.w;
    }

    ch.ready = true;
    q.copy(ch.value);
    return q;
  }
}
