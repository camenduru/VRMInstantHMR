/**
 * Foot planting: keep a planted foot on the floor.
 *
 * Both reference projects in `feet/` treat contact as the thing to detect
 * first and then hold. `UnderPressure/models.py` labels a foot "in contact"
 * from two cheap signals - its height over the ground and how fast it is
 * moving - and `footskate.py` holds that contact position for as long as the
 * label lasts, ramping the constraint in and out so it never snaps. FootMR
 * does the same thing from the other side (`postprocess.py`): it cancels the
 * displacement of the joints it believes are stationary, then drops the whole
 * sequence onto the ground at its lowest point.
 *
 * There is no mesh, no batch and no IK here, so the correction is a root
 * offset rather than an optimisation. While a foot is down and slow, the root
 * is nudged so that foot's height stays on the floor and its sideways drift
 * stops. The nudge is an integrator with clamps: a small gain can hold a plant
 * against a steady error without winding up, and the offset leaks back to zero
 * as soon as both feet leave the ground, so a jump never keeps the avatar
 * pulled down.
 *
 * Detecting the contact is deliberately conservative. Height alone is not
 * enough - a foot on the ground moves fast during a step - and speed alone is
 * not either, because a foot parked in mid air is stationary too. Requiring
 * both, with hysteresis so a foot does not flicker in and out, is what the
 * reference projects do and what works on live data.
 */

import * as THREE from 'three';

const CONTACT = {
  height: 0.055,        // metres over the bind height that still counts as down
  speed: 0.45,          // metres per second that still counts as planted
  releaseHeight: 0.085, // hysteresis: hold the contact until it is clearly up
  releaseSpeed: 0.8,
  attack: 0.06,         // seconds to count as planted
  release: 0.22,        // seconds to count as lifted
  anchor: 0.35,         // seconds for the hold point to follow a walking foot,
                        // so a real step is not fought, only a slide
  gainY: 0.45,          // seconds, height integrator
  gainXZ: 0.8,          // seconds, sideways integrator
  leak: 1.2,            // seconds for the offset to fade out with both feet up
  dead: 0.002,          // metres of error that are pure jitter
  maxY: 0.05,           // metres the plant may move the root vertically
  maxXZ: 0.06,          // ... and horizontally
};

const damp = (dt, tau) => 1 - Math.exp(-dt / tau);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

export class FootLock {
  constructor() {
    this.enabled = true;
    /** The root offset to add to this frame's target. */
    this.correction = new THREE.Vector3();
    /** The integrator behind it. */
    this.offset = new THREE.Vector3();
    this._feet = [newFoot(), newFoot()];
  }

  reset() {
    this.correction.set(0, 0, 0);
    this.offset.set(0, 0, 0);
    for (const foot of this._feet) {
      foot.weight = 0;
      foot.planted = false;
      foot.anchored = false;
      foot.seen = false;
      foot.err.set(0, 0, 0);
    }
  }

  /**
   * Consume this frame's foot positions and return the root offset.
   *
   * `left`/`right` are the feet's world positions (their bind-pose height is
   * the floor reference for the sole), `groundY` that height, `strength` the
   * share of the correction to apply (0 while the pose is still blending in).
   */
  solve(left, right, groundY, dt, strength = 1) {
    const correction = this.correction;
    correction.set(0, 0, 0);
    dt = Number.isFinite(dt) && dt > 0 ? clamp(dt, 0.001, 0.1) : 1 / 60;
    strength = Number.isFinite(strength) ? clamp(strength, 0, 1) : 0;
    if (!left || !right || !Number.isFinite(groundY)) return correction;

    let sum = 0;
    let support = null;
    let lowest = Infinity;
    for (let i = 0; i < 2; i++) {
      const p = i ? right : left;
      const foot = this._feet[i];
      if (!Number.isFinite(p.x) || !Number.isFinite(p.y) ||
          !Number.isFinite(p.z)) {
        // A missing foot is not a planted one; hold the old error while the
        // weight ramps out, but do not measure anything from it.
        foot.weight += (0 - foot.weight) * damp(dt, CONTACT.release);
        if (foot.weight < 1e-3) {
          foot.weight = 0;
          foot.planted = false;
        }
        continue;
      }
      const height = p.y - groundY;
      const speed = foot.seen
        ? Math.hypot(p.x - foot.last.x, p.z - foot.last.z) / dt
        : 0;
      foot.last.copy(p);
      foot.seen = true;

      const down = height < (foot.planted ? CONTACT.releaseHeight : CONTACT.height);
      const slow = speed < (foot.planted ? CONTACT.releaseSpeed : CONTACT.speed);
      const contact = down && slow;
      foot.planted = contact;
      foot.weight += ((contact ? 1 : 0) - foot.weight) *
        damp(dt, contact ? CONTACT.attack : CONTACT.release);
      if (contact) {
        if (!foot.anchored) {
          foot.anchor.set(p.x, p.z);
          foot.anchored = true;
        } else {
          // follow the foot slowly, so walking is not mistaken for sliding
          const k = damp(dt, CONTACT.anchor);
          foot.anchor.x += (p.x - foot.anchor.x) * k;
          foot.anchor.y += (p.z - foot.anchor.y) * k;
        }
        foot.err.set(foot.anchor.x - p.x, groundY - p.y, foot.anchor.y - p.z);
      } else {
        // released: keep the last error so the correction ramps out over a
        // lifted foot instead of chasing it upward
        foot.anchored = false;
      }
      if (foot.weight < 1e-3) {
        foot.weight = 0;
        continue;
      }
      sum += foot.weight;
      if (p.y < lowest) {
        lowest = p.y;
        support = foot;
      }
    }

    const offset = this.offset;
    if (sum <= 1e-3) {
      // Nothing is down: let the offset relax, so a jump or a lifted pose is
      // never held at a lower height than the person actually has.
      offset.multiplyScalar(1 - damp(dt, CONTACT.leak));
      correction.copy(offset).multiplyScalar(strength);
      return correction;
    }

    // Sideways: the mean of the planted feet, with a dead zone so a foot that
    // is merely noisy does not shake the whole avatar.
    let ex = 0;
    let ez = 0;
    for (const foot of this._feet) {
      if (!foot.weight) continue;
      ex += foot.err.x * foot.weight;
      ez += foot.err.z * foot.weight;
    }
    ex /= sum;
    ez /= sum;
    const slide = Math.hypot(ex, ez);
    if (slide < CONTACT.dead) {
      ex = 0;
      ez = 0;
    } else {
      const keep = (slide - CONTACT.dead) / slide;
      ex *= keep;
      ez *= keep;
    }
    const kxz = damp(dt, CONTACT.gainXZ);
    offset.x = clamp(offset.x + ex * kxz, -CONTACT.maxXZ, CONTACT.maxXZ);
    offset.z = clamp(offset.z + ez * kxz, -CONTACT.maxXZ, CONTACT.maxXZ);

    // Height: the support foot, the lowest of those down.
    const ky = damp(dt, CONTACT.gainY);
    offset.y = clamp(offset.y + support.err.y * support.weight * ky,
      -CONTACT.maxY, CONTACT.maxY);

    correction.copy(offset).multiplyScalar(strength);
    return correction;
  }
}

function newFoot() {
  return {
    weight: 0,
    planted: false,
    anchored: false,
    seen: false,
    anchor: new THREE.Vector2(),   // x, z of the hold point
    last: new THREE.Vector3(),
    err: new THREE.Vector3(),      // x, y, z error (y is height over ground)
  };
}
