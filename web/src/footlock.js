/**
 * Foot planting: decide which soles are in contact with the floor.
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
 * This module only DETECTS. The hold itself is a two-bone leg IK in
 * `footik.js`. That split matters: the first two versions of this pass tried
 * to cancel skate with a ROOT translation, which cannot work here. With
 * "follow root" off the root barely moves while the feet still slide ~14 cm
 * per stance, so the drift comes from the reconstructed LEG ROTATIONS, not
 * from body travel. Moving the root to cancel one foot just translates the
 * whole avatar (visible whole-body sliding), and a leaky root hold adds
 * centimetre-scale frame-to-frame jitter that makes the measured drift worse.
 * Only a per-leg constraint can hold a foot without moving the character.
 *
 * Detection therefore runs on the NATURAL pose, before the IK has touched any
 * bone. That removes the classic foot-lock failure where a hard hold makes the
 * foot look stationary forever, so a real step can never release: the IK never
 * hides the reconstruction's own motion from this detector.
 *
 * Anchors are stored in ROOT-RELATIVE XZ, so enabling "follow root" cannot
 * leave stale world-space anchors behind. A short leash lets a genuine slow
 * step drag its anchor along once the natural sole is more than a few
 * centimetres away; short-term jitter never moves it.
 */

import * as THREE from 'three';

const CONTACT = {
  height: 0.05,         // m over the bind sole that still counts as down
  releaseHeight: 0.085, // hysteresis: keep the contact until it is clearly up
  speed: 0.42,          // m/s of natural sole motion that still counts as planted
  releaseSpeed: 0.75,
  attack: 0.06,         // s to ramp a contact in
  release: 0.16,        // s to ramp it out
  leash: 0.06,          // m a natural sole may outrun its anchor before it drags it
};

const damp = (dt, tau) => 1 - Math.exp(-dt / tau);

class FootState {
  constructor() {
    this.weight = 0;
    this.planted = false;
    this.anchored = false;
    this.seen = false;
    this.speed = 0;
    this.error = 0;
    this.anchor = new THREE.Vector2();  // root-relative x, z
    this.last = new THREE.Vector2();    // previous natural root-relative x, z
    this.err = new THREE.Vector2();     // anchor - natural sole (x, z)
  }

  clear() {
    this.weight = 0;
    this.planted = false;
    this.anchored = false;
    this.seen = false;
    this.speed = 0;
    this.error = 0;
    this.err.set(0, 0);
  }
}

export class FootLock {
  constructor() {
    this.enabled = true;
    // Live-tunable: the measurement probes sweep this without a rebuild.
    this.leash = CONTACT.leash;
    this.feet = [new FootState(), new FootState()];
    this._feet = this.feet;  // legacy name used by the measurement probes
    this._u = new THREE.Vector2();
  }

  reset() {
    for (const foot of this.feet) foot.clear();
  }

  /** No pose / pass switched off: ramp the contacts out instead of snapping. */
  release(dt) {
    const k = damp(Number.isFinite(dt) && dt > 0 ? dt : 1 / 60,
      CONTACT.release);
    for (const foot of this.feet) {
      foot.weight *= 1 - k;
      if (foot.weight < 1e-3) {
        foot.weight = 0;
        foot.planted = false;
        foot.anchored = false;
      }
      foot.err.set(0, 0);
      foot.error = 0;
    }
  }

  /**
   * Consume this frame's NATURAL sole positions.
   *
   * `left`/`right` are the lowest skin points of each foot in world space,
   * `groundY` their bind-pose floor height, `rootX`/`rootZ` the avatar root,
   * and `strength` the share of the pass to apply (0 in a close-up).
   */
  update(left, right, groundY, rootX, rootZ, dt, strength = 1) {
    dt = Number.isFinite(dt) && dt > 0
      ? Math.min(0.1, Math.max(0.001, dt)) : 1 / 60;
    const active = Number.isFinite(strength) && strength > 1e-3;
    const soles = [left, right];
    for (let i = 0; i < 2; i++) {
      const p = soles[i];
      const foot = this.feet[i];
      if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y) ||
          !Number.isFinite(p.z)) {
        foot.weight *= 1 - damp(dt, CONTACT.release);
        foot.planted = false;
        foot.anchored = false;
        foot.err.set(0, 0);
        continue;
      }
      const u = this._u.set(p.x - rootX, p.z - rootZ);
      const speed = foot.seen
        ? Math.hypot(u.x - foot.last.x, u.y - foot.last.y) / dt : 0;
      foot.speed = speed;
      foot.last.copy(u);
      foot.seen = true;

      const height = p.y - groundY;
      const down = height < (foot.planted ? CONTACT.releaseHeight : CONTACT.height);
      const slow = speed < (foot.planted ? CONTACT.releaseSpeed : CONTACT.speed);
      const contact = active && down && slow;
      foot.planted = contact;
      foot.weight += ((contact ? 1 : 0) - foot.weight) *
        damp(dt, contact ? CONTACT.attack : CONTACT.release);

      if (!contact) {
        foot.anchored = false;
        foot.error = 0;
        foot.err.set(0, 0);
        continue;
      }
      if (!foot.anchored) {
        foot.anchor.copy(u);
        foot.anchored = true;
      } else {
        // Leash: a real slow step drags the anchor along, but only the part
        // beyond `leash`. Bounded leg bend, no hard fight with a genuine step,
        // and no sudden jump when a stance turns into a shuffle.
        const dx = u.x - foot.anchor.x;
        const dz = u.y - foot.anchor.y;
        const d = Math.hypot(dx, dz);
        if (d > this.leash) {
          const pull = (d - this.leash) / d;
          foot.anchor.x += dx * pull;
          foot.anchor.y += dz * pull;
        }
      }
      foot.err.set(foot.anchor.x - u.x, foot.anchor.y - u.y);
      foot.error = foot.err.length();
    }
    return this.feet;
  }
}
