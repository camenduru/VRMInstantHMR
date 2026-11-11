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
 * leave stale world-space anchors behind.
 *
 * THREE THINGS HAD TO BE FIXED BEFORE THE HOLD DID ANYTHING AT ALL. Each one
 * was measured, not guessed, and each one on its own made the plant a no-op:
 *
 * 1. The anchor must be FROZEN, not leashed. An anchor dragged along behind
 *    the natural sole leaves the IK nothing to correct beyond the distance it
 *    is allowed to lag, so the foot slides with the reconstruction. A 6 cm
 *    leash measured as removing almost nothing: a planted foot still travelled
 *    45 cm across the dance clip.
 *
 * 2. The anchor must OUTLIVE a dropout. On this reconstruction the sole speed
 *    spikes above the contact gate for a single frame every 2-4 frames, and
 *    clearing the anchor on the first missed frame re-seeded it at the current
 *    position, so the hold was permanently reset to wherever the foot already
 *    was. `hold` debounces the end of a stance.
 *
 * 3. The step test must not read a single frame. The dance clip contains
 *    one-frame jumps of 10-16 cm (up to 4.8 m/s) - tracking glitches, not
 *    motion. Comparing the anchor against the raw sole released the hold on
 *    every glitch, which is why raising the IK allowance changed nothing. The
 *    anchor, the step test AND the contact gate all read a low-passed position
 *    (`filter`). The gate needs it most: on a 17 fps live feed the raw jitter
 *    of a standing person reads as 0.3 m/s, so a raw gate finds no contact at
 *    all and the whole pass silently does nothing. A low-pass keeps a
 *    sustained glide at its true speed while averaging zero-mean jitter away.
 *
 * TUNING. The thresholds are instance fields (`this.contact`) rather than
 * module constants so they can be swept against a real clip without a
 * rebuild; `CONTACT` below is only the default. Measured on the bundled dance
 * clip, the reconstruction keeps BOTH soles under 5 cm for 159-160 of 160
 * frames, so height alone can never separate a stance from a step there -
 * speed has to do that work.
 */

import * as THREE from 'three';

const CONTACT = {
  height: 0.035,        // m over the bind sole that still counts as down
  releaseHeight: 0.06,  // hysteresis: keep the contact until it is clearly up
  speed: 0.08,          // m/s of natural sole motion that still counts as planted
  releaseSpeed: 0.16,
  attack: 0.04,         // s to ramp a contact in
  release: 0.12,        // s to ramp it out
  hold: 0.25,           // s a stance survives with no contact before it ends
  step: 0.20,           // m the sole may leave its anchor before it is a new step
  filter: 0.08,         // s low-pass on the position used to place the anchor
  creep: 0.0,           // m/s the anchor may drift; 0 means a held foot never drifts
  leash: 0.30,          // m: safety limit, drag the anchor past what a leg can absorb
};

const damp = (dt, tau) => 1 - Math.exp(-dt / tau);

class FootState {
  constructor() {
    this.weight = 0;
    this.planted = false;
    this.anchored = false;
    this.seen = false;
    this.speed = 0;
    this.speedRaw = 0;
    this.error = 0;
    this.lost = 0;                      // s since the last contact frame
    this.anchor = new THREE.Vector2();  // root-relative x, z
    this.last = new THREE.Vector2();    // previous natural root-relative x, z
    this.filt = new THREE.Vector2();    // low-passed natural position
    this.lastFilt = new THREE.Vector2();// previous low-passed position
    this.err = new THREE.Vector2();     // anchor - natural sole (x, z)
  }

  clear() {
    this.weight = 0;
    this.planted = false;
    this.anchored = false;
    this.seen = false;
    this.speed = 0;
    this.speedRaw = 0;
    this.error = 0;
    this.lost = 0;
    this.err.set(0, 0);
  }
}

export class FootLock {
  constructor() {
    this.enabled = true;
    // Live-tunable copies, so the measurement probes can sweep them.
    this.contact = { ...CONTACT };
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
      this.contact.release);
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
    const C = this.contact;
    dt = Number.isFinite(dt) && dt > 0
      ? Math.min(0.1, Math.max(0.001, dt)) : 1 / 60;
    const active = Number.isFinite(strength) && strength > 1e-3;
    const soles = [left, right];
    for (let i = 0; i < 2; i++) {
      const p = soles[i];
      const foot = this.feet[i];
      if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y) ||
          !Number.isFinite(p.z)) {
        foot.weight *= 1 - damp(dt, C.release);
        foot.planted = false;
        foot.anchored = false;
        foot.lost = 0;
        foot.seen = false;
        foot.err.set(0, 0);
        continue;
      }
      const u = this._u.set(p.x - rootX, p.z - rootZ);

      // Spike-free position. A one-frame tracking jump must not be mistaken
      // for the foot walking somewhere, so the anchor, the step test and the
      // contact gate all read this. The gate especially: on a 17 fps live feed
      // the raw per-frame jitter of a standing person already reads as
      // 0.3 m/s, so a gate fed the raw sole finds no contact at all and the
      // plant silently does nothing. A low-pass keeps a sustained glide or
      // step at its true speed while averaging zero-mean jitter away.
      if (!foot.seen) {
        foot.filt.copy(u);
        foot.lastFilt.copy(u);
      } else {
        foot.filt.lerp(u, damp(dt, C.filter));
      }
      foot.speedRaw = foot.seen
        ? Math.hypot(u.x - foot.last.x, u.y - foot.last.y) / dt : 0;
      const speed = foot.seen
        ? Math.hypot(foot.filt.x - foot.lastFilt.x,
          foot.filt.y - foot.lastFilt.y) / dt : 0;
      foot.speed = speed;
      foot.last.copy(u);
      foot.lastFilt.copy(foot.filt);
      foot.seen = true;

      const height = p.y - groundY;
      const down = height < (foot.planted ? C.releaseHeight : C.height);
      const slow = speed < (foot.planted ? C.releaseSpeed : C.speed);
      const contact = active && down && slow;
      foot.planted = contact;
      foot.lost = contact ? 0 : foot.lost + dt;

      // A stance opens on a contact frame and is anchored where the sole is.
      if (!foot.anchored && contact) {
        foot.anchor.copy(foot.filt);
        foot.anchored = true;
      }

      // How far the sole has walked away from its anchor. Past `step` the foot
      // is not being held any more - it has gone somewhere else.
      const stepDist = foot.anchored
        ? Math.hypot(foot.filt.x - foot.anchor.x,
          foot.filt.y - foot.anchor.y) : 0;

      // The anchor survives a missed frame (`hold`) but not a real step.
      const holding = foot.anchored && active &&
        foot.lost <= C.hold && stepDist <= C.step;
      if (!holding) foot.anchored = false;

      foot.weight += ((holding ? 1 : 0) - foot.weight) *
        damp(dt, holding ? C.attack : C.release);

      if (!holding) {
        foot.error = 0;
        foot.err.set(0, 0);
        continue;
      }

      // The anchor is FROZEN while the stance lasts. That is the whole point
      // of the plant: an anchor that follows the natural sole leaves the IK
      // nothing to correct beyond the distance it is allowed to lag, so the
      // foot slides with the reconstruction and the pass measures as doing
      // nothing at all.
      //
      // `creep` would bleed a genuine slow drift off at a few cm/s, but any
      // creep at all is slide the viewer can see over a long stance, so it is
      // 0 by default: a held foot stays exactly where it landed. `leash` is
      // only a safety limit: past the distance a leg can absorb, the anchor is
      // dragged rather than asking for an impossible correction.
      const dx = foot.filt.x - foot.anchor.x;
      const dz = foot.filt.y - foot.anchor.y;
      const d = Math.hypot(dx, dz);
      if (d > 1e-6) {
        const step = Math.min(d, C.creep * dt + Math.max(0, d - C.leash));
        foot.anchor.x += (dx / d) * step;
        foot.anchor.y += (dz / d) * step;
      }
      foot.err.set(foot.anchor.x - foot.filt.x, foot.anchor.y - foot.filt.y);
      foot.error = foot.err.length();
    }
    return this.feet;
  }
}
