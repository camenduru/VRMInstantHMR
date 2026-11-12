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
 *    (The ANCHOR is the one exception and reads the raw sole - see the plant
 *    below; denoising the thing you are about to hold puts the hold itself
 *    off target.)
 *
 * TUNING. The thresholds are instance fields (`this.contact`) rather than
 * module constants so they can be swept against a real clip without a
 * rebuild; `CONTACT` below is only the default.
 *
 * The gate reads three things, and only one of them had to be tightened. On a
 * 722-frame live capture the soles glide horizontally at 0.27 m/s (median,
 * filtered) even while they are on the floor - that footage is zero-mean
 * tracking jitter, not walking, because each sole nets only 10-25 cm over
 * 28 s. A horizontal-only gate at the old 0.08 m/s therefore called just
 * 8-15% of frames a contact, and a hold that is off 90% of the time cannot
 * stop a foot from skating. Horizontal speed is now only a coarse "that is a
 * real swing" test; the vertical derivative does the stance/step separation
 * (median |dy/dt| of the lower sole is 0.03-0.08 m/s against 0.5-1.5 m/s for
 * a foot being lifted), and height keeps a raised foot out.
 */

import * as THREE from 'three';

const CONTACT = {
  height: 0.05,         // m over the bind sole that still counts as down
  releaseHeight: 0.09,  // hysteresis: keep the contact until it is clearly up
  speed: 0.90,          // m/s of horizontal sole glide that still counts as planted
  releaseSpeed: 1.30,   // hysteresis: only a real swing is faster than this
  lift: 0.30,           // m/s of vertical motion that still counts as planted
  releaseLift: 0.60,
  attack: 0.04,         // s to ramp a contact in
  release: 0.12,        // s to ramp it out
  hold: 0.25,           // s a stance survives with no contact before it ends
  // s the HOLD survives with no contact; the anchor outlives the force.
  // One frame of this reconstruction's tracking glitches moves the ankle 15-23
  // cm, and that glitch is itself what makes the contact gate open - so a
  // force that let go after a single frame let the pose snap onto the glitch
  // and snap back on the next frame. That read as the leg popping exactly when
  // the lock was on. Two frames of grace rides the glitch out while a real
  // swing (0.25 s and up) still releases well before the foot lands.
  weightHold: 0.10,
  // m the sole may leave its anchor before it is a new step.
  //
  // This is also the largest hold a stance can ever ask the leg IK for, which
  // is what makes its size matter. At 0.30 a stance let the sole get 28 cm
  // from its anchor - a leg 0.729 m long standing at 94-98% of its extension,
  // so the IK was being asked to bend past what the bones allow - and when the
  // stance finally ended the whole correction was dropped in ONE frame: the
  // foot teleported 15-23 cm while the reconstruction itself moved it only
  // 0.4-3.7 cm. Measured over the 722-frame replay that was the single
  // largest source of the legs popping; the worst leg jump was 67.8 deg at
  // 0.30. At 0.13 the worst is 24 deg against 21.7 deg with the lock off, and
  // no frame exceeds 25 deg. The foot's total travel while it is on the floor
  // falls from 22.9 cm/s (lock off) to 14.5, and hold accuracy does not suffer
  // (error p90 1.7-1.8 cm): a stance ends while the correction is still small,
  // instead of after it has grown into something the leg cannot take.
  step: 0.13,
  // Fraction of `step` over which the force fades to zero. The correction is
  // proportional to the drift (that is the whole job), so ending a stance at
  // full force drops whatever it had grown to in one frame. Fading over the
  // last quarter means the leg is already back to its natural pose when the
  // stance ends: the release stops being an event at all. Measured on the
  // replay the worst held-to-landing jump was still 10 cm before this.
  taper: 0.25,
  filter: 0.08,         // s low-pass on the position used by the gate and step test
  creep: 0.0,           // m/s the anchor may drift; 0 means a held foot never drifts
  leash: 0.45,          // m: safety limit, drag the anchor past what a leg can absorb
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
    this.lift = 0;                      // m/s of vertical sole motion, low-passed
    this.error = 0;
    this.lost = 0;                      // s since the last contact frame
    this.anchor = new THREE.Vector2();  // root-relative x, z
    this.last = new THREE.Vector2();    // previous natural root-relative x, z
    this.filt = new THREE.Vector2();    // low-passed natural position
    this.lastFilt = new THREE.Vector2();// previous low-passed position
    this.filtY = 0;                     // low-passed height over the bind sole
    this.lastFiltY = 0;
    this.err = new THREE.Vector2();     // anchor - natural ankle (x, z)
    this.fresh = false;                 // this frame opened a new stance
  }

  clear() {
    this.weight = 0;
    this.planted = false;
    this.anchored = false;
    this.seen = false;
    this.speed = 0;
    this.speedRaw = 0;
    this.lift = 0;
    this.error = 0;
    this.lost = 0;
    this.filtY = 0;
    this.lastFiltY = 0;
    this.fresh = false;
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
   * Consume this frame's NATURAL foot positions.
   *
   * `left`/`right` are the lowest skin points of each foot in world space and
   * `ankles` the two foot bones (ankles). The sole is what decides CONTACT -
   * it is the thing on the floor - but the ANKLE is what the hold pins: the
   * lowest skin point migrates from heel to toe as the shoe pitches, and a
   * constraint on that point has to translate the whole foot to follow it.
   * Measured on the live capture, pinning the sole still let the foot bone
   * move 5-6 cm per stance, purely as the pitch changed; pinning the ankle
   * leaves the sole free to roll, which is what a real planted foot does.
   *
   * `groundY` is the bind-pose floor height, `rootX`/`rootZ` the avatar root,
   * and `strength` the share of the pass to apply (0 in a close-up).
   */
  update(left, right, ankles, groundY, rootX, rootZ, dt, strength = 1) {
    const C = this.contact;
    dt = Number.isFinite(dt) && dt > 0
      ? Math.min(0.1, Math.max(0.001, dt)) : 1 / 60;
    const active = Number.isFinite(strength) && strength > 1e-3;
    const soles = [left, right];
    for (let i = 0; i < 2; i++) {
      const p = soles[i];
      const foot = this.feet[i];
      foot.fresh = false;
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
      const A = ankles && ankles[i];
      const u = this._u.set(
        (A && Number.isFinite(A.x) ? A.x : p.x) - rootX,
        (A && Number.isFinite(A.z) ? A.z : p.z) - rootZ);

      // Spike-free position. A one-frame tracking jump must not be mistaken
      // for the foot walking somewhere, so the step test and the contact gate
      // read this. The gate especially: on a live feed the raw per-frame
      // jitter of a standing person already reads as 0.3 m/s, so a gate fed
      // the raw foot finds no contact at all and the plant silently does
      // nothing. A low-pass keeps a sustained glide or step at its true speed
      // while averaging zero-mean jitter away.
      //
      // The height is low-passed as well, but only to differentiate it: a
      // stance is a foot that is DOWN and STAYS down. Horizontal glide is not
      // a useful contact signal on this reconstruction (measured below), so
      // it is the vertical derivative that separates a stance from a swing -
      // a foot being lifted moves vertically at 0.5-1.5 m/s while a planted
      // one barely moves at all.
      if (!foot.seen) {
        foot.filt.copy(u);
        foot.lastFilt.copy(u);
        foot.filtY = p.y - groundY;
        foot.lastFiltY = foot.filtY;
      } else {
        foot.filt.lerp(u, damp(dt, C.filter));
        foot.filtY += (p.y - groundY - foot.filtY) * damp(dt, C.filter);
      }
      foot.speedRaw = foot.seen
        ? Math.hypot(u.x - foot.last.x, u.y - foot.last.y) / dt : 0;
      const speed = foot.seen
        ? Math.hypot(foot.filt.x - foot.lastFilt.x,
          foot.filt.y - foot.lastFilt.y) / dt : 0;
      foot.speed = speed;
      foot.lift = foot.seen
        ? Math.abs(foot.filtY - foot.lastFiltY) / dt : 0;
      foot.last.copy(u);
      foot.lastFilt.copy(foot.filt);
      foot.lastFiltY = foot.filtY;
      foot.seen = true;

      const height = p.y - groundY;
      const down = height < (foot.planted ? C.releaseHeight : C.height);
      const slow = speed < (foot.planted ? C.releaseSpeed : C.speed);
      const level = foot.lift < (foot.planted ? C.releaseLift : C.lift);
      const contact = active && down && slow && level;
      foot.planted = contact;
      const wasLost = foot.lost;
      foot.lost = contact ? 0 : foot.lost + dt;

      // A stance opens on a contact frame and is anchored where the sole IS,
      // not where the low-pass thinks it is. The filter lags by ~0.08 s, so
      // anchoring on `filt` started every stance 2-4 cm away from the real
      // sole (0.27 m/s of glide x the filter time constant) and the IK then
      // had to skate that distance back - measured as a 4 cm plant error that
      // took three time constants to correct. The raw position here is the one
      // the viewer sees touch down, which is exactly what a plant must hold.
      // The filter stays where it belongs: in the contact gate, whose job is
      // to ignore one-frame tracking spikes.
      if (!foot.anchored && contact) {
        foot.anchor.copy(u);
        foot.anchored = true;
        // A re-anchor after a 1-2 frame dropout is the SAME stance resuming,
        // not a new step. Marking it fresh made footik.js drop the correction
        // it was carrying - up to 11 cm at weight 0.72, i.e. the last of the
        // leg pops, all of them on frames exactly like these. Only a stance
        // that actually let go (the foot was missing longer than the force
        // keeps up with) may reset it; anything shorter is allowed to glide
        // back at the correction's own time constant.
        foot.fresh = wasLost > C.weightHold;
      }

      // How far the ankle has walked away from its anchor. Past `step` the
      // foot is not being held any more - it has gone somewhere else.
      const stepDist = foot.anchored
        ? Math.hypot(foot.filt.x - foot.anchor.x,
          foot.filt.y - foot.anchor.y) : 0;

      // The anchor survives a missed frame (`hold`) but not a real step.
      const holding = foot.anchored && active &&
        foot.lost <= C.hold && stepDist <= C.step;
      if (!holding) foot.anchored = false;

      // The ANCHOR may outlive the contact by `hold`, but the FORCE may not.
      // Letting the weight stay at 1 through a swing (which is what reading
      // `holding` alone did) meant the IK kept pulling a foot that was already
      // flying to its next step, and then every touchdown snapped the leg back
      // by whatever that pull had grown to - the "legs pop" seen with the lock
      // on. A stance now fades out over `weightHold` + `release`, so the
      // correction is back to ~0 before the foot lands again.
      //
      // The force also fades as the drift approaches `step` (`taper`), because
      // the correction grew with that same drift: releasing at full weight
      // threw away up to 13 cm of pull in one frame, which is the remaining
      // landing pop. See CONTACT.taper.
      const taper = Math.min(1, Math.max(0,
        (C.step - stepDist) / (C.step * C.taper)));
      const driven = holding && foot.lost <= C.weightHold;
      foot.weight += ((driven ? taper : 0) - foot.weight) *
        damp(dt, driven ? C.attack : C.release);

      if (!holding) {
        foot.error = 0;
        foot.err.set(0, 0);
        continue;
      }

      // The anchor is FROZEN while the stance lasts. That is the whole point
      // of the plant: an anchor that follows the natural foot leaves the IK
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
