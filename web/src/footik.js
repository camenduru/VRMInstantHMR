/**
 * Two-bone leg IK for a planted foot.
 *
 * `footlock.js` decides WHERE a foot landed; this module keeps it there.
 * A root translation cannot do that job: with "follow root" off the root is
 * essentially stationary while the reconstructed leg rotations still skate
 * the sole ~14 cm across a stance. Moving the root to cancel one foot simply
 * drags the whole character (and the other foot) across the floor.
 *
 * The solve is the classic analytic two-bone triangle:
 *
 *     H  hip (upperLeg origin)      a = |K - H|   thigh
 *     K  knee (lowerLeg origin)     b = |A - K|   shin
 *     A  ankle (foot origin)        d = |T - H|
 *     T  ankle target
 *
 *     x = (a^2 - b^2 + d^2) / (2d),  h = sqrt(a^2 - x^2)
 *     K' = H + u*x + pole*h,         u = (T - H)/d
 *
 * Two world-space swing quaternions then put the knee on K' and the ankle on
 * T. Nothing is integrated and there is no lag.
 *
 * STABILITY. An IK solve that changes every frame is a new source of high
 * frequency motion, and the reconstruction's leg directions are noisy. Three
 * things keep the corrected leg from shaking:
 *
 *   1. The applied correction is low-pass filtered (TAU_CORR) instead of
 *      jumping straight to `anchor - sole` on every frame. It also decays
 *      smoothly to zero when contact ends, so there is no release pop.
 *   2. The knee pole is smoothed across frames and re-projected off the
 *      hip->target axis. Reading a raw pole off a nearly-straight leg makes
 *      it flip sign between frames, which shows up as a knee glitch.
 *   3. mocap.js runs its global output filter BEFORE this pass, then a floor
 *      clamp after it, so the IK stays the last writer on the legs (a filter
 *      applied to corrected legs drags the sole straight off its anchor).
 *
 * The shoe's world orientation is restored afterwards, so the IK never twists
 * the foot: only the leg is re-aimed and the shoe keeps the pitch and roll the
 * reconstruction gave it. The sole is then free to roll heel-to-toe over the
 * floor instead of being dragged sideways to keep one contact point still.
 * Vertical grounding stays entirely with grounding.js; this pass is XZ only.
 *
 * Everything is written onto the three-vrm NORMALIZED bones, which is what
 * mocap.js already poses. `vrm.update()` copies them onto the raw skeleton
 * before the skin is rendered.
 */

import * as THREE from 'three';

const DEAD = 0.0015;          // m of error that is pure jitter
// m of applied correction at which the knee is fully aimed by the pole. Below
// it the knee is blended back onto the NATURAL one, so a solve that is doing
// (almost) nothing reproduces the natural leg exactly. Without this the IK
// switched itself off under DEAD and the leg jumped straight to the natural
// knee in one frame - measured 9.7 degrees of shin while the correction itself
// moved only 1 cm. That switch, once per stance boundary, was the "legs
// jumping" the lock-on view still showed at 60 fps.
const KNEE_BLEND = 0.02;
// m: never bend a leg further than this. Measured on the 722-frame live
// capture at a 0.30 m `step`: at 0.20 the cap was hit on 4% of held frames and
// left 16% of them more than 4 cm off target (p90 5.7 cm); at 0.30 nothing hit
// the cap and the p90 error fell to 0.7 cm. With `step` now at 0.13 (see
// footlock.js) a stance ends long before the correction reaches this, so the
// value is a backstop for the raw-vs-filtered overshoot rather than a tuning
// knob - but it also has to stay at or above what `step` allows, or the cap
// itself becomes the thing that lets go.
const MAX_CORRECTION = 0.30;
const REACH = 0.995;          // stay off the fully-straight singular pose
const TAU_CORR = 0.05;        // s: low-pass on the correction itself
const POLE_TAU = 0.12;        // s: how fast the pole may follow a shallow knee
const POLE_TAU_FAST = 0.03;   // s: ... and a deeply bent one, whose plane is real
// m of natural knee offset at which the reconstruction's knee plane is fully
// trusted. Below it the plane is scaled down instead of being normalized: on
// this avatar the leg stands at 94-98% extension, so the natural knee offset
// is 1-2 cm of tracking noise, and normalizing that turns a 2 mm change into a
// 40-degree swing of the knee - which the IK then amplifies by its own 15 cm
// knee offset. Measured: held frames demanded 13 cm of correction (p90 23 cm)
// and the pole turned 8 degrees per frame at p90.
const POLE_FULL = 0.06;
// deg: how far the applied knee may point away from the person's own knee
// plane. The pole is the femur's roll, and a hip reaches roughly 35-45 degrees
// of rotation past neutral; past that the leg reads as twisted off its joint
// rather than tracked. Measured without this cap: the applied pole sat 40-90
// degrees off the natural knee at full IK weight.
const MAX_POLE_TWIST = 35;
const COS_MAX_POLE = Math.cos(MAX_POLE_TWIST * Math.PI / 180);
const SIN_MAX_POLE = Math.sin(MAX_POLE_TWIST * Math.PI / 180);
const MAX_SLEW = 0.3;         // m/s: how fast the correction may shrink on release
// Fraction of the leg's reach the sideways hold may use. At 1.0 the solve
// sits exactly on the straight-leg singularity whenever the correction is at
// the cap; leaving a margin keeps the knee direction defined.
const REACH_BEND = 1.0;

// Exponential approach fraction for a time constant `tau`, so every rate in
// this file is per SECOND and a 60 fps render and a 15 fps feed smooth alike.
// The floor guards a non-positive tau, which would make this explode instead
// of smoothing (a blended tau once went negative and threw the knee 36 degrees
// in a frame).
const damp = (dt, tau) => 1 - Math.exp(-dt / Math.max(1e-4, tau));

export class FootIK {
  constructor(r) {
    this.r = r;
    // Live-tunable: the measurement probes sweep this without a rebuild.
    this.maxCorrection = MAX_CORRECTION;
    this.maxSlew = MAX_SLEW;
    this.tauCorr = TAU_CORR;
    this.reachBend = REACH_BEND;
    this.H = new THREE.Vector3();
    this.K = new THREE.Vector3();
    this.K2 = new THREE.Vector3();
    this.A = new THREE.Vector3();
    this.A2 = new THREE.Vector3();
    this.T = new THREE.Vector3();
    this.u = new THREE.Vector3();
    this.p = new THREE.Vector3();
    this.Kp = new THREE.Vector3();
    this.d1 = new THREE.Vector3();
    this.d2 = new THREE.Vector3();
    this.pq = new THREE.Quaternion();
    this.loc = new THREE.Quaternion();
    this.qD = new THREE.Quaternion();
    this.footQ = new THREE.Quaternion();
    /**
     * Smoothed correction (x, z) actually being applied, per side. This is
     * `anchor - ankle` at FULL weight; the weight is folded in before the
     * low-pass (see plant), so a weight change cannot move the foot in one
     * frame.
     */
    this.corr = {
      left: new THREE.Vector2(),
      right: new THREE.Vector2(),
    };
    /** Smoothed knee pole, kept perpendicular to the hip->target axis. */
    this.pole = {
      left: new THREE.Vector3(0, 0, 1),
      right: new THREE.Vector3(0, 0, 1),
    };
    /**
     * The same pole in the HIP's frame. Storing it there means a turn of the
     * body carries the knee with it in the same frame instead of the world
     * pole lagging a few frames behind and winging the knee mid-turn.
     */
    this.poleLocal = {
      left: new THREE.Vector3(0, 0, 1),
      right: new THREE.Vector3(0, 0, 1),
    };
    this.poleInit = { left: false, right: false };
    this.hipQ = new THREE.Quaternion();
    this.hipQi = new THREE.Quaternion();
    this.poleTmp = new THREE.Vector3();
    /** Where the pole is heading this frame, in the HIP's frame. */
    this.poleTarget = new THREE.Vector3();
    /** Natural knee direction, re-projected off the hip->target axis. */
    this.pNat = new THREE.Vector3();
    this.UP = new THREE.Vector3(0, 1, 0);
    this.hipL = new THREE.Vector3();
    this.hipR = new THREE.Vector3();
    this.hipAxis = new THREE.Vector3();
    this.fwd = new THREE.Vector3();
    /** Magnitude of the last correction per side, for the measurement probes. */
    this.applied = { left: 0, right: 0 };
    /** Weight at the previous call, to tell a held stance from a release. */
    this._prevW = { left: 0, right: 0 };
  }

  /**
   * PELVIS forward, from the hip bone positions alone.
   *
   * `up x (rightHip - leftHip)` is read from where the skeleton's hips actually
   * are, not from the reconstructed joint angles, so it cannot be flipped by
   * noise in one knee. Measured over 1444 bent-leg replay frames it agrees with
   * the natural knee direction at a mean dot of 0.96 and was on the wrong side
   * on only 2 frames - good enough to seed the pole with and to clamp against.
   */
  _bodyForward(out) {
    const b = this.r.b;
    b.leftUpperLeg.getWorldPosition(this.hipL);
    b.rightUpperLeg.getWorldPosition(this.hipR);
    this.hipAxis.subVectors(this.hipR, this.hipL);
    out.crossVectors(this.UP, this.hipAxis);
    if (out.lengthSq() < 1e-10) out.set(0, 0, 1);
    return out.normalize();
  }

  /** Apply a WORLD-space rotation to a bone without disturbing its parent. */
  _delta(bone, qWorld) {
    bone.parent.getWorldQuaternion(this.pq);
    this.loc.copy(this.pq).invert().multiply(qWorld)
      .multiply(this.pq).multiply(bone.quaternion);
    bone.quaternion.copy(this.loc).normalize();
    bone.updateWorldMatrix(false, true);
  }

  /**
   * Hold `side`'s sole at `(anchorX, anchorZ)` with the given weight.
   * Call this EVERY frame, even with weight 0: the smoothed correction then
   * decays to zero instead of being cut off, which is what stops a lifting
   * foot from snapping back to its natural position.
   *
   * `snap` marks the frame a NEW stance opens. The correction is then set
   * outright instead of low-passed, because a leftover correction from the
   * previous stance would otherwise be dragged onto the freshly planted foot:
   * after a release the correction decays at `maxSlew` (0.3 m/s), so a new
   * stance could start carrying 15-20 cm of stale pull in whatever direction
   * the last step was going. Measured on the live capture that was the single
   * biggest remaining source of skate - 37% of fully held frames sat more than
   * 4 cm off their anchor, with the correction pinned at its cap.
   */
  plant(side, anchorX, anchorZ, weight, dt, snap) {
    this.applied[side] = 0;
    const r = this.r;
    const upper = r.b[`${side}UpperLeg`];
    const lower = r.b[`${side}LowerLeg`];
    const foot = r.b[`${side}Foot`];
    if (!upper || !lower || !foot) return 0;

    upper.getWorldPosition(this.H);
    lower.getWorldPosition(this.K);
    foot.getWorldPosition(this.A);

    const a = this.K.distanceTo(this.H);
    const b = this.A.distanceTo(this.K);
    if (a < 1e-4 || b < 1e-4) return 0;

    let tx = 0;
    let tz = 0;
    const w = Math.min(1, Math.max(0, weight || 0));
    if (w > 1e-3 && Number.isFinite(anchorX) && Number.isFinite(anchorZ)) {
      // XZ only: vertical grounding stays entirely with grounding.js.
      // The offset is kept at FULL scale; `w` is applied to the correction
      // when it is used (see below), never baked into the stored target. That
      // matters at both ends of a stance: baking it in means the stored
      // correction still holds 60-80% of a 20 cm pull while the weight ramps
      // down through a swing, and every touchdown (or the moment the weight
      // crosses 0.05) then snaps the leg by that much in a single frame.
      tx = anchorX - this.A.x;
      tz = anchorZ - this.A.z;

      // How far the ankle can move sideways while the hip stays where it is.
      // This avatar stands with the leg at 94-98% of its length, so a purely
      // horizontal target runs out of reach almost immediately: on the live
      // capture the demand was 15 cm at the median and 26 cm at p90, while
      // the geometry allowed 25 cm at the median and 13 cm at p10. One held
      // frame in five was unreachable, and an unreachable target is what
      // makes the leg POP - the radial clamp leaves the leg locked straight
      // and pointing at the target, and a straight leg has no defined knee
      // direction, so the shin whips 25-35 degrees in a frame. Capping the
      // correction at what the leg can bend to trades that pop for a few
      // centimetres of foot slide, which is what a body that cannot reach
      // that far does anyway.
      const reach = (a + b) * REACH;
      const dy = this.A.y - this.H.y;
      const hMax = Math.sqrt(Math.max(0, reach * reach - dy * dy)) *
        this.reachBend;
      const cap = Math.min(this.maxCorrection, hMax);
      const e = Math.hypot(tx, tz);
      if (e > cap) {
        const k = cap / Math.max(1e-6, e);
        tx *= k;
        tz *= k;
      }
    }

    // The offset the viewer sees is `correction * weight`, so THAT is the
    // quantity that has to be continuous. Smoothing the full-scale correction
    // alone leaves a weight change free to move the foot in one frame:
    // measured at 60 fps, the taper taking weight 0.81 -> 0.54 moved the ankle
    // 4.7 cm in a single frame (6.3 degrees of shin) while the correction
    // itself never came near the slew limit.
    //
    // It must not cap how fast the offset GROWS. A symmetric limit at 0.3 m/s
    // cannot track a foot that is already moving at 0.3 m/s, so the correction
    // never builds up and the foot slides at ~60% of its natural speed - which
    // is exactly what a symmetric limit measured on the live feed.
    //
    // The shrink cap is also the wrong tool WHILE the stance is held. If the
    // reconstruction glides the natural sole back over its anchor, the offset
    // the IK needs is ~0 again, so the correction has to follow it down at
    // once; capping that at 0.3 m/s left a 15 cm pull sitting on a foot that
    // was already back on target, and the leg spent the next half second
    // dragging the sole back - measured as the 16-19 cm of stale correction
    // that every new stance used to inherit. So the cap applies only while the
    // contact is being given up (weight falling). A foot that is off the ground
    // carries no NEW correction, and whatever the last stance left is bled off
    // at the same rate rather than cut.
    const wx = tx * w;
    const wz = tz * w;
    const c = this.corr[side];
    const wPrev = this._prevW[side];
    this._prevW[side] = w;
    const hh = Number.isFinite(dt) && dt > 0 ? Math.min(0.1, dt) : 1 / 60;
    const lim = this.maxSlew * hh;
    if (w < 0.05 || snap) {
      // Nothing here is worth carrying: a fresh stance inherits the pull of the
      // previous one, and a foot in the air has nothing to hold at all - but in
      // both cases the leg is still posed at whatever offset the last stance
      // left. Bleed that off at the release rate and let the solve follow it
      // down. Snapping to zero in one frame WAS a pop: a single frame moved the
      // ankle 4.8 cm (7.6 degrees of shin) at 60 fps, because the weight can
      // collapse faster than the offset can unwind.
      const m = Math.hypot(c.x, c.y);
      if (m > lim) c.multiplyScalar((m - lim) / m);
      else c.set(0, 0);
    } else {
      const k = damp(hh, this.tauCorr);
      let nx = c.x + (wx - c.x) * k;
      let nz = c.y + (wz - c.y) * k;
      if (w < wPrev - 1e-4) {  // giving the foot up
        const before = Math.hypot(c.x, c.y);
        const after = Math.hypot(nx, nz);
        if (before - after > lim && after > 1e-9) {
          const m = Math.max(0, before - lim);
          nx *= m / after;
          nz *= m / after;
        }
      }
      c.x = nx;
      c.y = nz;
    }
    const ex = c.x;
    const ez = c.y;
    const mag = Math.hypot(ex, ez);
    if (mag < DEAD) return 0;

    // The shoe keeps whatever orientation the reconstruction gave it.
    foot.getWorldQuaternion(this.footQ);

    this.T.set(this.A.x + ex, this.A.y, this.A.z + ez);
    this.u.subVectors(this.T, this.H);
    let d = this.u.length();
    if (d < 1e-6) return 0;
    d = Math.min((a + b) * REACH,
      Math.max(Math.abs(a - b) + 1e-4, d));
    this.u.normalize();
    this.T.copy(this.H).addScaledVector(this.u, d);

    // Pole: the knee direction from the natural pose, smoothed across frames
    // and re-projected off the (moving) hip->target axis.
    //
    // The pole decides which way the knee points, and with the IK's own knee
    // offset at 13-18 cm (the leg has to bend that far to reach a 20 cm
    // correction) an error in the pole is a TWIST of the whole leg. Three things
    // are needed to keep it still:
    //
    //   1. Trust the natural plane only as far as it exists. A nearly straight
    //      leg has a 1-2 cm knee offset made of tracking noise; normalizing it
    //      makes a 2 mm change a 40-degree turn, and the old code did exactly
    //      that (it accepted anything above 1 mm). The blend rate now scales
    //      with the offset, so a shallow knee barely moves the pole.
    //   2. Keep it in the HIP's frame, not world space. A 0.25-per-frame world
    //      lerp lags a turning body by several frames, which swings the knee
    //      through the floor mid-step. Rotating the stored local pole by the
    //      hip's world rotation follows the body exactly, with no lag.
    //
    //   3. Never point the knee behind the pelvis. The pole is only a
    //      preference the IK aims for, so reflecting it into the forward half
    //      plane cannot move a knee that was already correct - it only catches
    //      the case where the natural plane itself comes out backwards (a
    //      mis-tracked hip, or a stance that opened on noise and locked in).
    //
    // Measured before this: the IK rotated the shin 74 degrees in a single
    // frame while the correction never moved, because the pole had drifted 44
    // degrees off the natural knee plane at full weight.
    const upperParent = upper.parent;
    if (upperParent) {
      upperParent.getWorldQuaternion(this.hipQ);
      this.hipQi.copy(this.hipQ).invert();
    } else {
      this.hipQ.identity();
      this.hipQi.identity();
    }
    // Body forward, projected off the hip->target axis so it is usable as a
    // pole. This is both the seed and the clamp direction.
    const fwd = this.fwd;
    this._bodyForward(fwd);
    fwd.addScaledVector(this.u, -fwd.dot(this.u));
    if (fwd.lengthSq() < 1e-8) fwd.set(0, 0, 1).addScaledVector(this.u, -this.u.z);
    fwd.normalize();
    const raw = this.p.subVectors(this.K, this.H);
    raw.addScaledVector(this.u, -raw.dot(this.u));
    const rawMag = raw.length();
    // Keep the natural knee direction: `raw` lives in this.p, which is about to
    // be overwritten by the stored pole.
    if (rawMag > 1e-6) this.pNat.copy(raw).divideScalar(rawMag);
    const acc = this.poleLocal[side];
    if (!this.poleInit[side]) {
      // Seed from the body forward, NOT from the reconstructed knee. A session
      // that opens on a straight leg has only tracking noise to read (1e-5 m
      // used to be enough), which locked the knee into a random - often
      // backwards - direction for the whole stance.
      acc.copy(fwd).applyQuaternion(this.hipQi);
      this.poleInit[side] = true;
    }
    if (rawMag > 1e-6) {
      // Planar trust is two-sided: a nearly straight leg is noise and barely
      // moves the pole (rawMag/POLE_FULL falls to zero at 0), a leg bent past
      // POLE_FULL has a real, reliable knee plane and is followed at full
      // rate. Without the low end a 1 cm noise offset still moved the pole
      // every frame; without the upper end a deep crouch (the reconstruction
      // snaps into one) left the pole 70 degrees behind the knee it was meant
      // to track, twisting the leg.
      //
      // Where the measurement is noise, the target is NOT the last pole: a
      // held value goes stale while the body keeps moving, and the femur ends
      // up rolled sideways - measured 40-90 degrees off the person's own knee
      // at full IK weight, which is the "impossible rotation" that reads as a
      // twisted leg. It falls back to the pelvis forward instead, which moves
      // with the body and is what a standing knee actually points at (mean dot
      // 0.96 against the natural knee over the replay).
      const tv = Math.min(3, rawMag / POLE_FULL);
      const conf = Math.min(1, tv);
      this.poleTmp.copy(raw).divideScalar(rawMag).applyQuaternion(this.hipQi);
      this.poleTarget.copy(fwd).applyQuaternion(this.hipQi)
        .lerp(this.poleTmp, conf);
      // a straight lerp between near-opposite directions passes through zero
      if (this.poleTarget.lengthSq() < 1e-6) this.poleTarget.copy(fwd);
      this.poleTarget.normalize();
      // The rate is TIME-based, not per call. A fixed 0.25 per call turned the
      // knee four times faster at 60 fps (the render rate) than at 15 fps (a
      // slow tracking feed), so the same reconstruction noise produced a knee
      // that visibly jittered live but read as stable in the replay. `damp`
      // makes the pole take the same wall-clock time to follow at any frame
      // rate.
      const deep = Math.min(1, Math.max(0, tv - 1));
      const tau = Math.max(POLE_TAU_FAST, POLE_TAU + (POLE_TAU_FAST - POLE_TAU) * deep);
      acc.lerp(this.poleTarget, damp(hh, tau)).normalize();
    }
    this.p.copy(acc).applyQuaternion(this.hipQ);
    this.p.addScaledVector(this.u, -this.p.dot(this.u));
    if (this.p.lengthSq() < 1e-8) {
      this.p.copy(fwd);
    } else {
      this.p.normalize();
    }
    // Forward half-plane: reflect a backwards pole about the plane normal to
    // `fwd`. `fwd` is already perpendicular to `u`, so the reflection keeps
    // the pole perpendicular to the hip->target axis.
    if (this.p.dot(fwd) < 0) {
      this.p.addScaledVector(fwd, -2 * this.p.dot(fwd)).normalize();
    }
    // Keep the stored copy consistent with the projected pole (NOT the blended
    // one), so the pole's own smoothing survives frames where the solve is
    // nearly idle.
    acc.copy(this.p).applyQuaternion(this.hipQi).normalize();

    // How much the knee may be aimed by the pole at all. With no applied
    // offset the solve must reproduce the natural leg, so the knee is blended
    // back onto the natural one as the correction approaches zero (quadratic,
    // so it is gone well before DEAD). This is what removes the on/off switch
    // at stance boundaries: the leg now slides between the two aims instead of
    // jumping between them.
    if (rawMag > 1e-6) {
      const g = Math.min(1, mag / KNEE_BLEND);
      if (g < 1) {
        this.p.lerp(this.pNat, 1 - g * g);
        if (this.p.dot(fwd) < 0) {
          this.p.addScaledVector(fwd, -2 * this.p.dot(fwd));
        }
        this.p.normalize();
      }
    }
    // Hard anatomical cap. Whatever the smoothing did, the knee may not point
    // further than a hip can rotate: past MAX_POLE_TWIST from the person's own
    // knee the leg is visibly rolled off its joint. This is the backstop that
    // makes an "impossible" leg impossible, and it is applied to the angle the
    // solve actually uses, not to the stored pole, so a stale pole cannot leak
    // a twist through the KNEE_BLEND path either.
    if (rawMag > 1e-6 && this.p.dot(this.pNat) < COS_MAX_POLE) {
      this.poleTmp.copy(this.p).addScaledVector(this.pNat, -this.p.dot(this.pNat));
      if (this.poleTmp.lengthSq() > 1e-8) {
        this.poleTmp.normalize();
        this.p.copy(this.pNat).multiplyScalar(COS_MAX_POLE)
          .addScaledVector(this.poleTmp, SIN_MAX_POLE)
          .normalize();
        if (this.p.dot(fwd) < 0) {
          this.p.addScaledVector(fwd, -2 * this.p.dot(fwd)).normalize();
        }
      } else {
        // the pole was exactly opposite the natural knee: nothing but the knee
        // itself is a sane direction
        this.p.copy(this.pNat);
      }
    }
    this.pole[side].copy(this.p);

    const x = (a * a - b * b + d * d) / (2 * d);
    const h = Math.sqrt(Math.max(0, a * a - x * x));
    this.Kp.copy(this.H).addScaledVector(this.u, x)
      .addScaledVector(this.p, h);

    // Thigh: swing the knee onto K'.
    this.d1.subVectors(this.K, this.H).normalize();
    this.d2.subVectors(this.Kp, this.H).normalize();
    this.qD.setFromUnitVectors(this.d1, this.d2);
    this._delta(upper, this.qD);

    // Shin: swing the ankle onto T. The thigh rotation has already moved the
    // whole subtree, so re-measure from the new knee/ankle positions.
    lower.getWorldPosition(this.K2);
    foot.getWorldPosition(this.A2);
    this.d1.subVectors(this.A2, this.K2).normalize();
    this.d2.subVectors(this.T, this.K2).normalize();
    this.qD.setFromUnitVectors(this.d1, this.d2);
    this._delta(lower, this.qD);

    // Cancel the rotations the IK imposed on the shoe.
    lower.getWorldQuaternion(this.pq);
    foot.quaternion.copy(this.pq).invert().multiply(this.footQ);
    foot.updateWorldMatrix(false, true);

    this.applied[side] = mag;
    return mag;
  }
}
