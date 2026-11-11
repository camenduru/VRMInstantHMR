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
 *   3. mocap.js runs its global output smoothing AFTER this pass (and
 *      re-solves the floor afterwards), so the final pose the viewer sees is
 *      filtered rather than raw IK output.
 *
 * The shoe's world orientation is restored afterwards. That makes
 * `sole - ankle` the same world offset as in the natural pose, so targeting
 * `ankle + (anchor - sole)` puts the SOLE on the anchor - and keeps its
 * height exactly where grounding.js put it. Vertical grounding stays entirely
 * with grounding.js; this pass is XZ only.
 *
 * Everything is written onto the three-vrm NORMALIZED bones, which is what
 * mocap.js already poses. `vrm.update()` copies them onto the raw skeleton
 * before the skin is rendered.
 */

import * as THREE from 'three';

const DEAD = 0.0015;          // m of error that is pure jitter
const MAX_CORRECTION = 0.09;  // m: never bend a leg further than this
const REACH = 0.995;          // stay off the fully-straight singular pose
const TAU_CORR = 0.06;        // s: low-pass on the correction itself
const POLE_BLEND = 0.25;      // how fast the knee pole may turn

const damp = (dt, tau) => 1 - Math.exp(-dt / tau);

export class FootIK {
  constructor(r) {
    this.r = r;
    // Live-tunable: the measurement probes sweep this without a rebuild.
    this.maxCorrection = MAX_CORRECTION;
    this.H = new THREE.Vector3();
    this.K = new THREE.Vector3();
    this.K2 = new THREE.Vector3();
    this.A = new THREE.Vector3();
    this.A2 = new THREE.Vector3();
    this.S = new THREE.Vector3();
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
    /** Smoothed correction (x, z) still being applied, per side. */
    this.corr = {
      left: new THREE.Vector2(),
      right: new THREE.Vector2(),
    };
    /** Smoothed knee pole, kept perpendicular to the hip->target axis. */
    this.pole = {
      left: new THREE.Vector3(0, 0, 1),
      right: new THREE.Vector3(0, 0, 1),
    };
    /** Magnitude of the last correction per side, for the measurement probes. */
    this.applied = { left: 0, right: 0 };
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
   */
  plant(side, anchorX, anchorZ, weight, dt) {
    this.applied[side] = 0;
    const r = this.r;
    const upper = r.b[`${side}UpperLeg`];
    const lower = r.b[`${side}LowerLeg`];
    const foot = r.b[`${side}Foot`];
    if (!upper || !lower || !foot) return 0;

    upper.getWorldPosition(this.H);
    lower.getWorldPosition(this.K);
    foot.getWorldPosition(this.A);
    const S = r.solePoint(side, this.S);

    let tx = 0;
    let tz = 0;
    const w = Math.min(1, Math.max(0, weight || 0));
    if (w > 1e-3 && Number.isFinite(anchorX) && Number.isFinite(anchorZ)) {
      tx = anchorX - S.x;
      tz = anchorZ - S.z;
      const e = Math.hypot(tx, tz);
      if (e > this.maxCorrection) {
        const k = this.maxCorrection / e;
        tx *= k;
        tz *= k;
      }
      tx *= w;
      tz *= w;
    }

    // Low-pass the correction. A contact is a slowly varying offset, so this
    // costs almost no hold accuracy but removes the frame-to-frame IK jitter
    // that otherwise reads as a shaking knee.
    const c = this.corr[side];
    const k = damp(Number.isFinite(dt) && dt > 0 ? Math.min(0.1, dt) : 1 / 60,
      TAU_CORR);
    c.x += (tx - c.x) * k;
    c.y += (tz - c.y) * k;
    const ex = c.x;
    const ez = c.y;
    const mag = Math.hypot(ex, ez);
    if (mag < DEAD) {
      c.set(0, 0);
      return 0;
    }

    const a = this.K.distanceTo(this.H);
    const b = this.A.distanceTo(this.K);
    if (a < 1e-4 || b < 1e-4) return 0;

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
    // and re-projected off the (moving) hip->target axis. A raw pole off a
    // nearly-straight leg flips sign between frames and glitches the knee.
    const raw = this.p.subVectors(this.K, this.H);
    raw.addScaledVector(this.u, -raw.dot(this.u));
    const cached = this.pole[side];
    if (raw.lengthSq() > 1e-6) {
      raw.normalize();
      cached.lerp(raw, POLE_BLEND);
    }
    this.p.copy(cached).addScaledVector(this.u, -cached.dot(this.u));
    if (this.p.lengthSq() < 1e-8) {
      this.p.set(0, 0, 1).addScaledVector(this.u, -this.u.z);
    }
    this.p.normalize();
    cached.copy(this.p);

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
