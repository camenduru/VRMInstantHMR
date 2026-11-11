/**
 * The one-sided floor constraint on the root translation.
 *
 * `soleY` is the lowest sole measured on the POSED avatar while the root is
 * still at `rootY`. Moving the root by d moves every sole by d, so
 * `rootY - (soleY - restSole)` is the absolute root height that leaves the
 * lowest sole exactly on the bind floor. Any target below that pushes the skin
 * through the floor; a target above it is a real hop and is left alone.
 *
 * The crouch term in mocap.js is a feed-forward estimate taken from the
 * person's hip height, but the avatar's legs are aimed by direction with fixed
 * bone lengths, so the two disagree by centimetres on every step. The slow
 * trim cannot catch that - on the bundled dance clip the soles were measured
 * 9 cm under the floor. This rule is the hard stop: it is a solve, not an
 * integrator, so it has no wind-up and cannot be late by a second.
 *
 * Pure on purpose: the retargeter and the harness call the same function.
 */

/** Seconds for the root to recover once a sole is under the floor. */
export const FLOOR_TAU = 0.04;

const MIN_GROUND = 0.001;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

/**
 * @returns {{y: number, floored: boolean}} the root target, and whether the
 *          floor had to lift it (the caller uses the fast recovery time constant).
 */
export function floorTargetY(targetY, rootY, soleY, restSole, ground = 1) {
  if (!Number.isFinite(targetY) || !Number.isFinite(rootY) ||
      !Number.isFinite(soleY) || !Number.isFinite(restSole) ||
      !Number.isFinite(ground) || ground <= MIN_GROUND) {
    return { y: targetY, floored: false };
  }
  const grounded = rootY - (soleY - restSole);
  const lift = (grounded - targetY) * clamp(ground, 0, 1);
  if (lift <= 0) return { y: targetY, floored: false };
  return { y: targetY + lift, floored: true };
}
