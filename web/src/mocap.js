import * as THREE from 'three';
import { mapBlendshapes } from './face-retarget.js';
import { MotionFilter } from './filters.js';
import { FootLock } from './footlock.js';
import { FootIK } from './footik.js';
import { floorTargetY, FLOOR_TAU } from './grounding.js';

/**
 * Retargets COCO-17 keypoints from a mirrored webcam image onto a VRM
 * humanoid. The mirror means the person's right side appears on the right of
 * the screen, which is the avatar's *left* side (it faces the viewer), so
 * sides are swapped below. The keypoints arrive already corrected for the
 * mirror by the pose engine (it flips the frame before inference, so the raw
 * labels would otherwise describe the wrong side), which is why the swap here
 * is by index name and not by screen position.
 *
 * Joint rotation is solved by aiming each bone at the next joint in the
 * chain: rotate the bone so its current direction to the child matches the
 * direction measured in the image. Depth is recovered from limb
 * foreshortening, which makes arms reach toward the camera when the projected
 * segment shortens.
 */

const KP = {
  nose: 0,
  leftEye: 1,
  rightEye: 2,
  leftEar: 3,
  rightEar: 4,
  leftShoulder: 5,
  rightShoulder: 6,
  leftElbow: 7,
  rightElbow: 8,
  leftWrist: 9,
  rightWrist: 10,
  leftHip: 11,
  rightHip: 12,
  leftKnee: 13,
  rightKnee: 14,
  leftAnkle: 15,
  rightAnkle: 16,
};

// bone name -> [joint A, joint B] indices (A~B is the bone direction)
const ARM = {
  left: [KP.rightShoulder, KP.rightElbow, KP.rightWrist], // person right
  right: [KP.leftShoulder, KP.leftElbow, KP.leftWrist],
};
const LEG = {
  left: [KP.rightHip, KP.rightKnee, KP.rightAnkle],
  right: [KP.leftHip, KP.leftKnee, KP.leftAnkle],
};

/**
 * SMPL-H body joints, as FlowHMR writes them (0-21; 22-51 are fingers and the
 * VRM humanoid rig has no finger bones). SMPL's "left" is the *person's* left,
 * so - exactly like the 2D path above - the avatar's left bones read the
 * person's right joints.
 */
const SMPL = {
  pelvis: 0,
  leftHip: 1, rightHip: 2,
  spine: 3,
  leftKnee: 4, rightKnee: 5,
  spine1: 6,
  leftAnkle: 7, rightAnkle: 8,
  spine2: 9,
  leftToe: 10, rightToe: 11,
  neck: 12,
  leftCollar: 13, rightCollar: 14,
  leftShoulder: 16, rightShoulder: 17,
  leftElbow: 18, rightElbow: 19,
  leftWrist: 20, rightWrist: 21,
  head: 15,
};
const SMPL_BODY_JOINTS = 22;

/**
 * How every deformable bone is oriented, beyond pointing at its child.
 *
 * A direction only fixes two of the three degrees of freedom: the missing one
 * is the roll around the bone. For the spine that roll *is* the torso twist;
 * for an arm it is the elbow's hinge orientation; for a foot it is the sole.
 * Each entry names the avatar child that defines the bone's primary axis and
 * an anatomical reference vector:
 *
 *   fwd  - the front of the body segment (spine, legs)
 *   back - the back of the arm at the elbow (upper/lower arm, hand)
 *   up   - the upward normal of the foot (feet)
 *
 * The person's matching reference is measured from extra landmarks when the
 * live worker supplies them (ears/nose, olecranon/cubital fossa, heels) and
 * falls back to the torso frame otherwise.
 */
const FRAME3D = {
  hips: { child: 'spine', ref: 'fwd' },
  spine: { child: 'chest', ref: 'fwd' },
  chest: { child: 'upperChest', ref: 'fwd', fallbackChild: 'neck' },
  upperChest: { child: 'neck', ref: 'fwd' },
  neck: { child: 'head', ref: 'fwd' },
  head: { axis: 'neck', ref: 'fwd' },
  leftShoulder: { child: 'leftUpperArm', ref: 'fwd' },
  rightShoulder: { child: 'rightUpperArm', ref: 'fwd' },
  leftUpperArm: { child: 'leftLowerArm', ref: 'back' },
  leftLowerArm: { child: 'leftHand', ref: 'back' },
  leftHand: { axis: 'forearm', ref: 'back' },
  rightUpperArm: { child: 'rightLowerArm', ref: 'back' },
  rightLowerArm: { child: 'rightHand', ref: 'back' },
  rightHand: { axis: 'forearm', ref: 'back' },
  leftUpperLeg: { child: 'leftLowerLeg', ref: 'fwd' },
  leftLowerLeg: { child: 'leftFoot', ref: 'fwd' },
  leftFoot: { child: 'leftToes', ref: 'up' },
  rightUpperLeg: { child: 'rightLowerLeg', ref: 'fwd' },
  rightLowerLeg: { child: 'rightFoot', ref: 'fwd' },
  rightFoot: { child: 'rightToes', ref: 'up' },
};


const JOINT_CONF = 0.3;
const TAU_HEAD = 0.08;
const TAU_ROOT = 0.09;
const TAU_BLEND = 0.2;

// 3D root motion. Horizontally it is confined to a cone around the spot where
// tracking began: reconstruction depth drifts by centimetres from frame to
// frame and the person really does move, so an unconstrained pelvis would walk
// the avatar off its mark. Vertically the root has to follow, because the legs
// are aimed independently of it - see the centre of gravity in basis3D.
const ROOT_CONE = 0.55;       // metres of horizontal travel allowed
const ROOT_DROP = 0.62;       // how far a deep crouch may lower the avatar
const ROOT_RISE = 0.22;       // and how far a hop may lift it
const TAU_ROOT_Y = 0.13;      // vertical follows the legs, so it tracks a bounce
const TAU_HIP = 0.5;          // how fast the standing height may be relearned
const TAU_TRIM = 1.2;         // how fast the feet settle onto the floor
const ROOT_TRIM = 0.12;       // metres the ground trim may correct

// Close-up shots. A camera that cuts the body at the waist leaves the
// reconstruction no ground to stand on, and it fills the gap with a guess:
// the pelvis drops, the thighs swing forward and the torso pitches away from
// the camera. The 2D joints say which legs were ever in the picture, so those
// frames are retargeted with a neutral lower body instead of the guess.
const TAU_CROP = 0.22;        // seconds for a limb to count as in/out of shot
const CROP_MARGIN = 0.02;     // a joint this far outside the image still counts
const CROP_EDGE = 0.015;      // box this close to the frame edge = body cut off
// How far a hidden arm hangs out from the ribs, as a share of the sideways
// axis: straight down is a soldier, this is a relaxed shoulder.
const ARM_SLACK = 0.16;
// How far a human elbow can flex, in radians, measured as the angle between
// the upper arm and the forearm: a fully straight arm is 0 and the forearm
// touching the biceps is ~2.7. The reconstruction sees a wrist position, not a
// joint, and monocular depth error regularly folds the forearm back into the
// upper arm; past this the avatar's elbow would pass through itself.
//
// There is deliberately no sideways bound. With the upper arm aimed at its
// measured direction and the forearm at its measured direction, the bend's
// azimuth IS the shoulder's internal/external rotation, which a real shoulder
// can carry through a large range - so a bend that looks off-plane against the
// avatar's own (clamped) humerus roll is a twist limit showing, not a joint
// violation. Forcing the forearm into that plane instead rewrites the wrist
// direction the measurement asked for, and on the live feed it collapsed a
// measured 130 degree bend to 10 degrees (see limitElbow).
const ELBOW_MAX = 2.70;       // rad
const TAU_FACE = 0.045;      // expression weight smoothing
const TAU_FACE_HEAD = 0.07;  // facial head pose smoothing
const TAU_FACE_GAZE = 0.09;  // iris jitter is fast; gaze needs its own band
// Radians a full-scale gaze (iris at the eye corner) asks for.
const GAZE_RANGE = { yaw: 0.5, pitch: 0.32 };

// Every expression this layer can write. Kept in one place so a source that
// goes away eases all of them to rest instead of leaving the last pose stuck.
const FACE_ZERO = Object.freeze({
  aa: 0, ih: 0, ou: 0, ee: 0, oh: 0,
  blink: 0, blinkLeft: 0, blinkRight: 0,
  happy: 0, angry: 0, sad: 0, relaxed: 0, surprised: 0,
});

// One Euro filter bands for the two fast, noisy signals: joint positions
// (pixels) and limb directions (radians). A fixed time constant has to choose
// between lag on a fast move and jitter when still; these raise the cutoff
// with the signal's own speed, so a still limb is smoothed hard and a swinging
// one is followed closely.
const POS_FILTER = { minCutoff: 1.2, beta: 0.04 };
const DIR_FILTER = { minCutoff: 1.0, beta: 0.7 };

// How fast the per limb foreshortening reference may grow (fraction of itself
// per frame) and how far it is allowed to stray from this avatar's own bone.
const REF_MAX = { rise: 0.01, decay: 0.9995, floor: 0.6, cap: 1.2 };

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const damp = (dt, tau) => 1 - Math.exp(-dt / tau);
const UP = new THREE.Vector3(0, 1, 0);

/** One Euro blend factor: more smoothing while `speed` is low, less while high. */
function oneEuro(dt, speed, { minCutoff, beta }) {
  const cutoff = minCutoff + beta * Math.max(0, speed);
  return 1 - Math.exp(-dt * 2 * Math.PI * cutoff);
}

const AXIS_Y = new THREE.Vector3(0, 1, 0);
const AXIS_Z = new THREE.Vector3(0, 0, 1);

// rad/s: how fast a bone may roll about its own axis. See poseBone: the roll
// comes from a reference direction whose projection is 1/sin amplified near
// the poses where a limb points along it, and the live feed turned a femur
// 20-50 degrees in one frame that way. A human turn-out peaks near 8 rad/s,
// so this clips only what a body cannot do. Kept in the file's measured style.
const ROLL_MAX_RATE = 12.2;

/**
 * How far a bone may be rolled about its own axis, in radians, measured from
 * the roll a pure swing carries (the neutral internal/external rotation).
 *
 * The roll is the one degree of freedom a direction-only transfer leaves to a
 * reference, and the reference is a noisy measurement: on the live dance feed
 * where the arm crosses the chest, the crease landmarks asked a humerus for
 * more than a full turn (-388 degrees, measured on one arm) and the bone
 * chased it at the rate limit the whole way - a shoulder spinning inside its
 * socket, which no body can do. The direction is unaffected by this clamp: the
 * arm still points where the person's arm points, it just stops winding around
 * itself past what the joint has.
 *
 * A clavicle has no roll to speak of - its axis is the shoulder line and the
 * upper arm is aimed absolutely from it, so rolling it only shears the
 * shoulder's skin. It keeps the rest roll (limit 0) and the swing does the
 * lifting. A humerus has about 70 degrees of internal and 100 of external
 * rotation, so 100 either way is the outer edge of what a shoulder can make;
 * the wrist rolls about 60 degrees each way. The forearm is absent on purpose:
 * its reference is derived from the upper arm's, so its measured roll already
 * includes a large geometric offset, and clamping that would bend the elbow
 * out of its own hinge. Spine, legs and feet keep the full circle.
 */
const TWIST_LIMIT = {
  leftShoulder: 0, rightShoulder: 0,
  leftUpperArm: 1.75, rightUpperArm: 1.75,
  leftHand: 1.05, rightHand: 1.05,
};

/**
 * rad/s a clamped bone may roll. Slower than the leg limit because these are
 * shoulders: chasing a bad crease sample at 700 deg/s is exactly the spin this
 * table exists to stop, and a real internal rotation peaks nearer 300 deg/s.
 */
const TWIST_RATE = {
  leftShoulder: 3.5, rightShoulder: 3.5,
  leftUpperArm: 5.5, rightUpperArm: 5.5,
};

/**
 * rad/s the forearm DIRECTION may turn.
 *
 * limitElbow() pins a bent wrist into the elbow's hinge plane, and where the
 * reconstructed forearm lies almost along that hinge (an arm crossing the
 * chest is the common case) the plane's normal is the noisy thing, so the pin
 * can flip to the other side of the joint in one frame. Measured on the live
 * feed: the person's forearm moved at most 15.6 degrees per frame while the
 * applied direction jumped 149 degrees - the forearm visibly swapping places
 * with itself. A real elbow at full flexion peaks near 15 rad/s, so 20 clips
 * only the impossible frames and is invisible on a real bend.
 */
const FOREARM_MAX_SLEW = 20;

export class MocapRetargeter {
  constructor(vrm, root) {
    this.vrm = vrm;
    this.root = root;

    const get = (name) => vrm.humanoid.getNormalizedBoneNode(name);
    this.b = {
      hips: get('hips'),
      spine: get('spine'),
      chest: get('chest'),
      upperChest: get('upperChest'),
      neck: get('neck'),
      head: get('head'),
      leftShoulder: get('leftShoulder'),
      rightShoulder: get('rightShoulder'),
      leftUpperArm: get('leftUpperArm'),
      leftLowerArm: get('leftLowerArm'),
      leftHand: get('leftHand'),
      rightUpperArm: get('rightUpperArm'),
      rightLowerArm: get('rightLowerArm'),
      rightHand: get('rightHand'),
      leftUpperLeg: get('leftUpperLeg'),
      leftLowerLeg: get('leftLowerLeg'),
      leftFoot: get('leftFoot'),
      leftToes: get('leftToes'),
      rightUpperLeg: get('rightUpperLeg'),
      rightLowerLeg: get('rightLowerLeg'),
      rightFoot: get('rightFoot'),
      rightToes: get('rightToes'),
    };
    if (!this.b.chest && this.b.upperChest) this.b.chest = this.b.upperChest;
    if (!this.b.chest) this.b.chest = this.b.spine;

    // Global output smoothing (see filters.js): one filter bank over the whole
    // rig, configured from the panel. Inert until main.js tunes it, so the
    // retargeter behaves exactly as before with smoothing off.
    this.smoothing = new MotionFilter();
    // Every bone once, even when two names share a node (a rig with no
    // `chest` aliases it to `upperChest`/`spine`; filtering the same bone
    // twice would double its smoothing and its lag).
    this._smoothBones = [];
    {
      const seen = new Set();
      for (const [name, bone] of Object.entries(this.b)) {
        if (!bone || seen.has(bone)) continue;
        seen.add(bone);
        this._smoothBones.push([name, bone]);
      }
    }

    this.smoothPx = new Float32Array(17 * 2);    this.scores = new Float32Array(17);
    this.hasJoint = new Uint8Array(17);
    this.blend = 0;
    this.head = { yaw: 0, roll: 0 };
    // Facial layer: MediaPipe gives a head quaternion (world space, since the
    // frame it sees is already mirrored like the preview) and ARKit-style
    // blendshape weights, which are mapped onto the VRM expression presets.
    this.face = {
      q: new THREE.Quaternion(),
      target: new THREE.Quaternion(),
      headW: 0,
      weights: Object.create(null),
      // rolling "wide open" reference for this speaker, so the mouth opens
      // fully at whatever jaw range the face/camera actually produces
      mouthRef: 0.25,
      fresh: false,
    };
    this._faceNames = null;
    // Audio2Face playback: when set ({blend, eye}) it replaces the webcam
    // face source until cleared, so the avatar performs a baked track.
    this.externalFace = null;
    // Browser face tracker: set by main.js while the in-page MediaPipe tracker
    // is running. It takes priority over the server's face layer because it
    // carries gaze and per-eye blink as well as the shared blendshapes, so a
    // close-up can drive the whole VRM face without the body in frame.
    this.faceOverride = null;
    // Eye gaze, in image-normalised units (x right, y down), and the smoothed
    // value actually applied to the look-at target.
    this.gaze = { x: 0, y: 0, w: 0 };
    this.gazeSmoothed = { x: 0, y: 0 };
    this._lookAtTarget = null;
    this.facing = 1; // +1 person faces the camera, -1 they turned away
    this.dirState = new Map();
    // longest each limb has been seen to project, relative to this avatar's
    // own bone: the reference foreshortening is measured against
    this.limbRef = new Map();
    this.torsoPx = 0;
    // on the instance so the bands can be retuned without touching the maths
    this.filter = { pos: { ...POS_FILTER }, dir: { ...DIR_FILTER } };
    this.refMax = { ...REF_MAX };

    // Captured before any animation runs: the head is driven relative to its
    // rest orientation so a bad idle clip can never flip it around.
    this.restHeadLocal = this.b.head
      ? this.b.head.quaternion.clone()
      : null;
    this.rest = this.measure();
    // The soles, not the ankle bones, are what touches the floor: mapSoles()
    // picks the lowest skin points per foot while the avatar is still in its
    // bind pose, which is the only moment the skin and the bones agree.
    this.rest.sole = this.mapSoles();
    this._ankleDrop = Math.max(0,
      (this.rest.ankle ?? this.rest.sole) - this.rest.sole);
    // Full-orientation transfer: each bone's rest frame (primary axis toward
    // its child, plus an anatomical roll reference) is captured once, before
    // any idle animation can overwrite the rig. See FRAME3D.
    this._frames = Object.create(null);
    this._lm3 = Object.create(null);
    this._lmRaw = new THREE.Vector3();
    this._captureFrames();
    this._fAxis = new THREE.Vector3();
    this._fRef = new THREE.Vector3();
    this._fDir = new THREE.Vector3();
    this._fSpin = new THREE.Vector3();
    this._fQa = new THREE.Quaternion();
    this._fQb = new THREE.Quaternion();
    this._fQc = new THREE.Quaternion();
    this._fQw = new THREE.Quaternion();
    this._fQp = new THREE.Quaternion();
    /** Cross product scratch for the roll measured in poseBone. */
    this._fRoll = new THREE.Vector3();
    /** Last applied roll per bone (rad), for the rate limit. */
    this._rollPrev = Object.create(null);
    /** Frame time, so poseBone's rate limit is per second not per call. */
    this._dt = 1 / 60;
    // body frame in avatar space, refreshed by basis3D every frame
    this._upBody = new THREE.Vector3(0, 1, 0);
    this._fwdBody = new THREE.Vector3(0, 0, 1);
    this._rightBody = new THREE.Vector3(1, 0, 0);
    this._hipFwd = new THREE.Vector3(0, 0, 1);
    this._headFwd = new THREE.Vector3(0, 0, 1);
    this._t1 = new THREE.Vector3();
    this._t2 = new THREE.Vector3();
    this._t3 = new THREE.Vector3();
    this._t4 = new THREE.Vector3();
    this._t5 = new THREE.Vector3();
    this._t6 = new THREE.Vector3();
    this._t7 = new THREE.Vector3();
    this._v1 = new THREE.Vector3();
    this._v2 = new THREE.Vector3();
    this._q1 = new THREE.Quaternion();
    this._q2 = new THREE.Quaternion();
    this._q3 = new THREE.Quaternion();
    this._q4 = new THREE.Quaternion();
    this._before = new THREE.Quaternion();
    this._headTarget = new THREE.Quaternion();
    this._target = new THREE.Vector3();
    this._soleA = new THREE.Vector3();
    this._soleB = new THREE.Vector3();
    this._ankleA = new THREE.Vector3();
    this._ankleB = new THREE.Vector3();
    this._soleTmp = new THREE.Vector3();
    // Foot planting: footlock.js detects contact on the natural pose and
    // footik.js holds a planted sole with a two-bone leg IK. A root offset
    // cannot do this - the skate comes from leg rotations, so translating the
    // root only drags the whole avatar. main.js can switch the pass off.
    this.footLock = new FootLock();
    this.footIK = new FootIK(this);
    this._eyePos = new THREE.Vector3();
    this._eyeA = new THREE.Vector3();
    this._eyeB = new THREE.Vector3();
    this._eyeC = new THREE.Vector3();

    // 3D (FlowHMR) path: raw joints as they arrive, the same joints in avatar
    // world space, and the frame axes that map one onto the other
    this._raw3 = Array.from({ length: SMPL_BODY_JOINTS },
      () => new THREE.Vector3());
    this._j3 = Array.from({ length: SMPL_BODY_JOINTS },
      () => new THREE.Vector3());
    this._midHip = new THREE.Vector3();
    this._midSh = new THREE.Vector3();
    this._up3 = new THREE.Vector3();
    this._side3 = new THREE.Vector3();
    this._fwd3 = new THREE.Vector3();
    this._right3 = new THREE.Vector3();
    this._d3 = new THREE.Vector3();
    this._personTorso = 0;
    // centre of gravity: person's pelvis height over the support foot, their
    // standing height, and the leg length it is expressed as
    this.hipRef = 0;
    this._hipHeight = 0;
    this._personLeg = 0;
    this._trim = 0;
    this.rootRef = null;
    // How much of each leg (person's left/right) the shot actually shows, and
    // how far the frame as a whole is a close-up. Both are smoothed weights,
    // so a leg that walks back into shot eases in instead of snapping.
    this._legSeen = { left: 1, right: 1 };
    // The same question, per arm and per segment: an arm cut by the frame is
    // an arm the reconstruction guessed, and it guessed a shake. `upper` is
    // shoulder-to-elbow, `lower` elbow-to-wrist.
    this._armSeen = {
      left: { upper: 1, lower: 1 },
      right: { upper: 1, lower: 1 },
    };
    this._armOut = new THREE.Vector3();
    this._armHang = new THREE.Vector3();
    this._armUpper = new THREE.Vector3();
    this._armFore = new THREE.Vector3();
    this._armKnuck = new THREE.Vector3();
    this._armBack = new THREE.Vector3();
    this._armBackP = { left: new THREE.Vector3(), right: new THREE.Vector3() };
    this._armBackOk = { left: false, right: false };
    this._armForeBack = new THREE.Vector3();
    this._elbowU = new THREE.Vector3();
    this._elbowB = new THREE.Vector3();
    // Last forearm direction limitElbow() actually applied, per side. The
    // reconstruction's wrist can jump a long way at the poses where the arm
    // crosses the chest, so the applied direction is slew limited against this
    // (see FOREARM_MAX_SLEW).
    this._foreDir = { left: new THREE.Vector3(), right: new THREE.Vector3() };
    this._foreDirOk = { left: false, right: false };
    this._slewAxis = new THREE.Vector3();
    this._slewQ = new THREE.Quaternion();
    this._armPos = new THREE.Vector3();
    this._armChest = new THREE.Vector3();
    this._crop = 0;
    this._cropUp = new THREE.Vector3();
    this._cropTarget = new THREE.Vector3();
    this._cropQ = new THREE.Quaternion();
    this._cropQw = new THREE.Quaternion();
    // mid-hip displacement from the calibration point, in body axes; the root
    // motion source once joint directions are pelvis-relative
    this._rootDelta = new THREE.Vector3();
    // Live reconstruction drifts by tens of centimetres even when the person
    // dances on the spot, which reads as floating. Keep the avatar on its mark
    // unless the caller explicitly opts into following the measured travel.
    this.rootFollow = false;

    this.enabled = true;
  }

  /**
   * Capture the rest orientation of every deformable bone.
   *
   * The avatar is still in its bind pose here (the retargeter is built before
   * the idle clip is loaded), so the vector to each child and the anatomical
   * reference can be expressed in the bone's own local frame. At runtime the
   * target world orientation is the rotation that sends those two local axes
   * onto the measured ones; the third axis follows from the cross product.
   */
  _captureFrames() {
    const p = (node) => node.getWorldPosition(new THREE.Vector3());
    try {
      const lSh = p(this.b.leftUpperArm);
      const rSh = p(this.b.rightUpperArm);
      const lHip = p(this.b.leftUpperLeg);
      const rHip = p(this.b.rightUpperLeg);
      const upW = lSh.clone().add(rSh).multiplyScalar(0.5)
        .sub(lHip.clone().add(rHip).multiplyScalar(0.5));
      if (upW.lengthSq() < 1e-8) upW.set(0, 1, 0);
      upW.normalize();
      const sideW = rSh.clone().sub(lSh);
      if (sideW.lengthSq() < 1e-8) sideW.set(-1, 0, 0);
      sideW.normalize();
      const fwdW = new THREE.Vector3().crossVectors(upW, sideW);
      if (fwdW.lengthSq() < 1e-8) fwdW.set(0, 0, 1);
      fwdW.normalize();

      // The feet are the most reliable facing cue: arms may be forward in an
      // A-pose, while the toes always follow the body's forward axis.
      const toeFwd = new THREE.Vector3();
      let toes = 0;
      for (const side of ['left', 'right']) {
        const foot = this.b[`${side}Foot`];
        const tip = this.b[`${side}Toes`];
        if (!foot || !tip) continue;
        toeFwd.add(p(tip).sub(p(foot)));
        toes += 1;
      }
      if (toes) {
        toeFwd.addScaledVector(upW, -toeFwd.dot(upW));
        if (toeFwd.lengthSq() > 1e-8) {
          toeFwd.normalize();
          if (toeFwd.dot(fwdW) < 0) toeFwd.negate();
          fwdW.copy(toeFwd);
        }
      }

      const restRef = {
        fwd: fwdW,
        back: fwdW.clone().negate(),
        up: upW,
      };
      const qInv = new THREE.Quaternion();
      const qBone = new THREE.Quaternion();
      for (const name of Object.keys(FRAME3D)) {
        const def = FRAME3D[name];
        const bone = this.b[name];
        if (!bone) continue;
        let child = def.child ? this.b[def.child] : null;
        if (child === bone) child = def.fallbackChild ? this.b[def.fallbackChild] : null;
        let axisW;
        if (def.axis === 'up') {
          axisW = upW.clone();
        } else if (def.axis === 'neck') {
          if (!this.b.neck) continue;
          axisW = p(bone).sub(p(this.b.neck));
        } else if (def.axis === 'forearm') {
          // the hand continues the forearm at rest
          const lower = this.b[name.replace('Hand', 'LowerArm')];
          if (!lower) continue;
          axisW = p(bone).sub(p(lower));
        } else if (child) {
          axisW = p(child).sub(p(bone));
        } else {
          continue;
        }
        if (axisW.lengthSq() < 1e-8) continue;
        axisW.normalize();
        bone.getWorldQuaternion(qBone);
        qInv.copy(qBone).invert();
        const axisL = axisW.clone().applyQuaternion(qInv).normalize();
        let refL = (restRef[def.ref] || fwdW).clone().applyQuaternion(qInv);
        refL.addScaledVector(axisL, -refL.dot(axisL));
        if (refL.lengthSq() < 1e-8) {
          refL.crossVectors(axisL, Math.abs(axisL.y) < 0.9 ? UP : AXIS_Z);
          if (refL.lengthSq() < 1e-8) continue;
        }
        refL.normalize();
        this._frames[name] = { axis: axisL, ref: refL };
      }
    } catch (err) {
      console.warn('[mocap] could not capture bone frames', err);
    }
  }

  measure() {
    const vrm = this.vrm;
    vrm.scene.updateMatrixWorld(true);
    const p = (node) => node.getWorldPosition(new THREE.Vector3());
    const fallback = { torso: 0.5, upperArm: 0.27, lowerArm: 0.25,
      upperLeg: 0.42, lowerLeg: 0.42, leg: 0.84 };

    try {
      const lSh = p(this.b.leftUpperArm);
      const rSh = p(this.b.rightUpperArm);
      const lHip = p(this.b.leftUpperLeg);
      const rHip = p(this.b.rightUpperLeg);
      const shMid = lSh.clone().add(rSh).multiplyScalar(0.5);
      const hipMid = lHip.clone().add(rHip).multiplyScalar(0.5);
      const upperLeg = p(this.b.leftUpperLeg)
        .distanceTo(p(this.b.leftLowerLeg)) || fallback.upperLeg;
      const lowerLeg = p(this.b.leftLowerLeg)
        .distanceTo(p(this.b.leftFoot)) || fallback.lowerLeg;
      return {
        torso: shMid.distanceTo(hipMid) || fallback.torso,
        upperArm: p(this.b.leftUpperArm).distanceTo(p(this.b.leftLowerArm)) ||
          fallback.upperArm,
        lowerArm: p(this.b.leftLowerArm).distanceTo(p(this.b.leftHand)) ||
          fallback.lowerArm,
        upperLeg,
        lowerLeg,
        // hips above the ankle with the leg straight: the height the root has
        // to keep when the person's own legs are straight and the root has to
        // drop by exactly as much as the legs fold
        leg: upperLeg + lowerLeg,
        // bind-pose ankle height, kept as the fallback reference for rigs
        // whose skin could not be scanned; the trim itself grounds on the
        // soles (mapSoles), because an ankle is not a sole
        ankle: Math.min(p(this.b.leftFoot).y, p(this.b.rightFoot).y),
      };
    } catch (err) {
      console.warn('[mocap] could not measure avatar, using defaults', err);
      return fallback;
    }
  }

  /**
   * The avatar's soles, as probe points on the foot bones.
   *
   * Grounding on the ankle bone is what lets the model sink: an ankle is not a
   * sole. Point the toes and the ankle rises while the shoe goes through the
   * floor, and the bones still report a perfectly sensible height - which is
   * exactly the "the mesh sinks but the bones are fine" symptom. The lowest
   * skin vertices at bind do follow the foot, so a handful of them per foot
   * gives the real contact height under any ankle pitch.
   *
   * Returns the bind-pose sole height: the ground the trim holds the skin on.
   */
  mapSoles() {
    this._soleSides = { left: [], right: [] };
    const groups = [
      { side: 'left', bone: this.b.leftFoot },
      { side: 'left', bone: this.b.leftToes },
      { side: 'right', bone: this.b.rightFoot },
      { side: 'right', bone: this.b.rightToes },
    ].filter((g) => g.bone);
    if (!groups.length) return this.rest.ankle ?? 0;

    this.vrm.scene.updateMatrixWorld(true);
    const v = new THREE.Vector3();
    const found = [];
    let low = Infinity;
    this.vrm.scene.traverse((object) => {
      if (!object.isMesh || !object.geometry) return;
      const pos = object.geometry.attributes &&
        object.geometry.attributes.position;
      if (!pos) return;
      const step = Math.max(1, Math.floor(pos.count / 4000));
      for (let i = 0; i < pos.count; i += step) {
        v.fromBufferAttribute(pos, i).applyMatrix4(object.matrixWorld);
        if (v.y < low) low = v.y;
        found.push(v.x, v.y, v.z);
      }
    });
    if (!found.length || !Number.isFinite(low)) return this.rest.ankle ?? 0;

    // Only the skin near the floor can touch it; anything above is a shin.
    const near = low + 0.035;
    const anchors = groups.map((g) =>
      g.bone.getWorldPosition(new THREE.Vector3()));
    const kept = groups.map(() => []);
    for (let i = 0; i < found.length; i += 3) {
      const y = found[i + 1];
      if (y > near) continue;
      v.set(found[i], y, found[i + 2]);
      let best = 0;
      let bestD = Infinity;
      for (let g = 0; g < groups.length; g++) {
        const d = anchors[g].distanceToSquared(v);
        if (d < bestD) {
          bestD = d;
          best = g;
        }
      }
      kept[best].push(v.x, y, v.z);
    }

    // Keep the sole OUTLINE, not just its lowest few vertices. The lowest
    // eight bind vertices all sit in one spot near the middle of the shoe;
    // pitch the ankle and a heel or toe edge that was 2 cm higher at bind
    // becomes the lowest point, and a sample that only knows the middle
    // reports the foot as grounded while the mesh is through the floor.
    // One lowest vertex per small XZ cell covers the whole outline.
    for (let g = 0; g < groups.length; g++) {
      const list = kept[g];
      if (!list.length) continue;
      let cell = 0.02;
      let buckets;
      for (;;) {
        buckets = new Map();
        for (let i = 0; i < list.length; i += 3) {
          const x = list[i], y = list[i + 1], z = list[i + 2];
          const key = `${Math.floor(x / cell)},${Math.floor(z / cell)}`;
          const prev = buckets.get(key);
          if (!prev || y < prev[1]) buckets.set(key, [x, y, z]);
        }
        if (buckets.size <= 160 || cell > 0.2) break;
        cell *= 1.5;
      }
      const points = [];
      for (const [x, y, z] of buckets.values()) {
        points.push(groups[g].bone.worldToLocal(
          new THREE.Vector3(x, y, z)));
      }
      this._soleSides[groups[g].side].push({ bone: groups[g].bone, points });
    }
    return low;
  }

  /**
   * The world position of the lowest skin point of one foot (see mapSoles),
   * or the ankle dropped to the sole when the scan found nothing to probe.
   */
  solePoint(side, out) {
    const groups = this._soleSides && this._soleSides[side];
    let bestY = Infinity;
    if (groups && groups.length) {
      for (let g = 0; g < groups.length; g++) {
        const { bone, points } = groups[g];
        if (!bone || !points.length) continue;
        bone.updateWorldMatrix(true, false);
        for (let i = 0; i < points.length; i++) {
          this._soleTmp.copy(points[i]).applyMatrix4(bone.matrixWorld);
          if (this._soleTmp.y < bestY) {
            bestY = this._soleTmp.y;
            out.copy(this._soleTmp);
          }
        }
      }
    }
    if (Number.isFinite(bestY)) return out;
    const bone = this.b[`${side}Foot`];
    if (!bone) {
      out.set(0, 0, 0);
      return out;
    }
    bone.getWorldPosition(out);
    out.y -= this._ankleDrop || 0;
    return out;
  }

  ingest(keypoints, scores, width, height, dt) {
    if (!Array.isArray(keypoints) || keypoints.length < 17) return;
    for (let i = 0; i < 17; i++) {
      const kp = keypoints[i];
      if (!kp) continue;
      const score = scores?.[i] ?? 0;
      const x = kp[0] * width;
      const y = kp[1] * height;
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const j = i * 2;
      if (!this.hasJoint[i]) {
        this.smoothPx[j] = x;
        this.smoothPx[j + 1] = y;
        this.hasJoint[i] = 1;
      } else if (score >= 0.2) {
        const speed = Math.hypot(x - this.smoothPx[j],
          y - this.smoothPx[j + 1]) / dt;
        const alpha = oneEuro(dt, speed, this.filter.pos);
        this.smoothPx[j] += (x - this.smoothPx[j]) * alpha;
        this.smoothPx[j + 1] += (y - this.smoothPx[j + 1]) * alpha;
      }
      this.scores[i] = score;
    }
  }

  px(i) {
    return { x: this.smoothPx[i * 2], y: this.smoothPx[i * 2 + 1] };
  }

  smoothDir(key, target, dt) {
    let current = this.dirState.get(key);
    if (!current) {
      current = target.clone();
      this.dirState.set(key, current);
    } else {
      const mag = current.length() * target.length();
      const angle = mag > 1e-9
        ? Math.acos(clamp(current.dot(target) / mag, -1, 1))
        : 0;
      current.lerp(target, oneEuro(dt, angle / dt, this.filter.dir));
      if (current.lengthSq() < 1e-5) current.copy(target);
      current.normalize();
    }
    return current;
  }

  /** 2D direction between two joints in avatar-world units (y up). */
  dir2(a, b, k) {
    return new THREE.Vector3((b.x - a.x) * k, -(b.y - a.y) * k, 0);
  }

  /**
   * 2D direction plus a depth component from foreshortening. When the
   * projected limb is shorter than the limb itself, it must be angled
   * toward/away from the camera; we assume toward (+Z).
   *
   * The reference is the longest the limb has ever projected relative to the
   * torso, not this avatar's own bone: the person and the avatar rarely have
   * the same proportions, and measuring against the avatar's bone would push
   * perfectly straight limbs at the camera. `key` names the limb the
   * calibration is shared by, so both sides agree.
   */
  limbDir(a, b, boneLength, k, key) {
    const v = this.dir2(a, b, k);
    const projected = v.length();
    const reference = this.limbReference(key, projected, boneLength, k);
    let z = 0;
    if (reference > 1e-4 && projected < reference) {
      const ratio = projected / reference;
      z = boneLength * Math.sqrt(Math.max(0, 1 - ratio * ratio)) * 0.55;
    }
    v.z = z;
    if (v.lengthSq() < 1e-8) v.set(0, -1, 0);
    return v.normalize();
  }

  /**
   * Calibrated reference length of one limb, in metres.
   *
   * The limb is measured as a fraction of the torso and compared with the same
   * fraction on this avatar, so camera distance and body size cancel. The
   * reference then creeps toward the longest that limb has ever projected:
   * the rise is rate limited, because a keypoint thrown off by the tracker
   * must not convince us the arm is three times its real length and leave
   * every frame afterwards leaning at the camera.
   */
  limbReference(key, projected, boneLength, k) {
    const cap = this.refMax;
    let scale = this.limbRef.get(key);
    if (scale === undefined) scale = cap.floor;
    if (this.torsoPx > 1 && k > 1e-6) {
      const seen = (projected / k) / this.torsoPx /
        (boneLength / this.rest.torso);
      if (Number.isFinite(seen)) {
        if (seen > scale && seen < 2) {
          scale += Math.min(seen - scale, cap.rise);
        } else {
          scale *= cap.decay;
        }
        scale = clamp(scale, cap.floor, cap.cap);
        this.limbRef.set(key, scale);
      }
    }
    return boneLength * scale;
  }

  /**
   * Rotate `bone` so the direction to `child` matches `dir` in world space.
   * `weight` < 1 blends back toward the pose the animation mixer supplied.
   */
  aim(bone, child, dir, weight) {
    if (!bone || !child || weight <= 0.001) return;
    bone.getWorldPosition(this._v1);
    child.getWorldPosition(this._v2);
    this._v2.sub(this._v1);
    if (this._v2.lengthSq() < 1e-10 || dir.lengthSq() < 1e-10) return;
    this._v2.normalize();

    this._q1.setFromUnitVectors(this._v2, this._target.copy(dir).normalize());

    bone.parent.getWorldQuaternion(this._q2);
    bone.parent.getWorldQuaternion(this._q3);
    this._q3.invert().multiply(this._q1).multiply(this._q2);

    this._before.copy(bone.quaternion);
    bone.quaternion.premultiply(this._q3);
    if (weight < 1) bone.quaternion.slerp(this._before, 1 - weight);
    bone.updateWorldMatrix(false, true);
  }

  aimBone(boneName, childName, dir, weight) {
    this.aim(this.b[boneName], this.b[childName], dir, weight);
  }

  /**
   * Set a bone's world orientation from a measured direction and roll
   * reference, then blend with the pose the mixer supplied.
   *
   * `frame.axis`/`frame.ref` are the bone's rest axes in *local* space, so
   * solving `Rw * axis = dir` and `Rw * ref = ref` gives the full world
   * orientation (swing plus the roll a direction-only aim leaves unconstrained).
   *
   * Returns how far the applied roll was moved from the measured one, in
   * radians, so a caller can carry the same turn into a child whose roll is
   * built from this bone's (the elbow hinge follows the shoulder's twist).
   */
  poseBone(name, dir, ref, weight) {
    const frame = this._frames[name];
    const bone = this.b[name];
    if (!frame || !bone || weight <= 0.001) return 0;
    if (!dir || dir.lengthSq() < 1e-10) return 0;

    /** How far the applied roll was moved from the measured one, in radians. */
    let rollAdj = 0;

    const axis = this._fDir.copy(dir).normalize();
    const refW = this._fRef.copy(ref || this._fwdBody);
    refW.addScaledVector(axis, -refW.dot(axis));
    // How much of the measured reference is left PERPENDICULAR to the bone.
    // Near zero, the bone points along the reference and it says nothing about
    // the roll; see the blend below.
    let perp = refW.lengthSq();
    if (perp < 1e-10) {
      refW.crossVectors(axis, Math.abs(axis.y) < 0.9 ? AXIS_Y : AXIS_Z);
      if (refW.lengthSq() < 1e-10) return 0;
      // the measured reference has nothing to say here: let the carried roll
      // win outright rather than snapping to a world axis
      perp = 0;
    }
    refW.normalize();

    // swing the local primary axis onto the measured direction
    this._fQa.setFromUnitVectors(frame.axis, axis);
    // The rest roll carried by that swing: the bone's own reference spun with
    // the bone, ALWAYS perpendicular to the new axis. This is the roll a ball
    // joint produces with no internal/external rotation, so it is both a
    // physically possible pose and a continuous function of the direction.
    this._fSpin.copy(frame.ref).applyQuaternion(this._fQa);
    this._fSpin.addScaledVector(axis, -this._fSpin.dot(axis));
    this._fQw.copy(this._fQa);
    if (this._fSpin.lengthSq() > 1e-10) {
      this._fSpin.normalize();
      // How much of the MEASURED reference is perpendicular to the bone. Near
      // zero the bone points along it and the reference says nothing about the
      // roll: its projected direction is 1/sin(angle) amplified, so a 1 degree
      // change of the leg turned the femur 20-47 degrees in a single frame,
      // measured on the live dance feed while the leg itself moved about 1
      // degree. Fade it out and keep the rest roll instead. Past 26 degrees
      // (sin > 0.44) the measurement is fully trusted, so an ordinary pose is
      // unchanged.
      const trust = clamp((Math.sqrt(perp) - 0.08) / 0.18, 0, 1);
      if (trust < 1) {
        refW.lerp(this._fSpin, 1 - trust);
        if (refW.lengthSq() < 1e-6) refW.copy(this._fSpin);
        refW.normalize();
      }

      // Rate-limit the spin about the bone axis. The roll is measured against
      // the rest roll the swing already carries, so it is ~0 for a bone that
      // merely moves and non-zero only when the person really rotates about the
      // limb; the bone direction cannot feed into it. What is left in it is
      // tracking noise on the reference, and the live feed turned the femur
      // 20-50 degrees in a single frame that way while the leg itself moved
      // 1-3 degrees - a rotation no hip can make. A real turn-out is ~400-500
      // deg/s, so a cap at 700 only ever clips things a body cannot do.
      this._fRoll.crossVectors(this._fSpin, refW);
      const theta = Math.atan2(this._fRoll.dot(axis), this._fSpin.dot(refW));
      // A joint's rotation about its own axis has stops, so hold the roll
      // inside them (see TWIST_LIMIT) before rate limiting it. Clamping first
      // means the limit is what the smoothing below chases, so the bone eases
      // into it and never overshoots by more than one frame of travel.
      const limit = TWIST_LIMIT[name];
      const rate = TWIST_RATE[name] ?? ROLL_MAX_RATE;
      let th = limit === undefined ? theta : clamp(theta, -limit, limit);
      const prev = this._rollPrev[name];
      if (prev !== undefined) {
        let d = th - prev;
        if (d > Math.PI) d -= 2 * Math.PI;
        else if (d < -Math.PI) d += 2 * Math.PI;
        const lim = rate * this._dt;
        if (d > lim) th = prev + lim;
        else if (d < -lim) th = prev - lim;
        if (limit !== undefined) th = clamp(th, -limit, limit);
      }
      this._rollPrev[name] = th;
      if (th !== theta) {
        this._fQb.setFromAxisAngle(axis, th);
        refW.copy(this._fSpin).applyQuaternion(this._fQb).normalize();
        rollAdj = th - theta;
      }

      // then spin about the bone axis until the local reference lands on it
      this._fQb.setFromUnitVectors(this._fSpin, refW);
      this._fQw.premultiply(this._fQb);
    }

    bone.parent.getWorldQuaternion(this._fQp);
    this._fQc.copy(this._fQp).invert().multiply(this._fQw);

    this._before.copy(bone.quaternion);
    bone.quaternion.copy(this._fQc);
    if (weight < 1) bone.quaternion.slerp(this._before, 1 - weight);
    bone.updateWorldMatrix(false, true);
    return rollAdj;
  }

  /** A projected landmark from `pose.landmarks`, or null when absent. */
  lm(name) {
    const v = this._lm3[name];
    return v && v.valid ? v : null;
  }

  update(pose, dt) {
    if (!this.enabled || !this.b.hips) return;
    dt = clamp(dt, 0.001, 0.1);
    this._dt = dt;

    // FlowHMR frames carry exact 3D joints instead of 2D keypoints; they take
    // the whole body path, including the root translation.
    if (pose && Array.isArray(pose.joints3d) &&
        pose.joints3d.length >= SMPL_BODY_JOINTS) {
      this.update3D(pose, dt);
      return;
    }

    // the face layer runs even when no body is in frame (a close-up), so it
    // must happen before the body early-returns below
    this.updateFace(pose, dt);

    const hasPose = !!pose && Array.isArray(pose.keypoints) &&
      pose.keypoints.length >= 17;
    if (hasPose) {
      this.ingest(pose.keypoints, pose.scores, pose.width || 1280,
        pose.height || 720, dt);
    }

    const confident = hasPose &&
      this.scores[KP.leftShoulder] >= JOINT_CONF &&
      this.scores[KP.rightShoulder] >= JOINT_CONF &&
      this.scores[KP.leftHip] >= JOINT_CONF &&
      this.scores[KP.rightHip] >= JOINT_CONF;

    const targetBlend = confident ? 1 : 0;
    this.blend += (targetBlend - this.blend) * damp(dt, TAU_BLEND);
    if (!hasPose && this.blend < 0.002) {
      this.blend = 0;
      // No pose means no contact: relax the plant hold instead of carrying a
      // stale offset into the next tracked segment.
      this.footLock.release(dt);
      this.root.position.lerp(this._target.set(0, 0, 0), damp(dt, TAU_ROOT));
      return;
    }

    const weight = this.blend;
    this.vrm.scene.updateMatrixWorld(true);

    const lSh = this.px(5);
    const rSh = this.px(6);
    const lHip = this.px(11);
    const rHip = this.px(12);
    const shMid = { x: (lSh.x + rSh.x) / 2, y: (lSh.y + rSh.y) / 2 };
    const hipMid = { x: (lHip.x + rHip.x) / 2, y: (lHip.y + rHip.y) / 2 };
    const torsoPx = Math.hypot(shMid.x - hipMid.x, shMid.y - hipMid.y);
    if (torsoPx < 8) return;
    const k = this.rest.torso / torsoPx;
    this.torsoPx = torsoPx;

    // --- which way the person is facing --------------------------------
    // The hip and shoulder lines are *lateral* pairs: they reverse sign when
    // the person turns their back to the camera, and every angle built from
    // such a pair (pelvis up, head roll) has a 180 degree ambiguity. The
    // evidence is only trusted when it is clear - in profile both lines
    // collapse and the previous value is held.
    const shDx = this.px(6).x - this.px(5).x;
    const evidence = shDx + (this.px(12).x - this.px(11).x);
    const scale = Math.max(40, Math.abs(shDx));
    if (Math.abs(evidence) > 0.6 * scale) this.facing = evidence > 0 ? 1 : -1;

    // --- torso ---------------------------------------------------------
    const hipLine = this.dir2(this.px(11), this.px(12), k);
    hipLine.z = 0;
    if (hipLine.lengthSq() > 1e-8) {
      hipLine.normalize();
      const up = new THREE.Vector3(-hipLine.y, hipLine.x, 0).normalize();
      // Pin the perpendicular to the upward half space: a person never stands
      // on their head, and without this the pelvis aims straight down and
      // folds the avatar in half as soon as the person faces away.
      if (up.y < 0) up.negate();
      const upDir = this.smoothDir('up', up, dt);
      this.aimBone('hips', 'spine', upDir, weight);

      const torso = this.smoothDir('torso',
        this.dir2(hipMid, shMid, k).normalize(), dt);
      if (this.b.spine && this.b.chest) {
        const blendDir = upDir.clone().lerp(torso, 0.6).normalize();
        this.aimBone('spine', 'chest', blendDir, weight);
      }
      if (this.b.chest && this.b.neck) {
        this.aimBone('chest', 'neck', torso, weight);
      }
    }

    // --- head ----------------------------------------------------------
    this.updateHead(dt, weight, k);

    // --- arms ----------------------------------------------------------
    for (const side of ['left', 'right']) {
      const [jS, jE, jW] = ARM[side];
      const shoulder = this.px(jS);
      const elbow = this.px(jE);
      const wrist = this.px(jW);
      const upper = side === 'left' ? 'leftUpperArm' : 'rightUpperArm';
      const lower = side === 'left' ? 'leftLowerArm' : 'rightLowerArm';
      const hand = side === 'left' ? 'leftHand' : 'rightHand';
      const dUpper = this.smoothDir(`${side}Upper`,
        this.limbDir(shoulder, elbow, this.rest.upperArm, k, 'upperArm'), dt);
      this.aimBone(upper, lower, dUpper, weight);
      const dLower = this.smoothDir(`${side}Lower`,
        this.limbDir(elbow, wrist, this.rest.lowerArm, k, 'lowerArm'), dt);
      this.aimBone(lower, hand, dLower, weight);
    }

    // --- legs ----------------------------------------------------------
    for (const side of ['left', 'right']) {
      const [jH, jK, jA] = LEG[side];
      const hip = this.px(jH);
      const knee = this.px(jK);
      const ankle = this.px(jA);
      const upper = side === 'left' ? 'leftUpperLeg' : 'rightUpperLeg';
      const lower = side === 'left' ? 'leftLowerLeg' : 'rightLowerLeg';
      const foot = side === 'left' ? 'leftFoot' : 'rightFoot';
      const toes = side === 'left' ? 'leftToes' : 'rightToes';
      const dUpper = this.smoothDir(`${side}UpperLeg`,
        this.limbDir(hip, knee, this.rest.upperLeg, k, 'upperLeg'), dt);
      this.aimBone(upper, lower, dUpper, weight);
      const dLower = this.smoothDir(`${side}LowerLeg`,
        this.limbDir(knee, ankle, this.rest.lowerLeg, k, 'lowerLeg'), dt);
      this.aimBone(lower, foot, dLower, weight);
      if (this.b[foot] && this.b[toes]) {
        const footDir = this.smoothDir(`${side}Foot`,
          new THREE.Vector3(0, -0.18, 1).normalize(), dt);
        this.aimBone(foot, toes, footDir, weight);
      }
    }

    // --- root translation ---------------------------------------------
    const nx = hipMid.x / (pose?.width || 1280);
    const ny = hipMid.y / (pose?.height || 720);
    this._target.set(
      clamp((nx - 0.5) * 1.8, -0.9, 0.9) * weight,
      clamp((0.6 - ny) * 1.2, -0.15, 0.15) * weight,
      0,
    );
    if (this.smoothing.active) this.smoothing.vector(this._target, 'root', dt);
    const floor = this._floorSample(1);
    if (floor.on) {
      this._target.y = floorTargetY(this._target.y, floor.rootY, floor.soleY,
        this.rest.sole, floor.ground).y;
    }
    this.root.position.lerp(this._target, damp(dt, TAU_ROOT));
    // Same order as update3D: plant -> smooth -> re-solve the floor.
    this._floorClamp(this._floorSample(1));
    this._plantFeet(dt, weight, 1);
    this._applyOutputFilter(dt);
    this._floorClamp(this._floorSample(1));
  }

  /** VRM expression names this avatar actually has, resolved once. */
  faceExpressions() {
    if (this._faceNames) return this._faceNames;
    const em = this.vrm && this.vrm.expressionManager;
    if (!em) return null;
    const map = em.expressionMap || {};
    const pick = (...names) => names.find((n) => n in map) || null;
    this._faceNames = {
      blink: pick('blink'),
      blinkLeft: pick('blinkLeft', 'blink_l'),
      blinkRight: pick('blinkRight', 'blink_r'),
      aa: pick('aa', 'a'),
      ih: pick('ih', 'i'),
      ou: pick('ou', 'u'),
      ee: pick('ee', 'e'),
      oh: pick('oh', 'o'),
      happy: pick('happy', 'joy'),
      angry: pick('angry'),
      sad: pick('sad', 'sorrow'),
      relaxed: pick('relaxed', 'fun'),
      surprised: pick('surprised'),
      neutral: pick('neutral'),
      // gaze presets, used only when the avatar has no look-at rig
      lookUp: pick('lookUp'),
      lookDown: pick('lookDown'),
      lookLeft: pick('lookLeft'),
      lookRight: pick('lookRight'),
    };
    return this._faceNames;
  }

  /**
   * Drive the face from the best available source.
   *
   * Priority: an Audio2Face playback track, then the in-page browser tracker
   * (`faceOverride`), then the server's face layer. The browser tracker wins
   * over the server because it carries gaze and per-eye blink; A2F wins over
   * both because it is a deliberate performance that owns the audio clock.
   *
   * A source still drives when the body is lost, which is what makes a
   * close-up animate the avatar (`face only` in the status pill).
   */
  updateFace(pose, dt) {
    const external = this.externalFace;
    const now = typeof performance !== 'undefined' ? performance.now() : 0;
    const override = this.faceOverride &&
      (!this.faceOverride.at || now - this.faceOverride.at < 700)
      ? this.faceOverride : null;
    const face = external
      ? { blend: external.blend || {}, eye: external.eye }
      : (override || (pose && pose.face ? pose.face : null));
    const hasQ = !!(face && Array.isArray(face.q) && face.q.length === 4);
    const fresh = !!(face && face.blend);
    this.face.fresh = fresh;

    // --- head ------------------------------------------------------------
    if (hasQ) {
      this.face.target.set(face.q[0], face.q[1], face.q[2], face.q[3]);
      if (this.face.target.lengthSq() < 1e-6) this.face.target.identity();
      else this.face.target.normalize();
    }
    this.face.headW += ((hasQ ? 1 : 0) - this.face.headW) *
      damp(dt, TAU_FACE_HEAD);

    const head = this.b.head;
    if (head && this.restHeadLocal && this.face.headW > 0.001) {
      this.face.q.slerp(this.face.target, damp(dt, TAU_FACE_HEAD));
      this.vrm.scene.updateMatrixWorld(true);
      head.parent.getWorldQuaternion(this._q3);
      this._q4.copy(this._q3).invert()
        .multiply(this.face.q).multiply(this._q3);
      this._headTarget.copy(this.restHeadLocal).premultiply(this._q4);
      head.quaternion.slerp(this._headTarget, this.face.headW);
    }

    // --- expressions -----------------------------------------------------
    const em = this.vrm && this.vrm.expressionManager;
    const names = this.faceExpressions();

    if (em && names) {
      let targets = FACE_ZERO;
      if (fresh) {
        // MuseTalk clips peak around jawOpen 0.25 while a shouted webcam pose
        // reaches 0.6; normalising against the speaker's own range keeps the
        // mouth visibly moving instead of pinned shut or permanently wide
        const get = typeof face.blend.get === 'function'
          ? (name) => face.blend.get(name) || 0
          : (name) => face.blend[name] || 0;
        const jawRaw = get('jawOpen');
        this.face.mouthRef += (jawRaw - this.face.mouthRef) *
          (jawRaw > this.face.mouthRef ? 0.35 : 0.008);
        this.face.mouthRef = clamp(this.face.mouthRef, 0.18, 0.7);
        targets = mapBlendshapes(face.blend, {
          mouthRef: this.face.mouthRef,
        });
      }

      // Per-eye blink is strictly better than the averaged preset: it keeps a
      // wink and an asymmetric blink. Writing both `blink` and `blinkLeft/Right`
      // would double the eyelid close, so one route is chosen per avatar.
      if (names.blinkLeft && names.blinkRight) targets = { ...targets, blink: 0 };
      else if (names.blink) {
        targets = { ...targets, blinkLeft: 0, blinkRight: 0 };
      }

      const a = damp(dt, TAU_FACE);
      for (const key of Object.keys(FACE_ZERO)) {
        const name = names[key];
        if (!name) continue;
        const want = targets[key] || 0;
        const cur = this.face.weights[key] || 0;
        const next = cur + (want - cur) * a;
        this.face.weights[key] = next;
        if (next > 0.0005 || cur > 0.0005) em.setValue(name, next);
      }
    }

    // --- gaze ------------------------------------------------------------
    if (external) this.applyExternalEyes(external.eye);
    else this.applyGaze(face && face.gaze ? face.gaze : null, dt);
  }

  /** Creates the look-at target once; the previous target is restored on release. */
  ensureLookAtTarget(lookAt) {
    if (this._lookAtTarget) return;
    this._lookAtTarget = new THREE.Object3D();
    this._lookAtTarget.name = 'mocap-gaze-target';
    this.vrm.scene.add(this._lookAtTarget);
    this._lookAtPrev = lookAt.target || null;
    lookAt.target = this._lookAtTarget;
  }

  /** Removes the gaze target and zeroes the fallback eye expressions. */
  releaseLookAt(lookAt) {
    if (lookAt && this._lookAtTarget) {
      lookAt.target = this._lookAtPrev || null;
      this.vrm.scene.remove(this._lookAtTarget);
      this._lookAtTarget = null;
    }
    const em = this.vrm && this.vrm.expressionManager;
    if (em && this._faceNames) {
      for (const key of ['lookUp', 'lookDown', 'lookLeft', 'lookRight']) {
        if (this._faceNames[key]) em.setValue(this._faceNames[key], 0);
      }
    }
  }

  /**
   * Eye gaze from the browser tracker, in image-normalised units
   * (x positive = towards the image's right, y positive = down).
   *
   * With a look-at rig the target sits a capped angle off the line from the
   * head to the camera, so the avatar looks out of the screen where the
   * person's irises are looking. Without one, the VRM look presets take over.
   */
  applyGaze(gaze, dt) {
    const head = this.b.head;
    if (!head) return;
    const has = !!(gaze && Number.isFinite(gaze.x) && Number.isFinite(gaze.y));
    const a = damp(dt, TAU_FACE_GAZE);
    this.gazeSmoothed.x += ((has ? clamp(gaze.x, -1.5, 1.5) : 0) -
      this.gazeSmoothed.x) * a;
    this.gazeSmoothed.y += ((has ? clamp(gaze.y, -1.5, 1.5) : 0) -
      this.gazeSmoothed.y) * a;
    this.gaze.w += ((has ? 1 : 0) - this.gaze.w) * damp(dt, TAU_FACE_HEAD);

    const lookAt = this.vrm && this.vrm.lookAt;
    if (lookAt) {
      if (this.gaze.w > 0.001) {
        this.ensureLookAtTarget(lookAt);
        head.getWorldPosition(this._eyePos);
        const dir = this._eyeA.copy(this.camera
          ? this.camera.position : this._eyeB.set(0, 1.4, 3)
        ).sub(this._eyePos).normalize();
        // camera right and up, independent of the avatar's own orientation
        const right = this._eyeB.crossVectors(UP, dir).normalize();
        const up = this._eyeC.crossVectors(dir, right).normalize();
        const yaw = Math.sin(this.gazeSmoothed.x * GAZE_RANGE.yaw) * 1.2;
        const pitch = Math.sin(this.gazeSmoothed.y * GAZE_RANGE.pitch) * 1.2;
        this._lookAtTarget.position.copy(this._eyePos)
          .addScaledVector(dir, 1.2)
          .addScaledVector(right, yaw)
          .addScaledVector(up, -pitch);
      } else if (!has) {
        this.releaseLookAt(lookAt);
      }
      return;
    }
    this.applyGazeExpressions(this.gazeSmoothed.x, this.gazeSmoothed.y);
  }

  /** Writes gaze into the VRM look presets, for avatars with no look-at rig. */
  applyGazeExpressions(x, y) {
    const em = this.vrm && this.vrm.expressionManager;
    const names = this.faceExpressions();
    if (!em || !names) return;
    const side = clamp(x, -1, 1);
    const up = clamp(-y, -1, 1);
    if (names.lookLeft || names.lookRight || names.lookUp || names.lookDown) {
      if (names.lookLeft) em.setValue(names.lookLeft, Math.max(0, side));
      if (names.lookRight) em.setValue(names.lookRight, Math.max(0, -side));
      if (names.lookUp) em.setValue(names.lookUp, Math.max(0, up));
      if (names.lookDown) em.setValue(names.lookDown, Math.max(0, -up));
      return;
    }
    // VRM 0.x per-eye names; "out" on the left eye is the character's left,
    // which is the screen's right, hence the same sign as `side`.
    em.setValue('eyeLookUpLeft', Math.max(0, up));
    em.setValue('eyeLookUpRight', Math.max(0, up));
    em.setValue('eyeLookDownLeft', Math.max(0, -up));
    em.setValue('eyeLookDownRight', Math.max(0, -up));
    em.setValue('eyeLookOutLeft', Math.max(0, side));
    em.setValue('eyeLookInRight', Math.max(0, side));
    em.setValue('eyeLookInLeft', Math.max(0, -side));
    em.setValue('eyeLookOutRight', Math.max(0, -side));
  }

  /**
   * Gaze for the Audio2Face track. The SDK eye output is (x = pitch,
   * y = yaw) per eye in degrees, so the lookAt target sits a small angle off
   * the straight line to the camera. Falls back to the VRM look presets when
   * this avatar has no lookAt rig.
   */
  applyExternalEyes(eye) {
    const head = this.b.head;
    if (!head || !Array.isArray(eye) || eye.length < 4) return;
    const yaw = (eye[1] + eye[3]) * 0.5 * Math.PI / 180;
    const pitch = (eye[0] + eye[2]) * 0.5 * Math.PI / 180;

    const lookAt = this.vrm && this.vrm.lookAt;
    if (lookAt) {
      this.ensureLookAtTarget(lookAt);
      head.getWorldPosition(this._eyePos);
      const dir = this._eyeA.copy(this.camera
        ? this.camera.position : this._eyeB.set(0, 1.4, 3)
      ).sub(this._eyePos).normalize();
      const right = this._eyeB.crossVectors(UP, dir).normalize();
      const up = this._eyeC.crossVectors(dir, right).normalize();
      dir.addScaledVector(right, Math.sin(yaw))
        .addScaledVector(up, Math.sin(pitch)).normalize();
      this._lookAtTarget.position.copy(this._eyePos)
        .addScaledVector(dir, 1.2);
      return;
    }
    // no lookAt rig: approximate with the eye look expressions
    const range = 0.3; // radians that maps to a full expression weight
    this.applyGazeExpressions(yaw / range, -pitch / range);
  }

  /** Called by the browser tracker on every detected frame. */
  setFaceOverride(frame) {
    this.faceOverride = frame;
  }

  /** Stop the browser face source and hand the eyes back to the avatar. */
  clearFaceOverride() {
    this.faceOverride = null;
    this.gaze.w = 0;
    this.gazeSmoothed.x = 0;
    this.gazeSmoothed.y = 0;
    this.releaseLookAt(this.vrm && this.vrm.lookAt);
  }

  /** Stop the Audio2Face face source and restore the avatar's own gaze. */
  clearExternalFace() {
    this.externalFace = null;
    this.releaseLookAt(this.vrm && this.vrm.lookAt);
  }

  updateHead(dt, weight, k) {
    const head = this.b.head;
    if (!head) return;
    // when the facial tracker has a lock it owns the head pose outright
    if (this.face.headW > 0.001) return;
    let roll = this.head.roll;
    let yaw = this.head.yaw;

    const earsOk = this.scores[KP.leftEar] > JOINT_CONF &&
      this.scores[KP.rightEar] > JOINT_CONF;
    const eyesOk = this.scores[KP.leftEye] > JOINT_CONF &&
      this.scores[KP.rightEye] > JOINT_CONF;

    if (earsOk) {
      // mirrored view: person's right ear is the avatar's left ear. Turning
      // away swaps which ear lands on which side of the image, so the whole
      // head-frame angle flips with `facing`.
      const left = this.px(KP.rightEar);
      const right = this.px(KP.leftEar);
      const dx = (left.x - right.x) * k * this.facing;
      const dy = -(left.y - right.y) * k * this.facing;
      const width = Math.hypot(dx, dy);
      if (width > 0.01) {
        roll = clamp(Math.atan2(dy, dx), -0.7, 0.7);
        if (this.scores[KP.nose] > 0.4) {
          const nose = this.px(KP.nose);
          const midX = (left.x + right.x) / 2;
          yaw = clamp(Math.atan2((nose.x - midX) * k, width) * 1.4,
            -0.8, 0.8);
        }
      }
    } else if (eyesOk) {
      const left = this.px(KP.rightEye);
      const right = this.px(KP.leftEye);
      roll = clamp(Math.atan2(-(left.y - right.y) * this.facing,
        Math.abs(left.x - right.x)), -0.5, 0.5);
    }

    const a = damp(dt, TAU_HEAD);
    this.head.roll += (roll - this.head.roll) * a;
    this.head.yaw += (yaw - this.head.yaw) * a;

    if (weight <= 0.001 || !this.restHeadLocal) return;

    // Face angles are applied in world space on top of the rest head pose,
    // which is carried along by the (mocap driven) neck.
    this._q1.setFromAxisAngle(AXIS_Y, this.head.yaw);
    this._q2.setFromAxisAngle(AXIS_Z, this.head.roll);
    this._q1.multiply(this._q2);

    head.parent.getWorldQuaternion(this._q3);
    this._q4.copy(this._q3).invert().multiply(this._q1).multiply(this._q3);
    this._headTarget.copy(this.restHeadLocal).premultiply(this._q4);
    head.quaternion.slerp(this._headTarget, weight);
  }

  /* ------------------------------------------------------- 3D joints ---- */

  /**
   * How much of each limb the camera can actually see.
   *
   * A reconstruction that cannot see a limb still returns one. In a waist-up or
   * shoulder-height shot the missing half is invented as a crouch - the pelvis
   * drops, the thighs swing forward, the ankles float in front of the body -
   * which is a crouching avatar with its feet in the air. The 2D joints are
   * the receipt: InstantHMR projects the same skeleton into the submitted
   * frame, so a joint whose projection lands outside the image was never
   * observed. A leg counts as seen only when knee, ankle and toes are all
   * inside; anything less is a crop, not a leg. The detector's box says the
   * same thing from the other side, so a box that runs into the bottom edge of
   * the frame also withholds the margin there.
   *
   * The arms get the same treatment, segment by segment: an arm below a
   * waist-up shot is invented too, and unlike the legs it is invented *moving*
   * - the elbow and wrist wander and the roll spins, which reads as the model
   * shaking. Shoulder+elbow is the upper arm, elbow+wrist the forearm.
   *
   * The weights are smoothed, so a limb walking back into shot blends in rather
   * than snapping in one frame.
   */
  _cropWeights(pose, dt) {
    const screen = pose && pose.skeleton2d;
    let seenLeft = true;
    let seenRight = true;
    const armSeen = {
      left: { upper: true, lower: true },
      right: { upper: true, lower: true },
    };
    if (Array.isArray(screen) && screen.length >= SMPL_BODY_JOINTS) {
      // When the detector's box runs into the bottom edge the person really
      // does continue past the frame, so the tolerance the margin grants at
      // the other edges must not be granted there - a joint just outside the
      // picture is the cut, not a rounding error.
      const box = pose.box2d;
      const cut = Array.isArray(box) && box.length >= 4 &&
        box[3] >= 1 - CROP_EDGE;
      const floor = cut ? 1 : 1 + CROP_MARGIN;
      const inside = (i) => {
        const q = screen[i];
        return Array.isArray(q) && q.length >= 2 &&
          Number.isFinite(q[0]) && Number.isFinite(q[1]) &&
          q[0] >= -CROP_MARGIN && q[0] <= 1 + CROP_MARGIN &&
          q[1] >= -CROP_MARGIN && q[1] <= floor;
      };
      const leg = (knee, ankle, toe) =>
        inside(knee) && inside(ankle) && inside(toe);
      seenLeft = leg(SMPL.leftKnee, SMPL.leftAnkle, SMPL.leftToe);
      seenRight = leg(SMPL.rightKnee, SMPL.rightAnkle, SMPL.rightToe);
      for (const side of ['left', 'right']) {
        const shoulder = inside(SMPL[`${side}Shoulder`]);
        const elbow = inside(SMPL[`${side}Elbow`]);
        const wrist = inside(SMPL[`${side}Wrist`]);
        armSeen[side].upper = shoulder && elbow;
        armSeen[side].lower = elbow && wrist;
      }
    }
    const k = damp(dt, TAU_CROP);
    this._legSeen.left += ((seenLeft ? 1 : 0) - this._legSeen.left) * k;
    this._legSeen.right += ((seenRight ? 1 : 0) - this._legSeen.right) * k;
    for (const side of ['left', 'right']) {
      const seen = this._armSeen[side];
      seen.upper += ((armSeen[side].upper ? 1 : 0) - seen.upper) * k;
      seen.lower += ((armSeen[side].lower ? 1 : 0) - seen.lower) * k;
    }
    // Close-up only when no leg at all is in shot: one visible leg still pins
    // the camera down enough to trust the reconstruction's height and pitch.
    const target = 1 - Math.max(this._legSeen.left, this._legSeen.right);
    this._crop += (target - this._crop) * k;
  }

  /**
   * Undo the pitch a close-up camera invents.
   *
   * Without a floor in view, InstantHMR guesses a camera looking down at the
   * person, and the guess shows up as a constant lean away from the viewer: in
   * a waist-up shot the shoulders sit some 40 cm behind the pelvis in depth
   * while their 2D projection is dead vertical. The image can only ever show
   * left-right lean, so the part of the torso axis that points into the screen
   * is exactly the part that was invented. Dropping that component and
   * rotating the skeleton about the pelvis by the difference stands the person
   * up, and leaves any lean the image does show alone.
   *
   * This only runs while the frame is a close-up, so the real forward lean of
   * a full-body dance is untouched.
   */
  _uprightCrop(dt) {
    this._cropQw.identity();
    if (this._crop < 0.002) {
      // drop the filter state, so re-entering a close-up starts from the
      // measurement instead of a stale direction from the previous shot
      this.dirState.delete('cropUp');
      return;
    }
    const up = this._cropUp.copy(this._j3[SMPL.leftShoulder])
      .add(this._j3[SMPL.rightShoulder]).multiplyScalar(0.5);
    if (up.lengthSq() < 1e-8) return;
    up.normalize();
    // one-euro, because the shoulder line is the noisiest pair in the skeleton
    const smoothed = this.smoothDir('cropUp', up, dt);
    this._cropTarget.set(smoothed.x, smoothed.y, 0);
    if (this._cropTarget.lengthSq() < 1e-6) this._cropTarget.copy(smoothed);
    this._cropTarget.normalize();
    this._cropQ.setFromUnitVectors(smoothed, this._cropTarget);
    this._cropQw.identity().slerp(this._cropQ, this._crop);
    const q = this._cropQw;
    for (let i = 0; i < SMPL_BODY_JOINTS; i++) this._j3[i].applyQuaternion(q);
    for (const key of Object.keys(this._lm3)) {
      const v = this._lm3[key];
      if (v.valid) v.applyQuaternion(q);
    }
  }

  /**
   * A leg bone's direction, pinned to straight down when the shot hides it.
   *
   * `weight` is how much of the leg the camera can see (1 = all of it). At 0
   * the measured direction - which is an invention - is replaced by the only
   * stance a hidden leg can be in: hanging under the hip, parallel to the
   * torso's own up axis.
   */
  legDir(a, b, weight, out) {
    out.copy(this._j3[b]).sub(this._j3[a]);
    if (out.lengthSq() < 1e-10) return out.copy(this._upBody).negate();
    out.normalize();
    if (weight >= 0.999) return out;
    this._t7.copy(this._upBody).negate();
    out.lerp(this._t7, 1 - weight);
    if (out.lengthSq() < 1e-8) out.copy(this._t7);
    return out.normalize();
  }

  /**
   * An arm bone's direction, blended toward a resting pose the shot can see.
   *
   * `weight` is how much of that arm segment the camera can see (1 = all of
   * it). At 0 the measured direction - which is an invention - is replaced by
   * `rest`: the upper arm hangs under its shoulder, the forearm continues the
   * upper arm, and the hand continues the forearm. That chain is the only arm
   * stance a missing arm can be in, and it cannot shake.
   */
  armDir(a, b, weight, rest, out) {
    out.copy(this._j3[b]).sub(this._j3[a]);
    if (out.lengthSq() < 1e-10) return out.copy(rest);
    out.normalize();
    if (weight >= 0.999) return out;
    out.lerp(rest, 1 - weight);
    if (out.lengthSq() < 1e-8) out.copy(rest);
    return out.normalize();
  }

  /**
   * Keep the forearm inside the range a human elbow can reach.
   *
   * Only the ANGLE between the upper arm and the forearm is limited, never the
   * direction the wrist was measured in. `upper` is the upper-arm direction and
   * `back` the anatomical back of that arm (the azimuth the bend should have);
   * the forearm may straighten fully or flex up to ELBOW_MAX.
   *
   * Why the angle and not the plane: with the upper arm aimed at its measured
   * direction, the azimuth of the bend IS the shoulder's internal/external
   * rotation. That rotation is clamped to TWIST_LIMIT (a shoulder cannot spin
   * inside its socket, and the measured roll used to wind past 300 degrees), so
   * on the frames the clamp bites, the forearm legitimately leaves the plane
   * the avatar's own humerus roll describes. Pinning it back into that plane
   * does not fix a joint - it rewrites the wrist direction, and on the live feed
   * it collapsed a measured 130 degree bend to 10 degrees: the avatar stood
   * with a straight arm while the person's elbow was fully folded.
   *
   * `side` names the arm the result is remembered for; see _slewFore, which
   * keeps the applied direction continuous from frame to frame.
   */
  limitElbow(side, upper, fore, back, out) {
    const u = this._elbowU.copy(upper);
    const along = clamp(fore.dot(u), -1, 1);
    if (Math.acos(along) <= ELBOW_MAX) return this._slewFore(side, fore, out);

    // Shorten the bend to the joint's stop along the direction the wrist is
    // already in. That is the smallest change that makes the pose legal, and it
    // is what a real elbow does when it runs out of flexion.
    const perp = this._elbowB.copy(fore).addScaledVector(u, -along);
    if (perp.lengthSq() < 1e-8) {
      perp.copy(back).addScaledVector(u, -back.dot(u));
      if (perp.lengthSq() < 1e-8) return this._slewFore(side, fore, out);
    }
    perp.normalize();
    out.copy(u).multiplyScalar(Math.cos(ELBOW_MAX))
      .addScaledVector(perp, Math.sin(ELBOW_MAX));
    return this._slewFore(side, out.normalize(), out);
  }

  /**
   * Limit how fast the applied forearm direction may turn, and remember it.
   *
   * The wrist direction comes out of a reconstruction that is noisiest exactly
   * where the arm crosses the chest, and copying it through unchanged let the
   * applied forearm swing up to 149 degrees in one frame (measured) while the
   * person's own forearm moved 16. Nothing is invented here: when the requested
   * direction is further than a real elbow can travel in one frame, the result
   * is the nearest direction along that arc, so the forearm sweeps there over a
   * few frames instead of teleporting.
   */
  _slewFore(side, candidate, out) {
    const prev = this._foreDir[side];
    if (this._foreDirOk[side]) {
      const limit = FOREARM_MAX_SLEW * this._dt;
      const angle = Math.acos(clamp(prev.dot(candidate), -1, 1));
      if (angle > limit) {
        // Rotate last frame's direction by exactly the allowed arc. Blending
        // the two vectors and re-normalising (nlerp) does NOT hold the arc:
        // near the antipode the chord passes close to the origin, so "10% of
        // the way across 141 degrees" lands 143 degrees from where it started.
        // Measured on the live feed, that made the cap look ineffective.
        this._slewAxis.crossVectors(prev, candidate);
        if (this._slewAxis.lengthSq() < 1e-10) {
          // Exactly opposite: any great circle is equally good, so pick a
          // stable perpendicular rather than dividing by zero.
          this._slewAxis.crossVectors(
            Math.abs(prev.y) < 0.9 ? AXIS_Y : AXIS_Z, prev);
        }
        this._slewAxis.normalize();
        this._slewQ.setFromAxisAngle(this._slewAxis, limit);
        out.copy(prev).applyQuaternion(this._slewQ);
        prev.copy(out);
        return out;
      }
    }
    out.copy(candidate);
    prev.copy(out);
    this._foreDirOk[side] = true;
    return out;
  }

  /**
   * Put the person's camera-space skeleton into the avatar's world frame.
   *
   * InstantHMR reports joints in camera axes (x right, y down, z forward), and
   * the app camera looks down -Z. A person facing the capture camera therefore
   * has their up along -Y, their right along -X and their forward along -Z.
   * Sending every point through (x, y, z) -> (-x, -y, -z) stands them up in the
   * avatar's frame: +Y is up and +Z is the avatar's facing, so someone looking
   * into the capture camera looks into the app camera too.
   *
   * The map is FIXED - it never follows the body. That is what carries the
   * person's global rotation: deriving a basis from the shoulder line every
   * frame (the old behaviour) re-aimed the avatar at the camera on every frame
   * and silently deleted every turn, so the avatar could never face the same
   * way as the person for more than one frame.
   *
   * The map is a reflection (the avatar faces the viewer, so the person's
   * right is the avatar's left) and cross products in a reflected frame come
   * out negated. That cancels against the mirrored skeleton: chest forward is
   * `cross(side, up)` of the projected shoulder line, pelvis forward is
   * `cross(hipLine, up)`, and both land on +Z for a person facing the camera.
   *
   * Everything is projected relative to the pelvis, so the avatar does not
   * inherit the metres of depth between the person and the capture camera.
   */
  basis3D(pose, dt) {
    const joints = pose.joints3d;
    const raw = this._raw3;
    for (let i = 0; i < SMPL_BODY_JOINTS; i++) {
      const p = joints[i];
      if (!Array.isArray(p) || p.length < 3) return false;
      if (!Number.isFinite(p[0]) || !Number.isFinite(p[1]) ||
          !Number.isFinite(p[2])) return false;
      raw[i].set(p[0], p[1], p[2]);
    }

    // Filter the reconstructed joints here, in camera space and before a
    // single bone is solved: the shake is in the reconstruction, so low-passing
    // it first means no bone ever sees it, instead of each limb trying to
    // second-guess it afterwards.
    this.smoothing.joints(raw, SMPL_BODY_JOINTS, dt);

    const pelvis = this._midHip.copy(raw[SMPL.leftHip])
      .add(raw[SMPL.rightHip]).multiplyScalar(0.5);
    this._midSh.copy(raw[SMPL.leftShoulder])
      .add(raw[SMPL.rightShoulder]).multiplyScalar(0.5);

    const torso = this._midSh.distanceTo(pelvis);
    if (torso < 1e-3) return false;

    // camera axes -> avatar axes: a straight axis flip, no per-frame basis
    for (let i = 0; i < SMPL_BODY_JOINTS; i++) {
      const p = raw[i];
      this._j3[i].set(pelvis.x - p.x, pelvis.y - p.y, pelvis.z - p.z);
    }

    // Optional auxiliary landmarks (ears/nose, elbows, heels, knuckles) live
    // in the same camera axes as the joints; project them the same way. They
    // carry the roll and yaw a 22-joint skeleton cannot express.
    for (const key of Object.keys(this._lm3)) this._lm3[key].valid = false;
    const extra = pose.landmarks;
    if (extra) {
      for (const key of Object.keys(extra)) {
        const p = extra[key];
        if (!Array.isArray(p) || p.length < 3 ||
            !Number.isFinite(p[0]) || !Number.isFinite(p[1]) ||
            !Number.isFinite(p[2])) continue;
        let v = this._lm3[key];
        if (!v) {
          v = new THREE.Vector3();
          this._lm3[key] = v;
        }
        // same treatment as the joints: these carry the roll and yaw, so a
        // noisy landmark pair spins a bone that the joints alone would not
        this._lmRaw.set(p[0], p[1], p[2]);
        this.smoothing.vector(this._lmRaw, `lm:${key}`, dt);
        v.set(pelvis.x - this._lmRaw.x, pelvis.y - this._lmRaw.y,
          pelvis.z - this._lmRaw.z);
        v.valid = true;
      }
    }

    // A waist-up shot has no legs to measure, so settle that before any of the
    // geometry below is read: the frame gate first, then the pitch correction.
    this._cropWeights(pose, dt);
    this._uprightCrop(dt);

    // --- centre of gravity ------------------------------------------------
    // How high the hips ride over the foot that carries the weight. The lower
    // ankle is the support foot, so a lifted or kicking leg never raises the
    // body; and because both points are reconstructed in the same frame, this
    // is independent of where the camera is and how far away the person stands.
    {
      const j = this._j3;
      const support = j[SMPL.leftAnkle].y < j[SMPL.rightAnkle].y
        ? 'left' : 'right';
      const hipHeight = -j[SMPL[`${support}Ankle`]].y;
      const legLen = j[SMPL[`${support}Hip`]]
        .distanceTo(j[SMPL[`${support}Knee`]]) +
        j[SMPL[`${support}Knee`]].distanceTo(j[SMPL[`${support}Ankle`]]);
      if (hipHeight > 0.3 && hipHeight < 1.5 && legLen > 0.3) {
        // hipRef is the person's own standing height: the tallest the hips get
        // while the leg is still nearly straight, so a crouch can never become
        // the reference and a tucked-leg jump cannot raise it either.
        //
        // A close-up's legs are a guess, so they may not teach the reference -
        // otherwise the invented crouch would become this person's standing
        // height and the avatar would shrink for the rest of the shot.
        if (this._crop < 0.5) {
          if (!this.hipRef) {
            this.hipRef = hipHeight;
          } else if (hipHeight > this.hipRef && hipHeight / legLen > 0.9) {
            this.hipRef += (hipHeight - this.hipRef) * damp(dt, TAU_HIP);
          }
        }
        this._hipHeight = hipHeight;
        this._personLeg = legLen;
      }
    }

    // Current torso axes, read off the joints and expressed in avatar space.
    // The person's lean and turn live in these two vectors; the rest of the
    // body is measured relative to them below.
    const upBody = this.smoothDir('torsoUp',
      this._up3.copy(this._j3[SMPL.leftShoulder])
        .add(this._j3[SMPL.rightShoulder]).multiplyScalar(0.5).normalize(), dt);
    const sideBody = this.smoothDir('torsoSide',
      this._side3.copy(raw[SMPL.leftShoulder])
        .sub(raw[SMPL.rightShoulder])
        .applyQuaternion(this._cropQw).normalize(), dt);

    // side x up = the avatar's facing, since a reflected (x=left, y=up)
    // frame puts forward at +Z
    const fwd = this._fwd3.crossVectors(sideBody, upBody);
    if (fwd.lengthSq() < 1e-8) return false;
    fwd.normalize();
    const right = this._right3.crossVectors(upBody, fwd).normalize();

    this._upBody.copy(upBody);
    this._fwdBody.copy(fwd);
    this._rightBody.copy(right);

    // Root travel uses the same fixed mapping, so walking towards the camera
    // moves the avatar towards the viewer regardless of which way it faces.
    if (!this.rootRef) {
      this.rootRef = pelvis.clone();
      this._rootDelta.set(0, 0, 0);
    } else {
      this._rootDelta.set(
        this.rootRef.x - pelvis.x,
        this.rootRef.y - pelvis.y,
        this.rootRef.z - pelvis.z,
      );
    }
    this._personTorso = torso;
    return true;
  }

  /**
   * Drive the rig from SMPL-H 3D joints (FlowHMR) instead of 2D keypoints.
   *
   * The 2D path has to invent depth from limb foreshortening; here every bone
   * direction is measured, so each bone is aimed at its child exactly. Only the
   * root translation is still scaled against the avatar, by the torso ratio, so
   * a short person and a tall avatar cover the same ground.
   */
  update3D(pose, dt) {
    // Baked 3D frames carry no face, so anything the previous clip left on the
    // face eases back to rest. Live 3D frames do carry the server face layer,
    // so those keep driving the expressions and head pose.
    this.updateFace(pose && pose.face ? pose.face : null, dt);

    this.blend += (1 - this.blend) * damp(dt, TAU_BLEND);
    const weight = this.blend;
    if (weight <= 0.002) return;

    this.vrm.scene.updateMatrixWorld(true);
    if (!this.basis3D(pose, dt)) return;

    const j = this._j3;
    const w = weight;
    const dirJ = (a, b) => this._d3.copy(j[b]).sub(j[a]);

    // --- anatomical roll references --------------------------------------
    // The pelvis forward comes from the hip line and the chest forward from
    // the shoulder line: the angle between them is the torso twist, which a
    // direction-only aim silently throws away.
    const up = this._upBody;
    this._t1.copy(j[SMPL.rightHip]).sub(j[SMPL.leftHip]);
    if (this._t1.lengthSq() > 1e-8) {
      // hip line x up: the hip line is the mirrored person's right (the
      // avatar's left), so this is the pelvis facing in the avatar frame
      this._t1.crossVectors(this._t1.normalize(), up);
    }
    if (this._t1.lengthSq() > 1e-8) {
      this._hipFwd.copy(this._t1).normalize();
      if (this._hipFwd.dot(this._fwdBody) < 0) this._hipFwd.negate();
    } else {
      this._hipFwd.copy(this._fwdBody);
    }
    // smooth the twist so pelvis keypoint noise does not wobble the whole spine
    this._hipFwd.copy(this.smoothDir('hipFwd', this._hipFwd, dt));
    this._hipFwd.addScaledVector(up, -this._hipFwd.dot(up));
    if (this._hipFwd.lengthSq() < 1e-8) this._hipFwd.copy(this._fwdBody);
    this._hipFwd.normalize();

    const shoulderFwd = this._fwdBody;
    const mixFwd = (t, out) => {
      out.copy(this._hipFwd).lerp(shoulderFwd, t);
      if (out.lengthSq() < 1e-8) out.copy(this._rightBody);
      return out.normalize();
    };

    // Head yaw: the nose sits in front of the ear line, so nose - earMid is
    // the facing direction. Without face landmarks the chest forward stands in.
    this._headFwd.copy(shoulderFwd);
    const nose = this.lm('nose');
    const lEar = this.lm('left_ear');
    const rEar = this.lm('right_ear');
    if (nose && lEar && rEar) {
      this._t1.copy(lEar).add(rEar).multiplyScalar(0.5);
      this._t2.copy(nose).sub(this._t1);
      this._t2.addScaledVector(up, -this._t2.dot(up));
      if (this._t2.lengthSq() > 1e-8) {
        this._t2.normalize();
        if (this._t2.dot(shoulderFwd) > -0.9) {
          this._headFwd.copy(this.smoothDir('headFwd', this._t2, dt));
        }
      }
    }

    // --- spine: aim each segment, distributing the hip->chest twist ------
    this.poseBone('hips', dirJ(SMPL.pelvis, SMPL.spine), this._hipFwd, w);
    this.poseBone('spine', dirJ(SMPL.spine, SMPL.spine1),
      mixFwd(0.35, this._t3), w);
    if (this.b.upperChest && this.b.chest !== this.b.upperChest) {
      this.poseBone('chest', dirJ(SMPL.spine1, SMPL.spine2),
        mixFwd(0.7, this._t3), w);
      this.poseBone('upperChest', dirJ(SMPL.spine2, SMPL.neck),
        shoulderFwd, w);
    } else {
      this.poseBone('chest', dirJ(SMPL.spine2, SMPL.neck),
        mixFwd(0.8, this._t3), w);
    }
    this.poseBone('neck', dirJ(SMPL.neck, SMPL.head), shoulderFwd, w);
    this.poseBone('head', dirJ(SMPL.neck, SMPL.head), this._headFwd, w);

    // --- limbs: the avatar's left reads the person's right ---------------
    for (const side of ['left', 'right']) {
      const p = side === 'left' ? 'right' : 'left';

      // clavicle: collar -> shoulder carries shrug and protraction. Its roll
      // is pinned to the rest roll (TWIST_LIMIT 0) - a clavicle does not spin
      // about its own axis, and the arm hanging off its tip is aimed
      // absolutely anyway, so a measured roll there only shears the shoulder.
      this.poseBone(`${side}Shoulder`,
        dirJ(SMPL[`${p}Collar`], SMPL[`${p}Shoulder`]), shoulderFwd, w);

      // Where this arm hangs when the shot cuts it off. Read off the avatar's
      // own bones, so it works whichever way the VRM's axes point: the upper
      // arm root sits outboard of the chest, and the socket direction is the
      // sideways axis this arm belongs to.
      const armSeen = this._armSeen[p];
      this._armOut.copy(up).negate();
      const upperBone = this.b[`${side}UpperArm`];
      const chestBone = this.b.chest || this.b.hips;
      if (upperBone && chestBone) {
        upperBone.getWorldPosition(this._armPos);
        chestBone.getWorldPosition(this._armChest);
        this._armOut.copy(this._armPos).sub(this._armChest);
        this._armOut.addScaledVector(up, -this._armOut.dot(up));
        if (this._armOut.lengthSq() > 1e-8) this._armOut.normalize();
        else this._armOut.copy(up).negate();
      }
      this._armHang.copy(up).negate().addScaledVector(this._armOut, ARM_SLACK);
      if (this._armHang.lengthSq() < 1e-8) this._armHang.copy(up).negate();
      this._armHang.normalize();

      // Segment by segment: a hidden upper arm hangs, a hidden forearm
      // continues it, and a visible one keeps its reconstructed direction.
      this.armDir(SMPL[`${p}Shoulder`], SMPL[`${p}Elbow`], armSeen.upper,
        this._armHang, this._armUpper);
      this.armDir(SMPL[`${p}Elbow`], SMPL[`${p}Wrist`], armSeen.lower,
        this._armUpper, this._armFore);

      // Back of the UPPER arm - the roll a direction-only transfer leaves to
      // the rest pose - read from the elbow's own bend plane.
      //
      // With the upper arm aimed at its measured direction, the one degree of
      // freedom left is the roll, and the roll IS the plane the forearm bends
      // in: the elbow is a hinge, so how the forearm leaves the upper arm's
      // axis measures the shoulder's internal/external rotation exactly. Both
      // ends are joint positions the whole chain is already driven by.
      //
      // The elbow-crease landmarks used to be the source. They are two points
      // ~6 cm apart whose relative depth the reconstruction is least sure of,
      // so a bad sample asked a humerus for more than a full turn on this feed
      // (-388 degrees, measured) and the bone wound after it. They now have no
      // say at all. An elbow bent less than ~5 degrees has no plane to read -
      // its bend vector vanishes - so the reading fades out with the bend and
      // the carried roll takes over, which is also why a straight arm never
      // snaps its roll.
      const armBack = this._armBack.copy(this._armFore)
        .addScaledVector(this._armUpper, -this._armFore.dot(this._armUpper));
      const bendLen = armBack.length();
      if (bendLen > 1e-6) armBack.multiplyScalar(-1 / bendLen);
      else armBack.copy(shoulderFwd).negate();
      const elbowTrust = clamp((bendLen - 0.04) / 0.12, 0, 1) *
        armSeen.upper * armSeen.lower;

      const backPrev = this._armBackP[side];
      if (this._armBackOk[side] && elbowTrust < 1) {
        const carried = this._t4.copy(backPrev)
          .addScaledVector(this._armUpper, -backPrev.dot(this._armUpper));
        if (carried.lengthSq() > 1e-8) {
          carried.normalize();
          if (armBack.dot(carried) < 0) armBack.negate();
          armBack.lerp(carried, 1 - elbowTrust);
          if (armBack.lengthSq() < 1e-8) armBack.copy(carried);
          armBack.normalize();
        }
      }
      if (armBack.lengthSq() < 1e-8) {
        // First frame with no elbow bend at all: the torso's back is the
        // stable answer until a real bend arrives.
        armBack.copy(up).addScaledVector(this._armUpper,
          -up.dot(this._armUpper));
      }
      if (armBack.lengthSq() < 1e-8) armBack.copy(shoulderFwd).negate();

      // The shoulder's roll can be slowed or clamped (TWIST_LIMIT), so solve
      // the upper arm FIRST: that clamped roll decides which plane the elbow
      // is allowed to bend in. The forearm must be pinned into the plane the
      // avatar's shoulder actually reached, not the one the measurement asked
      // for - pinning into the unrotated plane and then rolling the shoulder
      // left the rendered elbow out of its own hinge by the whole clamp (up to
      // 100 degrees, measured), which is exactly the sideways elbow no arm has.
      const rollAdj = this.poseBone(`${side}UpperArm`, this._armUpper, armBack, w);
      if (rollAdj) {
        this._fQb.setFromAxisAngle(this._armUpper, rollAdj);
        armBack.applyQuaternion(this._fQb);
      }

      // The elbow is a hinge: pull a wrist the reconstruction put outside the
      // joint's range back to the nearest direction a human arm can reach
      // before the forearm roll is derived from that (now legal) bend.
      this.limitElbow(side, this._armUpper, this._armFore, armBack,
        this._armFore);

      // Back of the FOREARM: an elbow is a hinge, so the forearm has no roll
      // of its own to measure - it inherits the upper arm's roll plus the
      // flexion. Rotating the upper arm's back reference by the elbow bend
      // gives the anatomically possible answer at every flexion angle,
      // including the 90-degree pose that used to collapse to an arbitrary
      // sideways twist. Pronation/supination is a spin about the forearm and
      // does not move this reference, so it is unaffected.
      const foreBack = this._armForeBack.copy(armBack);
      this._fQa.setFromUnitVectors(this._armUpper, this._armFore);
      foreBack.applyQuaternion(this._fQa);
      foreBack.addScaledVector(this._armFore, -foreBack.dot(this._armFore));
      // A perfectly straight elbow leaves the bend undefined but the roll it
      // already has, which is the upper arm's.
      if (foreBack.lengthSq() < 1e-8) foreBack.copy(armBack);
      foreBack.normalize();

      backPrev.copy(armBack);
      backPrev.addScaledVector(this._armUpper, -backPrev.dot(this._armUpper));
      if (backPrev.lengthSq() > 1e-8) {
        backPrev.normalize();
        this._armBackOk[side] = true;
      }

      this.poseBone(`${side}LowerArm`, this._armFore, foreBack, w);

      // Hand: the wrist -> middle knuckle direction transfers wrist flex; the
      // roll follows the forearm. A wrist outside the frame has no knuckle to
      // read, so the hand simply continues the forearm.
      this._armKnuck.copy(this._armFore);
      const knuckle = this.lm(`${p}_middle`);
      if (knuckle) {
        this._armKnuck.copy(knuckle).sub(j[SMPL[`${p}Wrist`]]);
        if (this._armKnuck.lengthSq() > 1e-8) {
          this._armKnuck.normalize()
            .lerp(this._armFore, 1 - armSeen.lower);
          if (this._armKnuck.lengthSq() < 1e-8) {
            this._armKnuck.copy(this._armFore);
          }
        } else {
          this._armKnuck.copy(this._armFore);
        }
      }
      this.poseBone(`${side}Hand`, this._armKnuck, foreBack, w);

      // Legs: the front of the thigh and shin follows the torso. A leg the
      // camera cannot see is a leg the reconstruction invented, and it invents
      // a crouch; a hidden leg standing straight under its hip is both the
      // likelier truth and the only stance that cannot leave the avatar folded
      // in mid air with its feet off the floor.
      const legW = this._legSeen[p];
      this.poseBone(`${side}UpperLeg`,
        this.legDir(SMPL[`${p}Hip`], SMPL[`${p}Knee`], legW, this._t5),
        shoulderFwd, w);
      this.poseBone(`${side}LowerLeg`,
        this.legDir(SMPL[`${p}Knee`], SMPL[`${p}Ankle`], legW, this._t6),
        shoulderFwd, w);

      // Foot: ankle, toe and heel form a triangle whose lateral normal, crossed
      // with the sole line, gives the sole's upward normal (foot roll).
      let footUp = this._t2.copy(up);
      const heel = this.lm(`${p}_heel`);
      if (heel) {
        const ankle = j[SMPL[`${p}Ankle`]];
        const toe = j[SMPL[`${p}Toe`]];
        this._t3.copy(toe).sub(ankle);
        this._t4.copy(heel).sub(ankle);
        this._t1.crossVectors(this._t3, this._t4);
        if (this._t1.lengthSq() > 1e-10) {
          this._t3.copy(toe).sub(heel);
          this._t1.crossVectors(this._t1, this._t3);
          if (this._t1.lengthSq() > 1e-10) {
            if (this._t1.dot(up) < 0) this._t1.negate();
            footUp = this._t2.copy(
              this.smoothDir(`footUp${side}`, this._t1, dt));
          }
        }
      }
      // A cropped leg's ankle/toe/heel triangle is invented too, so the sole
      // falls back to flat-forward instead of following a heel that is not
      // there.
      if (legW < 0.999) footUp.lerp(up, 1 - legW).normalize();
      this._t5.copy(j[SMPL[`${p}Toe`]]).sub(j[SMPL[`${p}Ankle`]]);
      if (this._t5.lengthSq() < 1e-10) this._t5.copy(shoulderFwd);
      this._t5.normalize();
      if (legW < 0.999) {
        this._t5.lerp(shoulderFwd, 1 - legW);
        if (this._t5.lengthSq() < 1e-8) this._t5.copy(shoulderFwd);
        this._t5.normalize();
      }
      // No yaw clamp here. An earlier one constrained the ankle to the PELVIS
      // forward, which is not the frame an ankle lives in: a dancer turns the
      // whole leg out, the pelvis keeps facing forward, and a foot forced to
      // agree with the pelvis has to counter-rotate against its own shin - the
      // boot corkscrewed round on every hip turn-out. The honest reference is
      // the knee plane, and the reconstruction already respects it: measured
      // against the plane through the knee, the person's ankle yaw stays inside
      // +-35 degrees on every frame of the live feed, and so does the avatar's
      // (0 frames past 90, against 30-35 before the clamp went in).
      this.poseBone(`${side}Foot`, this._t5, footUp, w);
    }

    // --- root translation -------------------------------------------------
    // Horizontal travel is optional and confined to a cone, so a bad frame
    // cannot walk the avatar off its mark. Vertical is not optional: it is the
    // centre of gravity. The legs are aimed independently, so if the root keeps
    // its height a crouch simply folds the legs and leaves the feet hanging in
    // the air. `hipRef` is the person's standing height, so the root drops by
    // exactly as much as the legs fold, scaled to this avatar's leg.
    const scale = clamp(this.rest.torso / Math.max(0.05, this._personTorso),
      0.4, 1.6);
    // 1 while the legs are measured, 0 in a close-up where they are pinned
    const ground = 1 - this._crop;
    let x = this._rootDelta.x * scale;
    let z = this._rootDelta.z * scale;
    const radius = Math.hypot(x, z);
    if (radius > ROOT_CONE) {
      const pull = ROOT_CONE / radius;
      x *= pull;
      z *= pull;
    }
    let y = 0;
    if (this.hipRef > 0 && this._hipHeight > 0) {
      const legScale = clamp(this.rest.leg / Math.max(0.3, this._personLeg),
        0.5, 1.8);
      const drop = (this.hipRef - this._hipHeight) * legScale;
      y = clamp(-drop, -ROOT_RISE, ROOT_DROP);
    }
    // The crouch above is measured on the person; this avatar's legs are a
    // different length and the reconstruction's standing height is only known
    // to a few centimetres, so trim the residual slowly. The bones are already
    // posed at this point, so their world height is the ground truth: if the
    // support ankle hangs above its bind height, sink the root until it does
    // not. Slow enough that a jump or a bounce never fights it.
    if (this.rest.sole !== undefined) {
      // The skin, not the ankle: a pointed toe raises the ankle but drops the
      // sole, and it is the sole that must stay on the floor.
      const soleY = Math.min(
        this.solePoint('left', this._soleA).y,
        this.solePoint('right', this._soleB).y);
      const err = soleY - this.rest.sole;
      this._trim = clamp(this._trim - err * damp(dt, TAU_TRIM) * ground,
        -ROOT_TRIM, ROOT_TRIM);
      y += this._trim;
    }
    // In a close-up there is no measured height to follow and no measured
    // travel to walk: the legs are pinned, so keep the avatar at its standing
    // height and on its mark instead of following the invented crouch.
    y *= ground;
    const follow = this.rootFollow ? ground : 0;
    this._target.set(x * weight * follow, y * weight,
      z * weight * follow);
    // the root is the one translation the avatar has, so it gets the same
    // filter as the joints: a pelvis that shivers by a centimetre shakes the
    // whole character, feet included
    if (this.smoothing.active) this.smoothing.vector(this._target, 'root', dt);

    // Hard, one-sided floor solve. The crouch term above is feed-forward from
    // the person's hip height, but these legs are aimed by direction with fixed
    // bone lengths, so the two disagree by centimetres on a fast step. The
    // slow trim is an integrator that arrives a second late - which is how the
    // soles ended up 9 cm under the floor. The posed soles are ground truth,
    // so clamp the target after smoothing (otherwise the filter drags it back
    // down) and recover quickly.
    const floor = this._floorSample(ground);
    let floored = false;
    if (floor.on) {
      const fit = floorTargetY(this._target.y, floor.rootY, floor.soleY,
        this.rest.sole, floor.ground);
      this._target.y = fit.y;
      floored = fit.floored;
    }
    const k = damp(dt, TAU_ROOT);
    this.root.position.x += (this._target.x - this.root.position.x) * k;
    this.root.position.z += (this._target.z - this.root.position.z) * k;
    const ky = damp(dt, floored ? FLOOR_TAU : TAU_ROOT_Y);
    this.root.position.y += (this._target.y - this.root.position.y) * ky;

    // The plant IK rewrites leg rotations, so it has to be the LAST pass that
    // touches the legs. Running the global filter after it (as this used to)
    // meant every frame ended by blending the corrected thigh and shin back
    // towards the unfiltered reconstruction, which walks the sole straight off
    // its anchor - with contact now held on ~93% of frames that is the
    // difference between 0.5 cm and 18 cm of drift per stance. Filtering
    // first is also the calmer order: the IK's two-bone solve is a function of
    // the (already filtered) hip position and a fixed target, so its output is
    // smooth by construction and does not need a second filter.
    this._applyOutputFilter(dt);
    this._floorClamp(this._floorSample(ground));
    this._plantFeet(dt, weight, ground);
    this._floorClamp(this._floorSample(ground));
  }

  /**
   * One sample of the posed soles, taken wherever the floor is enforced.
   * Uses a reusable scratch record: the 3D path samples twice per frame (once
   * before the root is written, once after the output filter), and neither
   * sample allocates.
   *
   * `solePoint()` refreshes the bone world matrices, so `soleY` is measured
   * with the root still at `rootY`. Moving the root by d moves every sole by
   * d, which makes `rootY - (soleY - rest.sole)` an absolute height that does
   * not depend on where the root currently is. That is what lets grounding.js
   * solve the floor instead of integrating towards it.
   */
  _floorSample(ground) {
    const f = this._floorScratch || (this._floorScratch = {});
    f.on = this.rest.sole !== undefined && ground > 0.001;
    if (!f.on) return f;
    f.rootY = this.root.position.y;
    f.soleY = Math.min(
      this.solePoint('left', this._soleA).y,
      this.solePoint('right', this._soleB).y);
    f.ground = ground;
    return f;
  }

  /** The final one-sided constraint on the root position actually written. */
  _floorClamp(f) {
    if (!f || !f.on) return;
    this.root.position.y = floorTargetY(this.root.position.y, f.rootY,
      f.soleY, this.rest.sole, f.ground).y;
  }

  /**
   * Hold every planted foot where it landed (footlock.js + footik.js).
   *
   * Detection runs on the NATURAL pose, so the IK can never hide a real step
   * from the contact detector; the IK then re-poses each leg independently.
   * The root is left completely alone - it is the one transform shared by
   * both feet, so moving it can only trade foot skate for whole-body sliding.
   *
   * Order matters: this runs AFTER the global output filter (see update3D),
   * because a filter applied to the corrected legs drags the sole off its
   * anchor; the filter is followed by one last floor clamp, so neither pass
   * can push a sole through the floor.
   */
  _plantFeet(dt, weight, ground) {
    const lock = this.footLock;
    const strength = weight * ground;
    if (this.rest.sole === undefined ||
        !this.b.leftFoot || !this.b.rightFoot) return;

    // Natural soles, before any IK has touched a bone. Detection runs even
    // with the pass switched off, so toggling the panel back on never picks
    // up a stale anchor (and the measurement probes can compare ON and OFF
    // on the very same contact windows). The soles decide contact; the ankles
    // are the points the hold actually pins (see footlock.js update()).
    const left = this.solePoint('left', this._soleA);
    const right = this.solePoint('right', this._soleB);
    const leftAnkle = this.b.leftFoot.getWorldPosition(this._ankleA);
    const rightAnkle = this.b.rightFoot.getWorldPosition(this._ankleB);
    const rx = this.root.position.x;
    const rz = this.root.position.z;
    const feet = lock.update(left, right, [leftAnkle, rightAnkle],
      this.rest.sole, rx, rz, dt, strength);

    // Ease the constraint in/out with the checkbox and with the close-up
    // fade, instead of snapping a leg to its anchor on the first frame.
    const want = lock.enabled && strength > 1e-3 ? 1 : 0;
    const gain = this._plantGain || 0;
    this._plantGain = gain + (want - gain) * damp(dt, 0.12);
    // Call plant for BOTH sides every frame, even at weight 0: footik.js
    // low-passes its correction and needs the chance to decay it to zero.
    // Skipping those frames is what made a lifting leg snap and shake.
    for (let i = 0; i < 2; i++) {
      const foot = feet[i];
      const side = i ? 'right' : 'left';
      this.footIK.plant(side, foot.anchor.x + rx, foot.anchor.y + rz,
        foot.weight * strength * this._plantGain, dt, foot.fresh);
    }
  }

  /**
   * The global smoothing pass over the finished pose.
   *
   * Every bone's local rotation has been written by now (by the retargeter, or
   * left to the idle mixer for the bones it does not drive), so filtering here
   * smooths whatever the per-limb logic decided *and* any residual from the
   * animation underneath. Local rotations are used rather than world ones on
   * purpose: the parent is already smooth, so a local low-pass also passes
   * everything the parent does without a second copy of the same lag.
   */
  _applyOutputFilter(dt) {
    if (!this.smoothing.active) return;
    for (let i = 0; i < this._smoothBones.length; i++) {
      const entry = this._smoothBones[i];
      this.smoothing.quaternion(entry[1].quaternion, entry[0], dt);
    }
  }

  setRootFollow(on) {
    this.rootFollow = !!on;
    // recalibrate at the current pose so enabling follow never snaps the
    // avatar to an offset accumulated while it was pinned
    this.rootRef = null;
    this._rootDelta.set(0, 0, 0);
  }

  reset() {
    this.blend = 0;
    this.hasJoint.fill(0);
    this.dirState.clear();
    this.limbRef.clear();
    this._rollPrev = Object.create(null);
    for (const key of Object.keys(this._lm3)) this._lm3[key].valid = false;
    this._hipFwd.set(0, 0, 1);
    this._headFwd.set(0, 0, 1);
    this.torsoPx = 0;
    this.hipRef = 0;
    this._hipHeight = 0;
    this._personLeg = 0;
    this._trim = 0;
    this._legSeen.left = 1;
    this._legSeen.right = 1;
    for (const side of ['left', 'right']) {
      this._armSeen[side].upper = 1;
      this._armSeen[side].lower = 1;
      this._armBackOk[side] = false;
      this._foreDirOk[side] = false;
    }
    this._crop = 0;
    this._cropQw.identity();
    this.facing = 1;
    this.head.roll = 0;
    this.head.yaw = 0;
    this.face.q.identity();
    this.face.target.identity();
    this.face.headW = 0;
    this.face.mouthRef = 0.25;
    this.face.weights = Object.create(null);
    this.faceOverride = null;
    this.gaze.w = 0;
    this.gazeSmoothed.x = 0;
    this.gazeSmoothed.y = 0;
    this.releaseLookAt(this.vrm && this.vrm.lookAt);
    this.root.position.set(0, 0, 0);
    this.rootRef = null;
    this._rootDelta.set(0, 0, 0);
    this.smoothing.reset();
    this.footLock.reset();
    this._plantGain = 0;
    if (this.footIK) {
      this.footIK.corr.left.set(0, 0);
      this.footIK.corr.right.set(0, 0);
      this.footIK._prevW.left = 0;
      this.footIK._prevW.right = 0;
    }
  }
}
