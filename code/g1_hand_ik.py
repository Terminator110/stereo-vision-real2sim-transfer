"""Bounded, kinematic G1 arm IK. No fingers, dynamics, cameras, or hardware I/O."""
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ARM_JOINTS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
              'wrist_roll', 'wrist_pitch', 'wrist_yaw')
POSITION_TOLERANCE_M = 0.002
ORIENTATION_TOLERANCE_DEG = 2.0


@dataclass(frozen=True)
class HandTarget:
    position: np.ndarray
    quaternion: np.ndarray | None = None  # MuJoCo w,x,y,z; None holds current orientation.


@dataclass
class IKResult:
    converged: bool
    candidate_qpos: np.ndarray
    position_error_m: float
    orientation_error_deg: float
    evaluations: int


def quaternion(rotation):
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return xyzw[[3, 0, 1, 2]]


class HandIK:
    def __init__(self, model, side, include_waist_yaw=False):
        if side not in ('left', 'right'):
            raise ValueError('side must be left or right')
        self.model, self.side = model, side
        if include_waist_yaw and side != 'right':
            raise ValueError('Waist-yaw IK currently supports only the right arm')
        self.include_waist_yaw = bool(include_waist_yaw)
        self.data = mujoco.MjData(model)
        self.joint_names = [f'g1_01_{side}_{name}_joint' for name in ARM_JOINTS]
        if self.include_waist_yaw:
            self.joint_names.append('g1_01_waist_yaw_joint')
        self.ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                             for name in self.joint_names])
        if np.any(self.ids < 0):
            raise ValueError(f'Missing {side} arm joints')
        if np.any(model.jnt_type[self.ids] != mujoco.mjtJoint.mjJNT_HINGE) or not model.jnt_limited[self.ids].all():
            raise ValueError('Expected bounded arm/waist hinge joints')
        self.addresses = model.jnt_qposadr[self.ids].copy()
        self.lower, self.upper = model.jnt_range[self.ids].T.copy()
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f'g1_01_dex5_{side}_frame')
        if self.site < 0:
            raise ValueError(f'Missing {side} Dex5 hand frame')

    def hand_pose(self, qpos):
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (self.model.nq,) or not np.isfinite(qpos).all():
            raise ValueError('Expected a finite full-model qpos vector')
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def validate_target(self, qpos, target):
        position = np.asarray(target.position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('Target position must be three finite world coordinates in metres')
        if target.quaternion is None:
            _, rotation = self.hand_pose(qpos)
        else:
            q = np.asarray(target.quaternion, dtype=float)
            if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
                raise ValueError('Target quaternion must be finite, nonzero w,x,y,z')
            rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        return position.copy(), rotation

    def solve(self, qpos, target):
        qpos = np.asarray(qpos, dtype=float).copy()
        self.hand_pose(qpos)
        position, rotation = self.validate_target(qpos, target)
        seed = qpos[self.addresses].copy()
        if np.any(seed < self.lower) or np.any(seed > self.upper):
            raise ValueError('Starting arm joints exceed model limits')

        def residual(joints):
            self.data.qpos[:] = qpos
            self.data.qpos[self.addresses] = joints
            mujoco.mj_kinematics(self.model, self.data)
            actual_r = self.data.site_xmat[self.site].reshape(3, 3)
            error_r = Rotation.from_matrix(rotation.T @ actual_r).as_rotvec()
            # Metre-to-decimetre scaling plus a weak nearby-posture preference.
            return np.r_[10*(self.data.site_xpos[self.site]-position), error_r, 0.001*(joints-seed)]

        result = least_squares(residual, seed, bounds=(self.lower, self.upper),
                               max_nfev=80, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        candidate = qpos.copy()
        candidate[self.addresses] = result.x
        actual_p, actual_r = self.hand_pose(candidate)
        error_p = float(np.linalg.norm(actual_p-position))
        error_r = float(np.degrees(np.linalg.norm(Rotation.from_matrix(rotation.T @ actual_r).as_rotvec())))
        converged = bool(np.isfinite(candidate).all() and error_p <= POSITION_TOLERANCE_M
                         and error_r <= ORIENTATION_TOLERANCE_DEG)
        return IKResult(converged, candidate, error_p, error_r, int(result.nfev))


class HandIKController:
    """Accept solved targets, then slew-limit arm motion; failed targets hold."""
    def __init__(self, model, sides=('left', 'right'), max_velocity=1.0, include_waist_yaw=False):
        if not np.isfinite(max_velocity) or max_velocity <= 0:
            raise ValueError('max_velocity must be positive and finite')
        if not sides or len(set(sides)) != len(sides):
            raise ValueError('Choose distinct arm sides')
        if include_waist_yaw and tuple(sides) != ('right',):
            raise ValueError('Shared waist-yaw IK requires the right arm alone')
        self.model = model
        self.arms = {side: HandIK(model, side, include_waist_yaw) for side in sides}
        self.qpos = model.qpos0.copy()
        self.initial_qpos = self.qpos.copy()
        self.max_velocity = float(max_velocity)

    def step(self, targets, dt):
        if not np.isfinite(dt) or not 0 < dt <= 0.1+1e-12:
            raise ValueError('dt must be in (0, 0.1] seconds')
        dt = min(float(dt), 0.1)
        if not targets or any(side not in self.arms for side in targets):
            raise ValueError('Targets must select configured arm sides')
        # Validate every request before advancing either arm.
        goals = {side: self.arms[side].validate_target(self.qpos, target) for side, target in targets.items()}
        solutions = {side: self.arms[side].solve(self.qpos, HandTarget(p, quaternion(r)))
                     for side, (p, r) in goals.items()}
        for side, result in solutions.items():
            if result.converged:
                addresses = self.arms[side].addresses
                change = result.candidate_qpos[addresses]-self.qpos[addresses]
                self.qpos[addresses] += np.clip(change, -self.max_velocity*dt, self.max_velocity*dt)
        states = {}
        for side, (p, r) in goals.items():
            actual_p, actual_r = self.arms[side].hand_pose(self.qpos)
            error_p = float(np.linalg.norm(actual_p-p))
            error_r = float(np.degrees(np.linalg.norm(Rotation.from_matrix(r.T @ actual_r).as_rotvec())))
            result = solutions[side]
            reached = result.converged and error_p <= POSITION_TOLERANCE_M and error_r <= ORIENTATION_TOLERANCE_DEG
            states[side] = dict(status=('REACHED' if reached else 'MOVING') if result.converged else 'IK_FAILED_HOLD',
                                reached=bool(reached), solver_converged=result.converged,
                                position_error_m=error_p, orientation_error_deg=error_r,
                                solver_position_error_m=result.position_error_m,
                                solver_orientation_error_deg=result.orientation_error_deg,
                                solver_evaluations=result.evaluations)
        return self.qpos.copy(), states
