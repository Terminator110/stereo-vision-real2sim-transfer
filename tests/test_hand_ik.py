from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from g1_hand_ik import HandIK, HandIKController, HandTarget, quaternion


class HandIKTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT/'models/scenario_g1_dex5_stationary.xml'))
        self.controller = HandIKController(self.model)
        self.initial = self.controller.qpos.copy()

    def target(self, side, offset=(0.08, 0, 0.04)):
        arm = self.controller.arms[side]
        p, r = arm.hand_pose(self.initial)
        return HandTarget(p+np.array(offset), quaternion(r))

    def test_arm_binding_has_seven_joints_and_correct_site(self):
        for side, arm in self.controller.arms.items():
            self.assertEqual(len(arm.addresses), 7)
            self.assertEqual(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, arm.site),
                             f'g1_01_dex5_{side}_frame')

    def test_neutral_pose_is_unchanged_and_model_references_not_modified(self):
        saved = self.model.qpos0.copy()
        targets = {side: self.target(side, (0, 0, 0)) for side in self.controller.arms}
        pose, states = self.controller.step(targets, 0.02)
        np.testing.assert_array_equal(self.model.qpos0, saved)
        np.testing.assert_allclose(pose, self.initial, atol=1e-10)
        self.assertTrue(all(state['reached'] for state in states.values()))

    def test_both_arms_solve_reachable_full_pose(self):
        for side, arm in self.controller.arms.items():
            result = arm.solve(self.initial, self.target(side))
            self.assertTrue(result.converged)
            self.assertLess(result.position_error_m, 0.002)
            self.assertLess(result.orientation_error_deg, 2.0)
            fixed = np.setdiff1d(np.arange(self.model.nq), arm.addresses)
            np.testing.assert_array_equal(result.candidate_qpos[fixed], self.initial[fixed])
            self.assertTrue(np.all(result.candidate_qpos[arm.addresses] >= arm.lower))
            self.assertTrue(np.all(result.candidate_qpos[arm.addresses] <= arm.upper))

    def test_pose_generated_by_fk_can_be_recovered(self):
        for side, arm in self.controller.arms.items():
            source = self.initial.copy()
            source[arm.addresses] += [-0.12, 0.05, 0.07, 0.10, 0.08, -0.07, 0.06]
            p, r = arm.hand_pose(source)
            result = arm.solve(self.initial, HandTarget(p, quaternion(r)))
            self.assertTrue(result.converged)
            self.assertLess(result.position_error_m, 0.002)
            self.assertLess(result.orientation_error_deg, 2.0)

    def test_joint_speed_is_limited_until_pose_reached(self):
        target = self.target('right')
        previous = self.initial.copy()
        states = None
        for index in range(100):
            pose, states = self.controller.step({'right': target}, 0.02)
            if index == 0:
                self.assertTrue(states['right']['solver_converged'])
                self.assertFalse(states['right']['reached'])
                self.assertEqual(states['right']['status'], 'MOVING')
            self.assertLessEqual(float(np.max(np.abs(pose-previous))), 0.0200000001)
            previous = pose
        self.assertTrue(states['right']['reached'])

    def test_other_arm_fingers_waist_and_legs_stay_fixed(self):
        allowed = self.controller.arms['right'].addresses
        fixed = np.setdiff1d(np.arange(self.model.nq), allowed)
        for _ in range(20):
            pose, _ = self.controller.step({'right': self.target('right')}, 0.02)
            np.testing.assert_array_equal(pose[fixed], self.initial[fixed])

    def test_failed_target_holds_only_its_arm(self):
        targets = {'right': HandTarget(np.array([3.0, -0.2, 1.0])), 'left': self.target('left')}
        pose, states = self.controller.step(targets, 0.02)
        self.assertEqual(states['right']['status'], 'IK_FAILED_HOLD')
        self.assertFalse(states['right']['reached'])
        right = self.controller.arms['right'].addresses
        np.testing.assert_array_equal(pose[right], self.initial[right])
        self.assertTrue(states['left']['solver_converged'])

    def test_invalid_target_does_not_partially_advance_either_arm(self):
        bad = HandTarget(np.array([np.nan, 0, 1]))
        with self.assertRaises(ValueError):
            self.controller.step({'left': self.target('left'), 'right': bad}, 0.02)
        np.testing.assert_array_equal(self.controller.qpos, self.initial)

    def test_invalid_quaternion_rejected(self):
        for q in ([0, 0, 0, 0], [1, 0, np.nan, 0], [1, 0, 0]):
            with self.assertRaises(ValueError):
                self.controller.step({'right': HandTarget(np.array([.3, -.2, 1.0]), np.array(q))}, .02)

    def test_quaternion_wxyz_and_sign_equivalence(self):
        arm = self.controller.arms['right']
        rotation = Rotation.from_euler('xyz', [.1, -.2, .3]).as_matrix()
        q = quaternion(rotation)
        for sign in (-1, 1):
            _, actual = arm.validate_target(self.initial, HandTarget(np.array([.3, -.2, 1]), sign*q))
            np.testing.assert_allclose(actual, rotation, atol=1e-12)

    def test_bad_sides_dt_and_velocity_rejected(self):
        with self.assertRaises(ValueError):
            HandIK(self.model, 'drone')
        for sides in ((), ('right', 'right')):
            with self.assertRaises(ValueError):
                HandIKController(self.model, sides)
        for v in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                HandIKController(self.model, max_velocity=v)
        for dt in (0, -1, .2, np.nan):
            with self.assertRaises(ValueError):
                self.controller.step({'right': self.target('right')}, dt)


if __name__ == '__main__':
    unittest.main()
