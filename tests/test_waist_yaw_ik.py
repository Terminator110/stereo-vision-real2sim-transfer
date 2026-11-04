"""Right arm + one waist-yaw DOF, without changing legacy arm-only behavior."""
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from g1_hand_ik import HandIK, HandIKController, HandTarget, quaternion
from g1_workspace import CollisionChecker


class WaistYawTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT/'models/scenario_g1_dex5_stationary.xml'))
        self.controller = HandIKController(self.model, ('right',), include_waist_yaw=True)
        self.arm = self.controller.arms['right']

    def test_exactly_eight_joints_and_only_waist_yaw_added(self):
        self.assertEqual(len(self.arm.addresses), 8)
        self.assertEqual(self.arm.joint_names[-1], 'g1_01_waist_yaw_joint')
        self.assertNotIn('g1_01_waist_roll_joint', self.arm.joint_names)
        self.assertNotIn('g1_01_waist_pitch_joint', self.arm.joint_names)
        np.testing.assert_allclose([self.arm.lower[-1], self.arm.upper[-1]], [-2.618, 2.618])

    def test_legacy_binding_stays_at_seven(self):
        self.assertEqual(len(HandIK(self.model, 'right').addresses), 7)

    def test_reject_independent_shared_waist_solves(self):
        with self.assertRaises(ValueError):
            HandIKController(self.model, ('left', 'right'), include_waist_yaw=True)
        with self.assertRaises(ValueError):
            HandIK(self.model, 'left', include_waist_yaw=True)

    def test_generated_pose_is_recovered_without_other_joint_changes(self):
        initial = self.controller.qpos.copy()
        desired = initial.copy()
        desired[self.arm.addresses] += [-.1, .1, .05, .2, .1, -.05, .1, .35]
        p, r = self.arm.hand_pose(desired)
        result = self.arm.solve(initial, HandTarget(p, quaternion(r)))
        self.assertTrue(result.converged)
        self.assertGreater(abs(result.candidate_qpos[self.arm.addresses[-1]]), 1e-4)
        fixed = np.setdiff1d(np.arange(self.model.nq), self.arm.addresses)
        np.testing.assert_array_equal(result.candidate_qpos[fixed], initial[fixed])
        np.testing.assert_array_equal(self.model.qpos0, initial)
        self.assertTrue(np.all(result.candidate_qpos[self.arm.addresses] >= self.arm.lower))
        self.assertTrue(np.all(result.candidate_qpos[self.arm.addresses] <= self.arm.upper))

    def test_waist_obeys_same_velocity_limit(self):
        initial = self.controller.qpos.copy()
        desired = initial.copy()
        desired[self.arm.addresses[-1]] = .4
        p, r = self.arm.hand_pose(desired)
        pose, states = self.controller.step({'right':HandTarget(p, quaternion(r))}, .02)
        self.assertTrue(states['right']['solver_converged'])
        self.assertLessEqual(np.max(np.abs(pose-initial)), .0200000001)

    def test_collision_scope_includes_torso_and_left_arm(self):
        old = CollisionChecker(self.model)
        checker = CollisionChecker(self.model, include_waist_yaw=True)
        self.assertGreater(len(checker.pairs), len(old.pairs))
        bodies = {int(self.model.geom_bodyid[g]) for pair in checker.pairs for g in pair}
        for name in ('g1_01_torso_link', 'g1_01_left_wrist_yaw_link'):
            self.assertIn(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name), bodies)
        self.assertEqual(checker.check(self.model.qpos0), [])


if __name__ == '__main__':
    unittest.main()
