import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stderr
import io

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from g1_hand_ik import HandIKController, quaternion
from g1_workspace import CollisionChecker
from trajectory_replay import trajectory_start_pose


class TrajectoryStartTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT/'models/scenario_g1_dex5_stationary.xml'))
        self.controller = HandIKController(self.model, ('right',))
        self.arm = self.controller.arms['right']
        self.original = self.controller.qpos.copy()
        p, r = self.arm.hand_pose(self.original)
        self.target = p+np.array([-.06, -.06, -.03])
        self.orientation = quaternion(r)

    def test_prepared_pose_is_checked_and_only_right_arm_differs(self):
        checker = CollisionChecker(self.model)
        pose, info = trajectory_start_pose(self.controller, checker, self.target, self.orientation)
        fixed = np.setdiff1d(np.arange(self.model.nq), self.arm.addresses)
        np.testing.assert_array_equal(pose[fixed], self.original[fixed])
        np.testing.assert_array_equal(self.controller.qpos, self.original)
        self.assertLess(info['first_pose_position_error_m'], .002)
        self.assertLess(info['first_pose_orientation_error_deg'], 2)
        self.assertFalse(info['initialization_is_motion'])
        self.assertEqual(checker.check(pose), [])

    def test_colliding_first_pose_is_rejected_without_changing_controller(self):
        checker = SimpleNamespace(check=lambda q: [dict(distance_m=-.01)])
        with self.assertRaisesRegex(ValueError, 'collides'):
            trajectory_start_pose(self.controller, checker, self.target, self.orientation)
        np.testing.assert_array_equal(self.controller.qpos, self.original)

    def test_unreachable_first_pose_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'IK solution'):
            trajectory_start_pose(self.controller, CollisionChecker(self.model), np.array([3., 0, 1]), self.orientation)
        np.testing.assert_array_equal(self.controller.qpos, self.original)

    def test_option_requires_explicit_trajectory_execution(self):
        spec = importlib.util.spec_from_file_location('ik_cli', ROOT/'code/13_test_g1_hand_ik.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        for argv in (['13', '--start-at-trajectory'], ['13', '--trajectory', 'x.csv', '--start-at-trajectory']):
            with patch.object(sys, 'argv', argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    cli.parse_args()
        with patch.object(sys, 'argv', ['13', '--trajectory', 'x.csv', '--execute-trajectory', '--start-at-trajectory', '--workspace-reviewed']):
            self.assertTrue(cli.parse_args().start_at_trajectory)


if __name__ == '__main__':
    unittest.main()
