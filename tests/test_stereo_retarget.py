import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stderr
import io

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
spec = importlib.util.spec_from_file_location('stereo_app', ROOT/'code/12_stereo_g1_dex5.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
CAL = ROOT/'code/stereo/calibration/results_new/stereo_calibration.npz'
TABLE = ROOT/'code/stereo/calibration/results_new/table_frame_multi.npz'


def hand_points():
    points = np.zeros((21, 3))
    for chain, x, y in [((5, 6, 7, 8), 0.03, 0.07), ((9, 10, 11, 12), 0.01, 0.075),
                        ((13, 14, 15, 16), -0.01, 0.07), ((17, 18, 19, 20), -0.03, 0.06)]:
        for i, index in enumerate(chain):
            points[index] = [x, y+i*0.025, i*0.004]
    points[1:5] = [[0.035, 0.02, 0], [0.055, 0.03, 0.005],
                   [0.07, 0.05, 0.012], [0.075, 0.075, 0.02]]
    return points+[0.02, 0.02, 0.12]


class StereoTests(unittest.TestCase):
    def setUp(self):
        self.cal, self.size, self.table_r, self.table_t = app.load_calibration(CAL, TABLE)
        self.table = hand_points()
        self.camera = (self.table-self.table_t) @ self.table_r
        self.left, self.right = self.project(self.camera)

    def project(self, points):
        a, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), self.cal['K1'], self.cal['D1'])
        rvec, _ = cv2.Rodrigues(self.cal['R'])
        b, _ = cv2.projectPoints(points, rvec, self.cal['T'], self.cal['K2'], self.cal['D2'])
        return a.reshape(-1, 2), b.reshape(-1, 2)

    def test_real_calibration_roundtrip(self):
        result = app.reconstruct_hand(self.left, self.right, self.cal, self.table_r, self.table_t)
        self.assertEqual(result['status'], 'STEREO_VALID')
        np.testing.assert_allclose(result['points_cam1'], self.camera, atol=1e-5)
        np.testing.assert_allclose(result['points_table'], self.table, atol=1e-5)
        self.assertLess(result['errors'].max(), 0.01)

    def test_defaults_use_correct_tags_and_new_calibration_paths(self):
        with patch.object(sys, 'argv', ['12_stereo_g1_dex5.py']):
            args = app.parse_args()
        self.assertEqual(args.tag_ids, [4, 5])
        self.assertEqual((args.left, args.right), (2, 0))
        self.assertIn('results_new', args.calib.parts)
        self.assertIn('results_new', args.table_frame.parts)

    def test_single_video_is_not_accepted_as_stereo(self):
        with patch.object(sys, 'argv', ['12_stereo_g1_dex5.py', '--left-video', 'one.mp4']), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                app.parse_args()

    def test_reconstructed_palm_orientation_matches_table_frame(self):
        result = app.reconstruct_hand(self.left, self.right, self.cal, self.table_r, self.table_t)
        points = result['points_table']
        actual = app.g1.palm_frame(points[0], points[5], points[9], points[17])
        camera = app.g1.palm_frame(self.camera[0], self.camera[5], self.camera[9], self.camera[17])
        np.testing.assert_allclose(actual, self.table_r @ camera, atol=1e-5)

    def test_no_monocular_depth_is_read(self):
        class Landmark:
            x, y = 0.4, 0.5
            @property
            def z(self):
                raise AssertionError('Monocular landmark z must not be read')
        class Result:
            hand_landmarks = [[Landmark() for _ in range(21)]]
            handedness = [[SimpleNamespace(category_name='Left')]]
            @property
            def hand_world_landmarks(self):
                raise AssertionError('Monocular world landmarks must not be read')
        points, label = app.image_landmarks(Result(), self.size)
        self.assertEqual(points.shape, (21, 2))
        self.assertEqual(label, 'Left')
        np.testing.assert_allclose(points[0], np.array(self.size)*[0.4, 0.5])

    def test_missing_view_never_uses_other_camera_alone(self):
        for a, b, status in [(None, self.right, 'MISSING_CAM1'), (self.left, None, 'MISSING_CAM2')]:
            result = app.reconstruct_hand(a, b, self.cal, self.table_r, self.table_t)
            self.assertEqual(result['status'], status)
            self.assertIsNone(result['points_table'])

    def test_one_bad_tip_does_not_reject_other_fingers_or_palm(self):
        bad = self.right.copy()
        bad[8] += [40, 40]
        result = app.reconstruct_hand(self.left, bad, self.cal, self.table_r, self.table_t)
        self.assertEqual(result['status'], 'PARTIAL_STEREO')
        self.assertFalse(result['landmark_valid'][8])
        self.assertTrue(np.isnan(result['points_table'][8]).all())
        self.assertFalse(result['groups']['index'])
        self.assertTrue(all(good for name, good in result['groups'].items() if name != 'index'))

    def test_nonfinite_point_isolated_without_monocular_fallback(self):
        bad = self.left.copy()
        bad[8, 0] = np.nan
        result = app.reconstruct_hand(bad, self.right, self.cal, self.table_r, self.table_t)
        self.assertEqual(result['status'], 'PARTIAL_STEREO')
        self.assertTrue(result['groups']['wrist'])
        self.assertFalse(result['groups']['index'])
        self.assertTrue(np.isnan(result['points_table'][8]).all())

    def test_negative_depth_point_isolated(self):
        camera = self.camera.copy()
        camera[8] -= [0, 0, 3]
        a, b = self.project(camera)
        result = app.reconstruct_hand(a, b, self.cal, self.table_r, self.table_t)
        self.assertEqual(result['status'], 'PARTIAL_STEREO')
        self.assertFalse(result['landmark_valid'][8])
        self.assertTrue(result['groups']['wrist'])

    def test_implausible_bone_disables_only_its_group(self):
        camera = self.camera.copy()
        camera[8] += [0.3, 0, 0]
        a, b = self.project(camera)
        result = app.reconstruct_hand(a, b, self.cal, self.table_r, self.table_t)
        self.assertTrue(result['landmark_valid'][8])
        self.assertFalse(result['groups']['index'])
        self.assertTrue(result['groups']['middle'])
        self.assertTrue(result['groups']['wrist'])

    def test_scoped_finger_math_matches_existing_mapper(self):
        expected = app.g1.retarget.landmark_angles(self.table)
        for name in app.g1.retarget.CHAINS:
            actual = app.finger_angles(self.table, name)
            for joint in app.g1.retarget.FINGERS[name][1:]:
                self.assertAlmostEqual(actual[joint], expected[joint], places=12)

    def test_vertical_flip_of_measurements_rejected(self):
        flipped = self.right.copy()
        flipped[:, 1] = self.size[1]-1-flipped[:, 1]
        self.assertNotEqual(app.reconstruct_hand(self.left, flipped, self.cal, self.table_r, self.table_t)['status'], 'STEREO_VALID')

    def test_negative_depth_rejected(self):
        a, b = self.project(self.camera-[0, 0, 3])
        result = app.reconstruct_hand(a, b, self.cal, self.table_r, self.table_t)
        self.assertNotEqual(result['status'], 'STEREO_VALID')

    def test_inconsistent_table_calibration_rejected(self):
        with np.load(TABLE, allow_pickle=False) as archive:
            changed = {key: archive[key] for key in archive.files}
        changed['t_table_to_cam2'] = changed['t_table_to_cam2']+[0.1, 0, 0]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'table.npz'
            np.savez(path, **changed)
            with self.assertRaises(ValueError):
                app.load_calibration(CAL, path)

    def test_degenerate_hand_is_rejected(self):
        a, b = self.project(np.repeat(self.camera[:1], 21, axis=0))
        self.assertNotEqual(app.reconstruct_hand(a, b, self.cal, self.table_r, self.table_t)['status'], 'STEREO_VALID')

    def test_model_only_fingers_and_wrist_change(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT/'models/scenario_g1_dex5_stationary.xml'))
        controller = app.StereoRetargeter(model)
        allowed = set(controller.wrist.addresses.tolist()) | {a for a, _, _ in controller.bindings.values()}
        fixed = sorted(set(range(model.nq))-allowed)
        reconstructed = app.reconstruct_hand(self.left, self.right, self.cal, self.table_r, self.table_t)['points_table']
        for i in range(10):
            pose, state = controller.update(reconstructed, i/30)
            self.assertTrue(np.isfinite(pose).all())
            np.testing.assert_array_equal(pose[fixed], controller.neutral[fixed])
        self.assertIsNotNone(controller.wrist.reference)
        _, state = controller.update(None, 0.5)
        self.assertEqual(state['wrist_state'], 'LOST_HOLD')

    def test_input_xyz_translation_does_not_move_robot(self):
        models = [mujoco.MjModel.from_xml_path(str(ROOT/'models/scenario_g1_dex5_stationary.xml')) for _ in range(2)]
        first, second = [app.StereoRetargeter(model) for model in models]
        for i in range(8):
            pose1, _ = first.update(self.table, i/30)
            pose2, _ = second.update(self.table+[0.5, -0.25, 0.4], i/30)
            np.testing.assert_allclose(pose1, pose2, atol=1e-6)

    def test_rejected_frames_do_not_complete_wrist_calibration(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT/'models/scenario_g1_dex5_stationary.xml'))
        controller = app.StereoRetargeter(model)
        for i in range(4):
            controller.update(self.table, i/30)
        self.assertIsNone(controller.wrist.reference)
        controller.update(None, 0.2)
        self.assertEqual(len(controller.wrist.pending), 0)

    def controller(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT/'models/scenario_g1_dex5_stationary.xml'))
        return app.StereoRetargeter(model)

    def test_quality_log_keeps_thumb_solver_and_control_states_distinct(self):
        headers = app.quality_csv_header(self.controller())
        self.assertEqual(len(headers), len(set(headers)))
        self.assertIn('thumb_state', headers)
        self.assertIn('thumb_control_state', headers)
        self.assertIn('index_state', headers)

    def test_partial_hand_calibrates_wrist(self):
        controller = self.controller()
        points = self.table.copy()
        points[8] = np.nan
        for i in range(5):
            _, state = controller.update(points, i/30)
        self.assertIsNotNone(controller.wrist.reference)
        self.assertEqual(state['group_states']['index'], 'HOLD')
        self.assertEqual(state['group_states']['middle'], 'TRACKING')
        self.assertEqual(state['calibration'], 'Wrist calibration complete')

    def test_missing_palm_point_resets_calibration_but_not_unrelated_finger(self):
        controller = self.controller()
        for i in range(4):
            controller.update(self.table, i/30)
        points = self.table.copy()
        points[9] = np.nan
        _, state = controller.update(points, 0.2)
        self.assertIsNone(controller.wrist.reference)
        self.assertEqual(len(controller.wrist.pending), 0)
        self.assertEqual(state['group_states']['index'], 'TRACKING')
        self.assertEqual(state['group_states']['middle'], 'HOLD')
        self.assertEqual(state['group_states']['thumb'], 'HOLD')
        self.assertIn('waiting', state['calibration'])

    def test_moving_palm_restarts_then_steady_palm_completes_calibration(self):
        controller = self.controller()
        controller.update(self.table, 0.0)
        rotation, _ = cv2.Rodrigues(np.array([0, 0, np.deg2rad(15)]))
        rotated = (self.table-self.table[0]) @ rotation.T + self.table[0]
        _, state = controller.update(rotated, 1/30)
        self.assertEqual(state['wrist_state'], 'CALIBRATION_MOVING')
        self.assertEqual(len(controller.wrist.pending), 1)
        self.assertIsNone(controller.wrist.reference)
        for i in range(2, 6):
            controller.update(rotated, i/30)
        self.assertIsNotNone(controller.wrist.reference)

    def test_invalid_group_holds_indefinitely_while_other_group_updates(self):
        controller = self.controller()
        for i in range(8):
            before, _ = controller.update(self.table, i/30)
        index_addresses = [controller.bindings[joint][0] for joint in app.g1.retarget.FINGERS['index']]
        partial = self.table.copy()
        partial[8] = np.nan
        partial[14:17, 2] += [0.015, 0.03, 0.04]
        ring_addresses = [controller.bindings[joint][0] for joint in app.g1.retarget.FINGERS['ring']]
        for i in range(60):
            pose, state = controller.update(partial, 0.5+i/30)
            np.testing.assert_array_equal(pose[index_addresses], before[index_addresses])
        self.assertEqual(state['group_states']['index'], 'HOLD')
        self.assertGreater(np.max(np.abs(pose[ring_addresses]-before[ring_addresses])), 0.02)
        for now in (3.0, 4.0, 8.0):
            held, state = controller.update(None, now)
            np.testing.assert_array_equal(held, pose)
            self.assertTrue(all(value == 'HOLD' for value in state['group_states'].values()))

    def test_bad_thumb_does_not_stop_fingers_or_wrist(self):
        controller = self.controller()
        points = self.table.copy()
        points[3] = np.nan
        for i in range(6):
            pose, state = controller.update(points, i/30)
        self.assertEqual(state['thumb_state'], 'LOST_HOLD')
        self.assertEqual(state['group_states']['thumb'], 'HOLD')
        self.assertTrue(all(state['group_states'][name] == 'TRACKING' for name in app.g1.retarget.CHAINS))
        self.assertIsNotNone(controller.wrist.reference)
        self.assertTrue(np.isfinite(pose).all())

    def test_thumb_solver_hold_does_not_advance_filtered_pose(self):
        controller = self.controller()
        controller.update(self.table, 0)
        thumb_addresses = [controller.bindings[joint][0] for joint in app.g1.retarget.FINGERS['thumb']]
        before = controller.pose[thumb_addresses].copy()
        with patch.object(controller.thumb, 'update', return_value=({}, 'OUTLIER_HOLD', 10.0)):
            pose, state = controller.update(self.table, 0.1)
        np.testing.assert_array_equal(pose[thumb_addresses], before)
        self.assertEqual(state['group_states']['thumb'], 'HOLD')


if __name__ == '__main__':
    unittest.main()
