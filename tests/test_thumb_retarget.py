import importlib.util
from pathlib import Path
import unittest
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('thumb_retarget',ROOT/'code/thumb_retarget.py')
thumb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(thumb)
spec = importlib.util.spec_from_file_location('baseline',ROOT/'code/07_retarget_dex5_right.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class ThumbTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT/'models/dex5_1/Dex5-URDF-R/Dex5-URDF-R.urdf'))
        self.bindings = baseline.joint_bindings(self.model)
        self.solver = thumb.ThumbRetargeter(self.model,self.bindings)

    def synthetic_hand(self,q):
        chain = self.solver.kinematics(q)
        points = np.zeros((21,3))
        points[1:5] = chain
        # Standalone root is world, so these use the exact model palm coordinates.
        for landmark,name in [(5,'Roll_21R'),(9,'Roll_31R'),(17,'Roll_51R')]:
            joint = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,name)
            points[landmark] = self.solver.data.xanchor[joint]
        return points

    def test_open_hand_not_artificially_curled(self):
        points = self.synthetic_hand(np.zeros(4))
        q,error = self.solver.solve(points)
        self.assertLess(error,0.5)
        self.assertLess(np.max(np.abs(q)),0.03)

    def test_all_four_joints_recover_known_poses(self):
        for expected in [np.array([0.3,0.4,0.8,0.5]),np.array([-0.25,0.9,1.0,0.7]),
                         np.array([0.5,0.8,1.3,0.8]),np.array([0.6,1.3,1.4,1.2])]:
            points = self.synthetic_hand(expected)
            q,error = self.solver.solve(points)
            self.assertLess(error,2.5)
            self.assertLess(np.max(np.abs(q-expected)),0.08)
            self.assertGreater(abs(q[0]),0.15)

    def test_rigid_viewpoint_invariance(self):
        from scipy.spatial.transform import Rotation
        points = self.synthetic_hand(np.array([0.3,0.4,0.8,0.5]))
        rotation = Rotation.from_euler('xyz',[50,20,-65],degrees=True).as_matrix()
        transformed = points @ rotation.T+[1,2,-3]
        np.testing.assert_allclose(self.solver.targets(points),self.solver.targets(transformed),atol=1e-8)

    def test_integrated_g1_mount_uses_same_thumb_motion(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT/'models/scenario_g1_dex5_stationary.xml'))
        bindings = {}
        for group in baseline.FINGERS.values():
            for name in group:
                joint = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'g1_01_dex5_right_'+name)
                bindings[name] = (int(model.jnt_qposadr[joint]),*model.jnt_range[joint])
        solver = thumb.ThumbRetargeter(model,bindings)
        expected = np.array([0.3,0.4,0.8,0.5])
        points = self.synthetic_hand(expected)
        rotation = solver.data.xmat[solver.palm].reshape(3,3)
        points = points @ rotation.T+solver.data.xpos[solver.palm]
        q,error = solver.solve(points)
        self.assertLess(error,2.5)
        self.assertLess(np.max(np.abs(q-expected)),0.08)

    def test_invalid_holds_previous(self):
        self.solver.previous[:] = [0.2,0.4,0.5,0.6]
        values,status,_ = self.solver.update(np.zeros((21,3)))
        self.assertEqual(status,'INVALID_HOLD')
        np.testing.assert_allclose(list(values.values()),self.solver.previous)

    def test_single_flip_is_rejected_but_sustained_change_is_allowed(self):
        self.solver.accepted = True
        self.solver.solve = lambda _: (np.array([0.4,1.0,1.0,0.8]),1.0)
        _,status,_ = self.solver.update(np.ones((21,3)))
        self.assertEqual(status,'OUTLIER_HOLD')
        _,status,_ = self.solver.update(np.ones((21,3)))
        self.assertEqual(status,'OUTLIER_HOLD')
        values,status,_ = self.solver.update(np.ones((21,3)))
        self.assertEqual(status,'TRACKING')
        self.assertAlmostEqual(values['Yaw_11R'],0.4)


if __name__ == '__main__':
    unittest.main()
