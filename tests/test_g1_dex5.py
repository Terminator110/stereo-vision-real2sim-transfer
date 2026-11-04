import importlib.util
import sys
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
spec = importlib.util.spec_from_file_location('g1_dex5',ROOT/'code/10_g1_dex5_stationary.py')
g1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g1)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT/'models/scenario_g1_dex5_stationary.xml'))
        self.neutral = g1.setup_pose(self.model)

    def test_joint_names_and_counts(self):
        names = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,j) for j in range(self.model.njnt)]
        self.assertEqual(self.model.nq,69)
        self.assertNotIn('g1_01_floating_base_joint',names)
        original = ET.parse(ROOT/'models/robots/unitree_g1_29dof/unitree_g1_29dof_g1_01.xml')
        expected = [j.get('name') for j in original.findall('.//joint') if j.get('name') != 'g1_01_floating_base_joint']
        self.assertEqual(len(expected),29)
        self.assertTrue(set(expected)<=set(names))
        for side in ('left','right'):
            self.assertEqual(sum(n.startswith(f'g1_01_dex5_{side}_') for n in names),20)
            self.assertEqual(mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,f'g1_01_{side}_rubber_hand'),-1)

    def test_mount_direction(self):
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.neutral
        mujoco.mj_forward(self.model,data)
        for side in ('left','right'):
            palm = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,f'g1_01_dex5_{side}_palm')
            wrist = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,f'g1_01_{side}_wrist_yaw_link')
            self.assertGreater(np.dot(data.xmat[palm].reshape(3,3)[:,1],data.xmat[wrist].reshape(3,3)[:,0]),0.999)

    def test_original_palm_up_mounts(self):
        expected = {'right':[1,0,0,-1], 'left':[0,1,1,0]}
        for side,quaternion in expected.items():
            palm = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,f'g1_01_dex5_{side}_palm')
            quaternion = np.asarray(quaternion,dtype=float)
            quaternion /= np.linalg.norm(quaternion)
            np.testing.assert_allclose(self.model.body_quat[palm],quaternion,atol=1e-8)

    def test_demo_only_moves_hands_and_wrists_within_limits(self):
        allowed = [j for j in range(self.model.njnt) if 'dex5_' in mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,j)
                   or '_wrist_' in mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,j)]
        fixed = sorted(set(range(69))-{int(self.model.jnt_qposadr[j]) for j in allowed})
        for t in np.linspace(0,4,60):
            pose = g1.demo_pose(self.model,self.neutral,t)
            np.testing.assert_array_equal(pose[fixed],self.neutral[fixed])
            for j in range(self.model.njnt):
                self.assertGreaterEqual(pose[self.model.jnt_qposadr[j]],self.model.jnt_range[j,0]-1e-9)
                self.assertLessEqual(pose[self.model.jnt_qposadr[j]],self.model.jnt_range[j,1]+1e-9)

    def test_wrist_rotation_hold_and_recalibration(self):
        tracker = g1.WristTracker(self.model)
        hand = np.zeros((21,3))
        hand[5],hand[9],hand[17] = [-0.04,0.07,0],[0,0.08,0],[0.04,0.07,0]
        for i in range(5):
            tracker.update(hand,i/30)
        self.assertIsNotNone(tracker.reference)
        rotated = hand @ Rotation.from_euler('y',25,degrees=True).as_matrix().T
        previous = tracker.previous.copy()
        for i in range(5,125):
            q,state,error = tracker.update(rotated,i/30)
            self.assertLessEqual(np.abs(q-previous).max(),1.5/30+1e-8)
            self.assertTrue((q>=tracker.bounds[:,0]).all() and (q<=tracker.bounds[:,1]).all())
            previous = q.copy()
        self.assertLess(error,2)
        self.assertGreater(np.linalg.norm(q),0.1)
        held,state,_ = tracker.update(None,5)
        np.testing.assert_array_equal(held,q)
        self.assertEqual(state,'LOST_HOLD')
        tracker.reset()
        self.assertIsNone(tracker.reference)
        np.testing.assert_array_equal(tracker.previous,q)


if __name__ == '__main__':
    unittest.main()
