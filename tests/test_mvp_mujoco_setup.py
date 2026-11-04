from pathlib import Path
import json
import sys
import unittest

import mujoco
import numpy as np

ROOT=Path('/home/sparx3d/physikal_workflow/dex-5')
sys.path.insert(0,str(ROOT/'code'))
from mujoco_mvp_setup import get_setup_state,camera_metadata
from scipy.spatial.transform import Rotation


class MVPSetupTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'config/mujoco_mvp_setup.json').read_text())
        self.model=mujoco.MjModel.from_xml_path(str(ROOT/self.config['model']))
        self.data=mujoco.MjData(self.model)
        mujoco.mj_forward(self.model,self.data)

    def test_two_named_cameras_and_legacy_preserved(self):
        for name in [*self.config['cameras'].values(),'g1_01_head_camera']:
            self.assertGreaterEqual(mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_CAMERA,name),0)

    def test_ego_follows_torso_third_person_stays_fixed(self):
        ids=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_CAMERA,self.config['cameras'][key])for key in ('ego','third_person')]
        before=self.data.cam_xpos[ids].copy()
        yaw=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'g1_01_waist_yaw_joint')
        self.data.qpos[self.model.jnt_qposadr[yaw]]=.4
        mujoco.mj_forward(self.model,self.data)
        self.assertGreater(np.linalg.norm(self.data.cam_xpos[ids[0]]-before[0]),.01)
        np.testing.assert_array_equal(self.data.cam_xpos[ids[1]],before[1])

    def test_ten_touch_sensors_and_correct_parent_bodies(self):
        self.assertEqual(len(self.config['fingertip_touch']),10)
        for entry in self.config['fingertip_touch']:
            sensor=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SENSOR,entry['sensor'])
            site=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,entry['site'])
            body=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,entry['body'])
            self.assertEqual(self.model.sensor_type[sensor],mujoco.mjtSensor.mjSENS_TOUCH)
            self.assertEqual(self.model.sensor_dim[sensor],1)
            self.assertEqual(self.model.site_bodyid[site],body)

    def test_ready_state_has_no_invented_action_success_or_phase(self):
        state=get_setup_state(self.model,self.data,self.config)
        self.assertEqual(len(state['qpos']),69)
        self.assertEqual(len(state['qvel']),69)
        self.assertEqual(len(state['ctrl']),29)
        self.assertIsNone(state['action_target'])
        self.assertIsNone(state['success'])
        self.assertEqual(state['phase'],'unknown')
        self.assertIsNone(state['object_poses']['coaster'])
        self.assertEqual(len(state['fingertip_external_geometric_contact_flags']),10)

    def test_camera_table_extrinsics_round_trip(self):
        profile=json.loads((ROOT/'config/g1_right_workspace.json').read_text())
        origin=np.array(profile['origin_world_m'])
        rotation=Rotation.from_euler('z',profile['yaw_degrees'],degrees=True).as_matrix()
        for name in self.config['cameras'].values():
            result=camera_metadata(self.model,self.data,name,256,256,origin,rotation)
            np.testing.assert_allclose(rotation@result['position_table_m']+origin,result['position_world_m'])
            np.testing.assert_allclose(rotation@np.array(result['R_table_camera_opengl']),result['R_world_camera_opengl'])
            self.assertGreater(result['K_opencv_pixels'][0][0],0)


if __name__=='__main__':unittest.main()
