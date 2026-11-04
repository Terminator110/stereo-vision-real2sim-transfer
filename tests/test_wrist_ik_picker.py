from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

ROOT=Path('/home/sparx3d/physikal_workflow/dex-5')
sys.path.insert(0,str(ROOT/'code'))
from pick_wrist_ik_site import WristPicker


class PickerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'pick.json'
        self.picker=WristPicker(ROOT/'models/scenario_g1_dex5_stationary.xml',
                                'g1_01_right_wrist_yaw_link',self.path)

    def test_initial_wrist_origin_and_round_trip(self):
        np.testing.assert_allclose(self.picker.local,[0,0,0],atol=1e-12)
        point=self.picker.data.xpos[self.picker.parent]+[.03,-.02,.01]
        self.picker.local=self.picker.to_local(point)
        np.testing.assert_allclose(self.picker.world(),point,atol=1e-12)

    def test_original_palm_local_offset(self):
        local=self.picker.to_local(self.picker.data.site_xpos[self.picker.palm])
        np.testing.assert_allclose(local,[.0415,0,0],atol=1e-10)

    def test_nudging_preserves_robot_pose(self):
        original=self.picker.data.qpos.copy()
        self.picker.nudge(0,.001);self.picker.nudge(2,-.010)
        np.testing.assert_allclose(self.picker.local,[.001,0,-.01],atol=1e-12)
        np.testing.assert_array_equal(original,self.picker.data.qpos)

    def test_save_json_and_backup_without_xml_changes(self):
        source=self.picker.model_path.read_bytes()
        first=self.picker.save()
        self.assertEqual(first['parent_body'],'g1_01_right_wrist_yaw_link')
        self.assertEqual(first['quaternion_local_wxyz'],[1,0,0,0])
        self.picker.nudge(1,.002);self.picker.save()
        self.assertEqual(len(list(self.path.parent.glob('pick.json.backup_*'))),1)
        self.assertAlmostEqual(json.loads(self.path.read_text())['position_local_m'][1],.002)
        self.assertEqual(self.picker.model_path.read_bytes(),source)

    def test_table_selection_is_rejected(self):
        body=mujoco.mj_name2id(self.picker.model,mujoco.mjtObj.mjOBJ_BODY,'work_table')
        original=self.picker.local.copy()
        with patch.object(mujoco,'mjv_select',return_value=body):
            self.assertFalse(self.picker.pick(.5,.5,None,1))
        np.testing.assert_array_equal(self.picker.local,original)

    def test_right_surface_selection_converts_to_parent_frame(self):
        model=self.picker.model
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'g1_01_dex5_right_palm')
        point=self.picker.data.xpos[body]+[.005,.003,.002]
        def select(*args):
            args[7][:]=point
            args[8][0]=0
            return body
        with patch.object(mujoco,'mjv_select',side_effect=select):
            self.assertTrue(self.picker.pick(.5,.5,None,1))
        np.testing.assert_allclose(self.picker.world(),point,atol=1e-12)


if __name__=='__main__':unittest.main()
