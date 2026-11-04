from pathlib import Path
import importlib.util
import json
import sys
import unittest

import mujoco
import numpy as np

ROOT=Path('/home/sparx3d/physikal_workflow/dex-5')
sys.path.insert(0,str(ROOT/'code'))
from g1_waist_video_placement import apply_robot_offset,geometry_signature


class WaistCopyTests(unittest.TestCase):
    def model(self):
        return mujoco.MjModel.from_xml_path(str(ROOT/'models/scenario_g1_dex5_stationary.xml'))

    def test_offset_changes_only_root_xy(self):
        model=self.model();before=model.body_pos.copy();q=model.qpos0.copy()
        apply_robot_offset(model,[-.09,-.10])
        b=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'g1_01_pelvis')
        np.testing.assert_allclose(model.body_pos[b]-before[b],[-.09,-.10,0])
        others=np.arange(model.nbody)!=b
        np.testing.assert_array_equal(model.body_pos[others],before[others])
        np.testing.assert_array_equal(model.qpos0,q)

    def test_signature_ignores_opacity_but_detects_geometry(self):
        model=self.model();sig=geometry_signature(model)
        model.geom_rgba[:,3]=.1
        self.assertEqual(geometry_signature(model),sig)
        apply_robot_offset(model,[.01,0])
        self.assertNotEqual(geometry_signature(model),sig)

    def test_only_yaw_added_to_arm_list(self):
        path=ROOT/'code/dex5_video_to_sim_waist_rest_start.py'
        spec=importlib.util.spec_from_file_location('waist_copy',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(len(module.ARM),8)
        self.assertEqual(module.ARM[-1],'g1_01_waist_yaw_joint')
        self.assertFalse(any('waist_roll' in name or 'waist_pitch' in name for name in module.ARM))

    def test_config_current_geometry_and_workspace(self):
        config=json.loads((ROOT/'config/g1_waist_video_placement.json').read_text())
        workspace=json.loads((ROOT/'config/g1_right_workspace.json').read_text())
        self.assertEqual(config['geometry_signature'],geometry_signature(self.model()))
        self.assertEqual(config['workspace_origin_world_m'],workspace['origin_world_m'])


if __name__=='__main__':unittest.main()
