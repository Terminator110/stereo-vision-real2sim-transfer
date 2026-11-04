import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from trajectory_io import Trajectory, load_xyz, sha256, write_xyz
from g1_workspace import CollisionChecker, export_world
from stereo.reconstruct_approach import annotation_interval, isolated_outliers, reconstruct_tag, select_tag, verify_source


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(max_reproj=3., max_speed=2., outlier_deviation=.04,
            max_missing_frames=3, max_gap_seconds=.5, smoothing_window=5,
            max_smoothing_displacement=.01, resample_hz=20.)

    def reconstruct(self, n=30, missing=(), step=.005):
        frames = np.arange(n)
        times = np.arange(n)*.1
        rows = {i: dict(valid='1', x_table_m=str(i*step), y_table_m='-.25', z_table_m='.08', reproj_px='1')
                for i in frames if i not in missing}
        return reconstruct_tag(frames, times, rows, self.args)

    def test_annotation_interval_is_read_not_hardcoded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'annotation.json'
            path.write_text(json.dumps([dict(data=dict(video='/data/upload/1/x.mp4'), annotations=[dict(result=[dict(value=dict(timelinelabels=['Approach'], ranges=[dict(start=7,end=19)]))])])]))
            self.assertEqual(annotation_interval(path)[:2], (7,19))

    def test_one_and_three_missing_fill_but_four_ten_eleven_do_not(self):
        for count in (1,3,4,10,11):
            with self.subTest(count=count):
                result = self.reconstruct(n=40, missing=range(10,10+count))
                self.assertEqual(int(result['interpolated'].sum()), count if count<=3 else 0)
                if count>3:
                    self.assertTrue(np.isnan(result['filtered'][10:10+count]).all())
                    self.assertEqual(len(result['segments']),2)

    def test_no_extrapolation_at_boundaries(self):
        result = self.reconstruct(missing=(0,1,28,29))
        self.assertTrue(np.isnan(result['filtered'][[0,1,28,29]]).all())

    def test_actual_time_interpolation_and_exact_final_endpoint(self):
        frames=np.arange(10);times=np.array([0,.1,.23,.3,.4,.5,.6,.7,.8,.913])
        rows={i:dict(valid='1',x_table_m=str(times[i]*.05),y_table_m='-.25',z_table_m='.08',reproj_px='1') for i in frames if i!=2}
        self.args.smoothing_window=0
        result=reconstruct_tag(frames,times,rows,self.args)
        self.assertAlmostEqual(result['filtered'][2,0],.23*.05)
        self.assertAlmostEqual(result['segments'][0]['times'][-1],.913)

    def test_isolated_spike_rejected_but_fast_consistent_motion_retained(self):
        t=np.arange(12)*.1;p=np.column_stack([t*.05,np.zeros(12),np.zeros(12)]);p[5,0]+=4
        mask=isolated_outliers(t,p,np.ones(12,dtype=bool),2.,.04)
        self.assertEqual(np.flatnonzero(mask).tolist(),[5])
        fast=np.column_stack([t*4,np.zeros(12),np.zeros(12)])
        self.assertFalse(isolated_outliers(t,fast,np.ones(12,dtype=bool),2.,.04).any())

    def test_smoothing_bounded_and_never_crosses_long_gap(self):
        result=self.reconstruct(n=40,missing=range(12,20))
        self.assertTrue(np.isnan(result['filtered'][12:20]).all())
        self.assertTrue(all(s['max_smoothing_displacement_m']<=.01 for s in result['segments']))

    def test_tag_selection_checks_support_instead_of_assuming_tag2(self):
        results={2:self.reconstruct(missing=range(10,30)),4:self.reconstruct(missing=range(30)),5:self.reconstruct()}
        self.assertEqual(select_tag(results)[0],5)

    def test_video_mismatch_fails_before_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            take=Path(directory);(take/'label_studio.mp4').write_bytes(b'wrong');upload=take/'upload.mp4';upload.write_bytes(b'actual')
            with self.assertRaises(ValueError):
                verify_source(take,'/data/upload/1/x.mp4',0,3,3,upload)


class ReplayTests(unittest.TestCase):
    def test_csv_rejects_bad_timestamps_nan_and_mixed_tags(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trajectory.csv'
            for content in ('time_s,side,x,y,z\n0,right,0,0,0\n0,right,1,0,0\n',
                            'time_s,side,x,y,z\n0,right,0,0,0\n1,right,nan,0,0\n',
                            'time_s,side,x,y,z,tag_id\n0,right,0,0,0,2\n1,right,1,0,0,5\n'):
                path.write_text(content)
                with self.assertRaises(ValueError):load_xyz(path)

    def test_sampler_hides_large_gap_without_extrapolating(self):
        trajectory=Trajectory(np.array([0,.05,1.]),np.array([[0,0,0],[1,0,0],[2,0,0]]))
        self.assertIsNone(trajectory.sample(.5,.15))
        self.assertIsNone(trajectory.sample(-.1,.15))
        self.assertIsNone(trajectory.sample(1.1,.15))
        np.testing.assert_allclose(trajectory.sample(.025,.15),[.5,0,0])

    def test_shared_mapping_preserves_origin_z_lengths_times_and_calibration_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);source=directory/'source.csv';world=directory/'world.csv';profile=directory/'profile.json'
            t=np.array([0,.05,.1]);p=np.array([[0,0,0],[.02,-.03,.1],[.04,.01,.2]])
            write_xyz(source,t,p)
            source.with_suffix('.json').write_text(json.dumps(dict(coordinate_frame='calibrated_table_v3',csv_sha256=sha256(source),table_calibration_sha256='frameA')))
            config=dict(mapping_mode='shared_manipulation_workspace',scale=1,source_frame='calibrated_table_v3',origin_world_m=[.39,-.12,.762],yaw_degrees=-60,source_table_calibration_sha256='frameA')
            profile.write_text(json.dumps(config));export_world(source,profile,world);out=load_xyz(world)
            np.testing.assert_array_equal(out.times,t)
            np.testing.assert_allclose(out.points[0],[.39,-.12,.762])
            np.testing.assert_allclose(out.points[:,2],.762+p[:,2])
            np.testing.assert_allclose(np.linalg.norm(np.diff(out.points,axis=0),axis=1),np.linalg.norm(np.diff(p,axis=0),axis=1))
            config['source_table_calibration_sha256']='frameB';profile.write_text(json.dumps(config))
            with self.assertRaises(ValueError):export_world(source,profile,world)

    def test_collision_checker_includes_visual_only_cup(self):
        m=mujoco.MjModel.from_xml_path(str(ROOT/'models/scenario_g1_dex5_stationary.xml'));checker=CollisionChecker(m)
        self.assertEqual(checker.check(m.qpos0),[])
        cup=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'table_cup');m.body_pos[cup]=[.25,-.15,.84]
        self.assertTrue(checker.check(m.qpos0))


if __name__=='__main__':
    unittest.main()
