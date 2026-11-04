import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from tracking_quality import fill_short_gaps, runs, summarize


class QualityTests(unittest.TestCase):
    def test_runs(self):
        self.assertEqual(runs([True, True, False, True]), [(0,2),(3,4)])
        self.assertEqual(runs([]), [])

    def test_only_short_internal_gaps(self):
        detected = np.array([False,True,False,True,False,False,False,True,False])
        xyz = np.full((9,21,3), np.nan, dtype=np.float32)
        xyz[detected] = np.arange(9)[detected,None,None]
        filled, world, mask = fill_short_gaps(xyz, xyz, detected, np.arange(9)/10, 0.15)
        self.assertEqual(np.flatnonzero(mask).tolist(), [2])
        self.assertTrue(np.allclose(filled[2], 2))
        self.assertTrue(np.isnan(filled[[0,4,5,6,8]]).all())
        self.assertTrue(np.isnan(xyz[2]).all())

    def test_world_not_invented(self):
        xyz = np.ones((3,21,3)); xyz[1] = np.nan
        world = np.full_like(xyz, np.nan)
        _, result, mask = fill_short_gaps(xyz, world, [True,False,True], np.arange(3)/30)
        self.assertTrue(mask[1])
        self.assertTrue(np.isnan(result).all())

    def test_no_detections(self):
        data = dict(landmarks=np.full((3,21,3),np.nan), timestamps=np.arange(3)/30,
                    detected=np.zeros(3,dtype=bool), handedness=np.array(['Unknown']*3), fps=30)
        report = summarize(data)
        self.assertIsNone(report['detected_span_percent'])
        self.assertEqual(report['missing'], 3)


if __name__ == '__main__':
    unittest.main()
