import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
spec = importlib.util.spec_from_file_location('live_dex5',Path(__file__).resolve().parents[1]/'code/08_live_webcam_dex5.py')
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


class LiveTests(unittest.TestCase):
    def make_filter(self):
        names = [n for group in live.retarget.FINGERS.values() for n in group]
        model = SimpleNamespace(qpos0=np.zeros(20),nq=20)
        bindings = {n:(i,-0.1,1.0) for i,n in enumerate(names)}
        return live.LiveFilter(model,bindings)

    def test_bounded_movement(self):
        controller = self.make_filter()
        angles = {'Roll_12R':10.0,'Pitch_22R':10.0}
        for i in range(60):
            previous = controller.pose.copy()
            pose,_ = controller.update(angles,i/30)
            self.assertTrue(np.isfinite(pose).all())
            self.assertTrue(((pose>=-0.1)&(pose<=1)).all())
            self.assertLessEqual(abs(pose[1]-previous[1]),2/30+1e-9)
            self.assertLessEqual(abs(pose[5]-previous[5]),3/30+1e-9)

    def test_loss_and_reacquisition(self):
        controller = self.make_filter()
        for i in range(30):
            controller.update({'Pitch_22R':1.0},i/30)
        previous = controller.pose.copy()
        held,state = controller.update(None,1.0)
        self.assertIn('HOLD',state)
        np.testing.assert_array_equal(held,previous)
        pose,state = controller.update(None,2.0)
        self.assertIn('NEUTRAL',state)
        self.assertLess(pose[5],previous[5])
        previous = pose.copy()
        pose,state = controller.update({'Pitch_22R':1.0},2.0333333333)
        self.assertIn('TRACKING RIGHT',state)
        self.assertLessEqual(abs(pose[5]-previous[5]),0.1+1e-8)

    def test_no_detection_stays_neutral(self):
        controller = self.make_filter()
        pose,state = controller.update(None,0)
        np.testing.assert_array_equal(pose,np.zeros(20))
        self.assertIn('LOST',state)


if __name__ == '__main__':
    unittest.main()
