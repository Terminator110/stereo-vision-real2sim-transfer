import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np

spec = importlib.util.spec_from_file_location('retarget',Path(__file__).resolve().parents[1]/'code/07_retarget_dex5_right.py')
retarget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retarget)


def straight_hand():
    points = np.zeros((21,3))
    for chain,x in zip(retarget.CHAINS.values(),[-0.03,-0.01,0.01,0.03]):
        base = np.array([x,0.07,0])
        direction = base/np.linalg.norm(base)
        for i,index in enumerate(chain):
            points[index] = base+direction*i*0.025
    for i in range(1,5):
        points[i] = [-0.025*i,0.01*i,0]
    return points


class RetargetTests(unittest.TestCase):
    def test_shared_clock(self):
        times = np.arange(340)/30
        self.assertEqual(retarget.clock_frame(times,0),0)
        self.assertEqual(retarget.clock_frame(times,1),30)
        self.assertEqual(retarget.clock_frame(times,1,0.5),15)
        self.assertEqual(retarget.clock_frame(times,100),339)
        self.assertEqual(retarget.clock_frame(times,-1),0)

    def test_angles(self):
        self.assertAlmostEqual(retarget.bend(np.array([0,0,0]),np.array([1,0,0]),np.array([2,0,0])),0)
        self.assertAlmostEqual(retarget.bend(np.array([0,0,0]),np.array([1,0,0]),np.array([1,1,0])),np.pi/2)

    def test_straight_hand_neutral(self):
        angles = retarget.landmark_angles(straight_hand())
        self.assertEqual(len(angles),15)
        self.assertTrue(all(abs(v)<1e-6 for v in angles.values()))

    def test_rigid_transform_invariance(self):
        hand = straight_hand()
        hand[7] += [0,0,0.01]
        rotation = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        first = retarget.landmark_angles(hand)
        other = retarget.landmark_angles(hand@rotation.T+[3,-2,5])
        np.testing.assert_allclose(list(first.values()),list(other.values()),atol=1e-6)

    def test_degenerate_rejected(self):
        with self.assertRaises(ValueError):
            retarget.landmark_angles(np.zeros((21,3)))

    def test_limits_holds_and_provenance(self):
        names = [name for group in retarget.FINGERS.values() for name in group]
        bindings = {name:(i,0.0,0.25) for i,name in enumerate(names)}
        model = SimpleNamespace(qpos0=np.zeros(20),nq=20)
        hand = straight_hand()
        hand[7] += [0,0,0.03]
        points = np.array([np.full((21,3),np.nan),hand,np.full((21,3),np.nan),hand])
        eligible = np.array([False,True,False,True])
        detected = np.array([False,True,False,False])
        interpolated = np.array([False,False,False,True])
        trajectory,mapped,states,clipped = retarget.retarget(
            model,bindings,points,np.arange(4)/30,eligible,detected,interpolated,10,5)
        self.assertTrue(np.isfinite(trajectory).all())
        self.assertTrue(((trajectory>=0)&(trajectory<=0.25)).all())
        np.testing.assert_array_equal(trajectory[0],np.zeros(20))
        np.testing.assert_array_equal(trajectory[1],trajectory[2])
        self.assertEqual(states.tolist(),['missing_hold','observed','missing_hold','interpolated'])
        self.assertEqual(mapped.tolist(),eligible.tolist())
        self.assertGreater(clipped.sum(),0)


if __name__ == '__main__':
    unittest.main()
