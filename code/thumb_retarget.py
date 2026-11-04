"""Palm-relative, bounded right-Dex5 thumb IK; no model or wrist changes."""
from collections import deque
import mujoco
import numpy as np
from scipy.optimize import least_squares

THUMB_NAMES = ('Yaw_11R','Roll_12R','Pitch_13R','Pitch_14R')


def unit(vector):
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-6:
        raise ValueError('Degenerate thumb/palm geometry')
    return vector/length


def palm_basis(points):
    forward = unit(points[9]-points[0])
    across = points[5]-points[17]
    across = unit(across-forward*np.dot(across,forward))
    return np.column_stack([across,forward,unit(np.cross(across,forward))])


class ThumbRetargeter:
    def __init__(self,model,bindings):
        self.model = model
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = model.qpos0
        self.addresses = np.array([bindings[n][0] for n in THUMB_NAMES])
        self.bounds = np.array([bindings[n][1:] for n in THUMB_NAMES],dtype=float)
        self.joints = [int(np.flatnonzero(model.jnt_qposadr == a)[0]) for a in self.addresses]
        self.palm = int(model.body_parentid[model.jnt_bodyid[self.joints[0]]])
        self.distal = int(model.jnt_bodyid[self.joints[3]])
        mujoco.mj_forward(model,self.data)
        # The actual distal STL extends along local +X; use its terminal surface centre.
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == self.distal
                 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        if not geoms:
            raise ValueError('Thumb distal mesh missing')
        geom = geoms[0]
        mesh = int(model.geom_dataid[geom])
        start,count = int(model.mesh_vertadr[mesh]),int(model.mesh_vertnum[mesh])
        vertices = model.mesh_vert[start:start+count]
        world = vertices @ self.data.geom_xmat[geom].reshape(3,3).T+self.data.geom_xpos[geom]
        local = (world-self.data.xpos[self.distal]) @ self.data.xmat[self.distal].reshape(3,3)
        self.tip_offset = local[local[:,0] >= local[:,0].max()-0.002].mean(axis=0)
        robot_points = np.zeros((21,3))
        robot_points[0] = self.data.xpos[self.palm]
        for landmark,name in [(5,'Roll_21R'),(9,'Roll_31R'),(17,'Roll_51R')]:
            joint = int(np.flatnonzero(model.jnt_qposadr == bindings[name][0])[0])
            robot_points[landmark] = self.data.xanchor[joint]
        self.robot_basis = self.data.xmat[self.palm].reshape(3,3).T @ palm_basis(robot_points)
        self.previous = np.clip(np.zeros(4),self.bounds[:,0],self.bounds[:,1])
        chain = self.kinematics(self.previous)
        self.anchor = chain[0].copy()
        self.lengths = np.linalg.norm(np.diff(chain,axis=0),axis=1)
        self.pending = deque(maxlen=3)
        self.accepted = False

    def kinematics(self,q):
        self.data.qpos[self.addresses] = q
        mujoco.mj_kinematics(self.model,self.data)
        tip = self.data.xpos[self.distal]+self.data.xmat[self.distal].reshape(3,3) @ self.tip_offset
        chain = np.vstack([self.data.xanchor[self.joints[0]],self.data.xanchor[self.joints[2]],
                           self.data.xanchor[self.joints[3]],tip])
        return (chain-self.data.xpos[self.palm]) @ self.data.xmat[self.palm].reshape(3,3)

    def targets(self,points):
        if points.shape != (21,3) or not np.isfinite(points[[0,1,2,3,4,5,9,17]]).all():
            raise ValueError('Nonfinite world landmarks')
        basis = palm_basis(points)
        bones = np.diff(points[1:5],axis=0)
        directions = np.array([unit(bone) for bone in bones])
        # Preserve human bone directions in anatomical palm coordinates, with robot lengths.
        robot_directions = directions @ basis @ self.robot_basis.T
        chain = np.vstack([self.anchor,self.anchor+np.cumsum(robot_directions*self.lengths[:,None],axis=0)])
        return chain

    def solve(self,points):
        target = self.targets(points)
        length = float(self.lengths.sum())
        weights = np.array([0.7,1.0,1.8])[:,None]
        def residual(q):
            actual = self.kinematics(q)
            positions = ((actual[1:]-target[1:])*weights/length).ravel()
            direction = 0.2*(unit(actual[3]-actual[2])-unit(target[3]-target[2]))
            return np.r_[positions,direction,0.015*(q-self.previous)]
        lower,upper = self.bounds[:,0],self.bounds[:,1]
        seed = np.clip(self.previous,lower+0.005,upper-0.005)
        best = least_squares(residual,seed,bounds=(lower,upper),max_nfev=35,
                             ftol=1e-5,xtol=1e-5,gtol=1e-5)
        if not np.isfinite(best.x).all():
            raise ValueError('Thumb solver produced no finite solution')
        actual = self.kinematics(best.x)
        error_mm = float(np.linalg.norm(actual[-1]-target[-1])*1000)
        return best.x,error_mm

    def update(self,points):
        if points is None:
            self.pending.clear()
            return dict(zip(THUMB_NAMES,self.previous)),'LOST_HOLD',None
        try:
            candidate,error = self.solve(points)
        except ValueError:
            self.pending.clear()
            return dict(zip(THUMB_NAMES,self.previous)),'INVALID_HOLD',None
        # A large change must persist for three consistent samples before acceptance.
        # This rejects isolated estimate flips; it is not an occlusion-confidence score.
        if self.accepted and np.max(np.abs(candidate-self.previous)) > 0.7:
            if self.pending and np.max(np.abs(candidate-self.pending[-1])) > 0.2:
                self.pending.clear()
            self.pending.append(candidate)
            if len(self.pending) < 3:
                return dict(zip(THUMB_NAMES,self.previous)),'OUTLIER_HOLD',error
            candidate = np.median(np.stack(self.pending),axis=0)
        self.previous = candidate.copy()
        self.accepted = True
        self.pending.clear()
        return dict(zip(THUMB_NAMES,self.previous)),'LIMITED_FIT' if error > 15 else 'TRACKING',error
