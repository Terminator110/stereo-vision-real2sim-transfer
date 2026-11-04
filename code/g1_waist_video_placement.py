"""Choose an isolated G1 XY offset for the current workspace; no shared scene edits."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from g1_hand_ik import HandIK
from g1_workspace import CollisionChecker


def geometry_signature(model):
    digest = hashlib.sha256()
    for name in ('body_pos','body_quat','geom_pos','geom_quat','geom_size','geom_type',
                 'geom_contype','geom_conaffinity','jnt_range','jnt_axis','qpos0','mesh_vert','mesh_face'):
        digest.update(np.asarray(getattr(model,name)).tobytes())
    return digest.hexdigest()


def apply_robot_offset(model, xy):
    body = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'g1_01_pelvis')
    if body < 0:
        raise ValueError('Missing fixed G1 pelvis root')
    model.body_pos[body,:2] += np.asarray(xy,float)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs'/datetime.now().strftime('waist_video_placement_%Y%m%d_%H%M%S'))
    parser.add_argument('--config',type=Path,default=ROOT/'config/g1_waist_video_placement.json')
    args=parser.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    profile=json.loads((ROOT/'config/g1_right_workspace.json').read_text())
    origin=np.array(profile['origin_world_m']); assert profile['yaw_degrees']==-90
    path=ROOT/'models/scenario_g1_dex5_stationary.xml'
    original=mujoco.MjModel.from_xml_path(str(path)); signature=geometry_signature(original)
    scratch=mujoco.MjData(original); mujoco.mj_forward(original,scratch)
    table=mujoco.mj_name2id(original,mujoco.mjtObj.mjOBJ_GEOM,'work_table_top')
    near=float(scratch.geom_xpos[table,0]-original.geom_size[table,0])
    candidates=[]; rejected=[]
    torso_geoms=[g for g in range(original.ngeom) if original.geom_contype[g]
                 and any(k in (mujoco.mj_id2name(original,mujoco.mjtObj.mjOBJ_BODY,int(original.geom_bodyid[g])) or '')
                         for k in ('torso','waist','pelvis'))]
    yaw_values=np.radians(np.arange(-80,81,20))

    def prepare(xy):
        model=mujoco.MjModel.from_xml_path(str(path)); apply_robot_offset(model,xy)
        arm=HandIK(model,'right',include_waist_yaw=True)
        checker=CollisionChecker(model,include_waist_yaw=True)
        data=mujoco.MjData(model)
        joint=int(arm.ids[-2]); waist=int(arm.ids[-1])
        min_gap=float('inf'); sweep_ok=0
        for angle in yaw_values:
            data.qpos[:]=model.qpos0;data.qpos[arm.addresses[-1]]=angle
            mujoco.mj_forward(model,data)
            front=-float('inf')
            for geom in torso_geoms:
                mesh=int(model.geom_dataid[geom])
                if model.geom_type[geom]!=mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                start=int(model.mesh_vertadr[mesh]); count=int(model.mesh_vertnum[mesh])
                vertices=model.mesh_vert[start:start+count]
                xx=vertices@data.geom_xmat[geom].reshape(3,3)[0]+data.geom_xpos[geom,0]
                front=max(front,float(xx.max()))
            min_gap=min(min_gap,near-front)
            if not checker.check(data.qpos):sweep_ok+=1
        return model,arm,checker,data,joint,waist,min_gap,sweep_ok

    def sample(prepared,spacing):
        model,arm,checker,data,joint,waist,_,_=prepared
        ds=np.linspace(0,.5,int(round(.5/spacing))+1)
        rs=np.linspace(-.3,.3,int(round(.6/spacing))+1)
        hs=np.array([.05,.125,.20,.25]) if spacing==.1 else np.linspace(0,.25,6)
        rng=np.random.default_rng(20261103)
        seeds=[model.qpos0[arm.addresses].copy()]
        for angle in [-1.0,-.5,.5,1.0]:
            seed=seeds[0].copy();seed[-1]=angle;seed[3]=.8;seeds.append(seed)
        seeds += [rng.uniform(arm.lower*.7,arm.upper*.7)]
        grid=np.zeros((len(hs),len(ds),len(rs)),bool)
        poses=[];xyz=[];rows=[];warm=None
        mujoco.mj_forward(model,data)
        shoulder=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'g1_01_right_shoulder_pitch_link')
        sh=data.xpos[shoulder].copy();center=data.xanchor[waist].copy()
        radius=0.;b=int(model.jnt_bodyid[joint])
        while b!=shoulder:radius+=np.linalg.norm(model.body_pos[b]);b=int(model.body_parentid[b])
        circle=np.linalg.norm((sh-center)[:2])
        for di,depth in enumerate(ds):
            for ri,red in enumerate(rs):
                for hi,height in enumerate(hs):
                    point=np.array([near+depth,origin[1]-red,origin[2]+height])
                    status='OUTSIDE_YAW_BOUND'; error=None;accepted=None
                    bound=np.hypot(np.linalg.norm((point-center)[:2])-circle,point[2]-sh[2])
                    if bound<=radius+.002:
                        status='NO_ACCEPTED_IK'
                        def residual(q):
                            data.qpos[:]=model.qpos0;data.qpos[arm.addresses]=q
                            mujoco.mj_kinematics(model,data)
                            return 10*(data.xanchor[joint]-point)
                        for seed in ([warm] if warm is not None else [])+seeds:
                            fit=least_squares(residual,seed,bounds=(arm.lower,arm.upper),max_nfev=90,
                                              ftol=1e-7,xtol=1e-7,gtol=1e-7)
                            error=float(np.linalg.norm(residual(fit.x))/10)
                            if error>.002:continue
                            pose=model.qpos0.copy();pose[arm.addresses]=fit.x
                            if checker.check(pose):status='COLLISION_REJECTED';continue
                            accepted=pose;warm=fit.x.copy();status='REACHABLE';break
                    if accepted is not None:
                        grid[hi,di,ri]=True;poses.append(accepted);xyz.append(point)
                    rows.append(dict(depth_m=float(depth),red_m=float(red),height_m=float(height),
                                     world_m=point.tolist(),status=status,error_m=error))
        return dict(accepted=int(grid.sum()),total=int(grid.size),at_20cm=int(grid[list(hs).index(.20)].sum()),
                    max_depth_m=max((r['depth_m'] for r in rows if r['status']=='REACHABLE'),default=None)),grid,poses,xyz,rows

    for x in [-.06,-.09,-.12,-.15,-.18]:
        for y in [-.10,-.05,0,.05]:
            try:
                prepared=prepare([x,y]);gap,sweep=prepared[-2:]
            except ValueError as error:
                rejected.append(dict(xy=[x,y],reason=str(error)));continue
            if gap<.03 or sweep<len(yaw_values):
                rejected.append(dict(xy=[x,y],gap_m=gap,neutral_sweep_passes=sweep));continue
            candidates.append(dict(xy=[x,y],gap_m=gap,neutral_sweep_passes=sweep))
    # Rank eligible candidates geometrically, then test five with bounded IK.
    def rank(c):
        x,y=c['xy']
        return -abs(x)-.25*abs(y)
    shortlist=sorted(candidates,key=rank,reverse=True)[:5]
    if not shortlist:raise RuntimeError(f'No eligible 3 cm-clearance placement; details: {rejected}')
    results=[]
    for candidate in shortlist:
        stats,*_=sample(prepare(candidate['xy']),.1)
        candidate.update(stats);results.append(candidate)
        print('PLACEMENT',json.dumps(candidate),flush=True)
    best=max(results,key=lambda c:(c['accepted'],c['at_20cm'],c['gap_m']))
    prepared=prepare(best['xy'])
    stats,grid,poses,xyz,rows=sample(prepared,.05)
    cells=np.ones((5,10,12),bool)
    for h in (0,1):
        for d in (0,1):
            for r in (0,1):cells &= grid[h:h+5,d:d+10,r:r+12]
    report=dict(selected=best,dense=stats,grid_volume_estimate_litres=float(cells.sum()*.125),
                yaw_sweep_degrees=[-80,80],minimum_torso_edge_gap_m=.03,
                candidates_tested=results,eligible_candidates=candidates,rejected=rejected,
                waist_joint='g1_01_waist_yaw_joint',geometry_signature=signature,
                workspace_origin_world_m=origin.tolist(),workspace_yaw_degrees=-90.,
                constraints='3 cm torso edge gap and collision-free neutral yaw samples from -80 to +80 degrees',
                limitations='Finite search, static position-only wrist IK. Not a global optimum or certified swept-motion volume.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'samples.json').write_text(json.dumps(rows,indent=2)+'\n')
    np.savez_compressed(out/'reachable.npz',qpos=np.asarray(poses),xyz=np.asarray(xyz))
    config=dict(robot_offset_xy_world=best['xy'],geometry_signature=signature,
                workspace_origin_world_m=origin.tolist(),workspace_yaw_degrees=-90.,
                reachability_npz=str(out/'reachable.npz'),report=str(out/'report.json'))
    args.config.write_text(json.dumps(config,indent=2)+'\n')
    print('FINAL',json.dumps(report),flush=True)


if __name__=='__main__':main()
