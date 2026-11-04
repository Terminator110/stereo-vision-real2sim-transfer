"""Check/preview prepared MuJoCo MVP sensors. No recording, training or physics stepping."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]


def get_setup_state(model,data,config):
    """Read existing fields for a future recorder; do not invent actions or labels."""
    flags={};touch={}
    for entry in config['fingertip_touch']:
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,entry['body'])
        sensor=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SENSOR,entry['sensor'])
        touch[entry['sensor']]=float(data.sensordata[model.sensor_adr[sensor]])
        active=False
        for contact in data.contact[:data.ncon]:
            if contact.dist>0:continue
            a,b=(int(model.geom_bodyid[int(g)])for g in contact.geom)
            if a!=body and b!=body:continue
            other=b if a==body else a
            name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,other) or ''
            if not name.startswith('g1_01'):active=True
        flags[f"{entry['side']}_{entry['finger']}"]=active
    objects={}
    for label,name in config['objects'].items():
        if name is None:
            objects[label]=None;continue
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
        if body<0:raise ValueError(f'Missing object {name}')
        objects[label]=dict(position_world_m=data.xpos[body].tolist(),quaternion_wxyz=data.xquat[body].tolist())
    return dict(qpos=data.qpos.tolist(),qvel=data.qvel.tolist(),ctrl=data.ctrl.tolist(),
                fingertip_external_geometric_contact_flags=flags,touch_normal_force_N=touch,
                object_poses=objects,phase='unknown',success=None,
                action_target=None,note='No policy action, phase annotation or success result has been supplied.')


def camera_metadata(model,data,name,width,height,origin,table_rotation):
    camera=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,name)
    if camera<0:raise ValueError(f'Missing camera {name}')
    rotation=data.cam_xmat[camera].reshape(3,3)
    position=data.cam_xpos[camera]
    focal=height/(2*np.tan(np.radians(model.cam_fovy[camera])/2))
    return dict(name=name,resolution_wh=[width,height],fovy_degrees=float(model.cam_fovy[camera]),
                K_opencv_pixels=[[float(focal),0,width/2],[0,float(focal),height/2],[0,0,1]],
                position_world_m=position.tolist(),R_world_camera_opengl=rotation.tolist(),
                position_table_m=(table_rotation.T@(position-origin)).tolist(),
                R_table_camera_opengl=(table_rotation.T@rotation).tolist(),
                convention='MuJoCo/OpenGL camera: X right, Y up, -Z forward. OpenCV conversion diag(1,-1,-1).',
                calibration_note='Configured simulated pinhole camera, not a measured physical-camera calibration.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/mujoco_mvp_setup.json')
    parser.add_argument('--preview',action='store_true',help='Show both fixed-rest camera views; Q/Esc closes')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/mvp_setup_check')
    parser.add_argument('--keep-transparency',action='store_true',help='Retain current diagnostic robot transparency in camera preview')
    args=parser.parse_args()
    import cv2
    config=json.loads(args.config.read_text())
    model=mujoco.MjModel.from_xml_path(str(ROOT/config['model']))
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    if not args.keep_transparency:
        for g in range(model.ngeom):
            name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or ''
            if name.startswith('g1_01'):model.geom_rgba[g,3]=1
    width,height=config['rgb_resolution']
    options=mujoco.MjvOption();options.sitegroup[:]=0
    profile=json.loads((ROOT/'config/g1_right_workspace.json').read_text())
    origin=np.array(profile['origin_world_m'])
    table_rotation=Rotation.from_euler('z',profile['yaw_degrees'],degrees=True).as_matrix()
    args.output.mkdir(parents=True,exist_ok=True)
    metadata=dict(cameras={},joint_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j)for j in range(model.njnt)],
                  actuator_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,a)for a in range(model.nu)],
                  physics_timestep_s=float(model.opt.timestep),state=get_setup_state(model,data,config),
                  recorder_enabled=False,physics_stepping=False,depth_segmentation_export=False)
    with mujoco.Renderer(model,height,width) as renderer:
        images=[]
        for label,name in config['cameras'].items():
            renderer.update_scene(data,camera=name,scene_option=options)
            rgb=renderer.render().copy()
            if rgb.dtype!=np.uint8 or rgb.shape!=(height,width,3) or rgb.std()<2:
                raise ValueError(f'Blank or invalid camera render: {name}')
            if not cv2.imwrite(str(args.output/f'{label}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):
                raise IOError('Could not save camera verification image')
            images.append(rgb)
            metadata['cameras'][label]=camera_metadata(model,data,name,width,height,origin,table_rotation)
        (args.output/'setup_check.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print('Ready:',config['cameras'],'; touch zones:',len(config['fingertip_touch']),flush=True)
        print('Joint state sizes:',model.nq,model.nv,'; existing motor controls:',model.nu,flush=True)
        print('Verification only; no episode recorded. Images:',args.output,flush=True)
        if args.preview:
            view=cv2.cvtColor(np.concatenate(images,axis=1),cv2.COLOR_RGB2BGR)
            cv2.putText(view,'Ego RGB',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            cv2.putText(view,'Third-person RGB',(width+8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            try:
                while True:
                    cv2.imshow('MVP camera setup - Q to close',view)
                    if cv2.waitKey(30)&0xff in (ord('q'),ord('Q'),27):break
                    if cv2.getWindowProperty('MVP camera setup - Q to close',cv2.WND_PROP_VISIBLE)<1:break
                    time.sleep(.01)
            finally:cv2.destroyAllWindows()


if __name__=='__main__':main()
