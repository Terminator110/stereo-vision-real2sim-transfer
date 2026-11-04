#!/usr/bin/env python3
"""Interactive right-wrist IK reference picker. No physics, IK execution or XML edits."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import glfw
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class WristPicker:
    def __init__(self,model_path,parent_body,output,step_mm=1.):
        self.model_path=Path(model_path).resolve()
        self.model=mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data=mujoco.MjData(self.model)
        mujoco.mj_forward(self.model,self.data)
        self.parent=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,parent_body)
        if self.parent<0:raise ValueError(f'Unknown reference body: {parent_body}')
        self.parent_name=parent_body
        self.wrist=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'g1_01_right_wrist_yaw_joint')
        self.palm=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'g1_01_dex5_right_frame')
        if self.wrist<0 or self.palm<0:raise ValueError('Missing right wrist/palm reference')
        self.output=Path(output).resolve()
        self.step=float(step_mm)/1000
        self.local=self.to_local(self.data.xanchor[self.wrist])
        self.hit_body=None;self.hit_geom=None
        self.status='Select a point on the right wrist/hand; W = wrist centre, P = palm origin.'
        self.ghost=False
        self.options=mujoco.MjvOption();self.options.sitegroup[:]=0
        self.camera=mujoco.MjvCamera()
        self.camera.lookat[:]=self.data.xanchor[self.wrist]
        self.camera.distance=.48;self.camera.azimuth=135;self.camera.elevation=-28
        self.robot_geoms=np.array([g for g in range(self.model.ngeom)
                                  if (mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,int(self.model.geom_bodyid[g]))or'').startswith('g1_01')],int)
        self.model.geom_rgba[self.robot_geoms,3]=1

    def to_local(self,world):
        rotation=self.data.xmat[self.parent].reshape(3,3)
        return rotation.T@(np.asarray(world)-self.data.xpos[self.parent])

    def world(self):
        return self.data.xpos[self.parent]+self.data.xmat[self.parent].reshape(3,3)@self.local

    def nudge(self,axis,amount):
        self.local[axis]+=amount
        self.status='Adjusted in reference-body local axes.'

    def pick(self,relx,rely,scene,aspect):
        point=np.zeros(3,np.float64)
        geom=np.full(1,-1,np.int32);flex=geom.copy();skin=geom.copy()
        body=mujoco.mjv_select(self.model,self.data,self.options,aspect,float(relx),float(rely),
                              scene,point,geom,flex,skin)
        name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,body) if body>=0 else ''
        if not name or not name.startswith(('g1_01_right_wrist_','g1_01_dex5_right_')):
            self.status='No right wrist/hand hit. Orbit/zoom and click its visible surface.'
            return False
        self.local=self.to_local(point)
        self.hit_body=name
        self.hit_geom=int(geom[0])
        self.status=f'Picked {name}; adjust with local-axis keys, then Enter to save.'
        return True

    def save(self):
        xml='<site name="custom_right_wrist_ik" pos="'+' '.join(f'{v:.9f}'for v in self.local)+'" quat="1 0 0 0" size="0.008" rgba="1 1 0 1" />'
        result=dict(model_path=str(self.model_path),parent_body=self.parent_name,
                    position_local_m=self.local.tolist(),quaternion_local_wxyz=[1,0,0,0],
                    position_world_at_pick_m=self.world().tolist(),selected_surface_body=self.hit_body,
                    selected_geom_id=self.hit_geom,robot_qpos_at_pick=self.data.qpos.tolist(),
                    xml_site_snippet=xml,
                    note='Reference orientation follows parent body. This is an IK reference, not a relocated physical joint or calibrated human wristband offset.')
        self.output.parent.mkdir(parents=True,exist_ok=True)
        if self.output.exists():
            stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.output.replace(self.output.with_name(self.output.name+'.backup_'+stamp))
        self.output.write_text(json.dumps(result,indent=2)+'\n')
        self.status=f'Saved {self.output.name}; existing IK scripts/XML were not changed.'
        print('Saved:',self.output,flush=True)
        print('Parent:',self.parent_name,'local XYZ (m):',self.local.tolist(),flush=True)
        print(xml,flush=True)
        return result

    def add_markers(self,scene):
        def sphere(position,radius,color):
            if scene.ngeom>=scene.maxgeom:return
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,
                               np.full(3,radius),position,np.eye(3).ravel(),np.array(color))
            scene.ngeom+=1
        sphere(self.data.xanchor[self.wrist],.004,[1,1,1,1])
        sphere(self.data.site_xpos[self.palm],.004,[0,.9,1,1])
        sphere(self.world(),.0065,[1,.85,0,1])
        rotation=self.data.xmat[self.parent].reshape(3,3)
        for axis,color in enumerate(([1,.1,.1,1],[.1,1,.1,1],[.1,.3,1,1])):
            if scene.ngeom>=scene.maxgeom:break
            geom=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.zeros(3),self.world(),np.eye(3).ravel(),np.array(color))
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.0015,self.world(),self.world()+rotation[:,axis]*.035)
            scene.ngeom+=1


def run(picker,check_path=None):
    if not glfw.init():raise RuntimeError('Cannot initialize GLFW; use a desktop session with DISPLAY set')
    window=None;context=None
    try:
        if check_path is not None:glfw.window_hint(glfw.VISIBLE,glfw.FALSE)
        window=glfw.create_window(1100,800,'G1 wrist IK reference picker',None,None)
        if not window:raise RuntimeError('Cannot create GLFW window')
        glfw.make_context_current(window);glfw.swap_interval(1)
        context=mujoco.MjrContext(picker.model,mujoco.mjtFontScale.mjFONTSCALE_150)
        scene=mujoco.MjvScene(picker.model,maxgeom=10000)
        mouse=dict(last=None,press=None,dragged=False)

        def update_scene():
            mujoco.mjv_updateScene(picker.model,picker.data,picker.options,None,picker.camera,
                                   mujoco.mjtCatBit.mjCAT_ALL,scene)

        def cursor(window,x,y):
            previous=mouse['last'];mouse['last']=(x,y)
            if previous is None:return
            left=glfw.get_mouse_button(window,glfw.MOUSE_BUTTON_LEFT)==glfw.PRESS
            right=glfw.get_mouse_button(window,glfw.MOUSE_BUTTON_RIGHT)==glfw.PRESS
            if not left and not right:return
            if left and mouse['press'] is not None:
                if np.hypot(x-mouse['press'][0],y-mouse['press'][1])<4 and not mouse['dragged']:return
                mouse['dragged']=True
            _,height=glfw.get_window_size(window)
            action=mujoco.mjtMouse.mjMOUSE_MOVE_V if right else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            mujoco.mjv_moveCamera(picker.model,action,(x-previous[0])/max(height,1),
                                  (y-previous[1])/max(height,1),scene,picker.camera)

        def button(window,button,action,mods):
            if button!=glfw.MOUSE_BUTTON_LEFT:return
            x,y=glfw.get_cursor_pos(window)
            if action==glfw.PRESS:
                mouse.update(press=(x,y),last=(x,y),dragged=False)
            elif action==glfw.RELEASE:
                if not mouse['dragged']:
                    w,h=glfw.get_window_size(window);fw,fh=glfw.get_framebuffer_size(window)
                    if min(w,h,fw,fh)>0:
                        update_scene()
                        picker.pick(x/w,1-y/h,scene,fw/fh)
                mouse['press']=None

        def scroll(window,xoffset,yoffset):
            mujoco.mjv_moveCamera(picker.model,mujoco.mjtMouse.mjMOUSE_ZOOM,0,-.05*yoffset,scene,picker.camera)

        def key(window,key,scancode,action,mods):
            if action not in (glfw.PRESS,glfw.REPEAT):return
            directions={glfw.KEY_LEFT:(0,-1),glfw.KEY_RIGHT:(0,1),glfw.KEY_UP:(1,1),
                        glfw.KEY_DOWN:(1,-1),glfw.KEY_PAGE_UP:(2,1),glfw.KEY_PAGE_DOWN:(2,-1)}
            if key in directions:
                axis,sign=directions[key];picker.nudge(axis,sign*picker.step*(10 if mods&glfw.MOD_SHIFT else 1))
            elif key in (glfw.KEY_ESCAPE,glfw.KEY_Q):glfw.set_window_should_close(window,True)
            elif key==glfw.KEY_W:
                picker.local=picker.to_local(picker.data.xanchor[picker.wrist]);picker.status='Wrist-yaw joint centre selected.'
            elif key==glfw.KEY_P:
                picker.local=picker.to_local(picker.data.site_xpos[picker.palm]);picker.status='Original Dex5 palm site selected.'
            elif key==glfw.KEY_T:
                picker.ghost=not picker.ghost
                picker.model.geom_rgba[picker.robot_geoms,3]=.25 if picker.ghost else 1
            elif key==glfw.KEY_F:
                picker.camera.lookat[:]=picker.world();picker.camera.distance=.35
            elif key in (glfw.KEY_ENTER,glfw.KEY_KP_ENTER,glfw.KEY_S) and action==glfw.PRESS:
                try:picker.save()
                except OSError as error:picker.status=f'Save failed: {error}'

        glfw.set_cursor_pos_callback(window,cursor);glfw.set_mouse_button_callback(window,button)
        glfw.set_scroll_callback(window,scroll);glfw.set_key_callback(window,key)
        original=picker.data.qpos.copy()
        while not glfw.window_should_close(window):
            width,height=glfw.get_framebuffer_size(window)
            if min(width,height)<=0:glfw.poll_events();time.sleep(.05);continue
            viewport=mujoco.MjrRect(0,0,width,height)
            update_scene()
            if check_path is not None:
                if not picker.pick(.5,.5,scene,width/height):raise RuntimeError('Centre selection missed the right wrist/hand')
            picker.add_markers(scene)
            mujoco.mjr_render(viewport,scene,context)
            controls='Click: pick | Left drag: orbit | Right drag: pan | Wheel: zoom\nW: wrist | P: palm | Arrows: local XY | PgUp/PgDn: local Z\nShift: 10x step | T: transparency | F: focus | Enter/S: save | Q/Esc: exit'
            local=picker.local*1000
            detail=f'Frame: {picker.parent_name}\nLocal XYZ (mm): {local[0]:+.2f}  {local[1]:+.2f}  {local[2]:+.2f}\nYellow: selected  White: wrist centre  Cyan: original palm site\n{picker.status}'
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,viewport,controls,'',context)
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,viewport,detail,'',context)
            if check_path is not None:
                from PIL import Image
                pixels=np.zeros((height,width,3),np.uint8)
                mujoco.mjr_readPixels(pixels,None,viewport,context)
                Image.fromarray(np.flipud(pixels)).save(check_path)
                np.testing.assert_array_equal(picker.data.qpos,original)
                print('Hidden-window render and centre-click selection verified:',check_path,flush=True)
                break
            glfw.swap_buffers(window);glfw.poll_events()
    finally:
        if context is not None:context.free()
        if window is not None:glfw.destroy_window(window)
        glfw.terminate()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=ROOT/'models/scenario_g1_dex5_stationary.xml')
    parser.add_argument('--frame',default='g1_01_right_wrist_yaw_link',help='Parent body for the fixed local IK reference')
    parser.add_argument('--output',type=Path,default=ROOT/'config/wrist_ik_pick.json')
    parser.add_argument('--step-mm',type=float,default=1.)
    parser.add_argument('--check',type=Path,help='Verify a hidden GLFW window and centre pick; save screenshot and exit without saving a reference')
    args=parser.parse_args()
    if not np.isfinite(args.step_mm) or args.step_mm<=0:parser.error('--step-mm must be finite and positive')
    picker=WristPicker(args.model,args.frame,args.output,args.step_mm)
    run(picker,args.check)


if __name__=='__main__':main()
