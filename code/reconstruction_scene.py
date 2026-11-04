"""Per-take static object placement using a compiled copy of the authoritative MJCF."""
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ORIGIN=np.array([.385,-.120,.832])
ROTATION=Rotation.from_euler('z',-90,degrees=True).as_matrix()


def initial_cluster(rows,obj,tolerance=.01):
    key='cup_valid_tabletop' if obj=='cup' else 'coaster_valid'
    valid=[]
    for row in rows:
        if row.get(key) not in ('1','True','true'):continue
        try:p=np.array([float(row[f'{obj}_x_table_m']),float(row[f'{obj}_y_table_m'])])
        except (KeyError,ValueError):continue
        if np.isfinite(p).all():valid.append((int(row['frame']),p))
    valid.sort(key=lambda item:item[0])
    if not valid:return None
    selected=[]
    for frame,point in valid:
        if selected and np.linalg.norm(point-np.median([p for _,p in selected],axis=0))>tolerance:break
        selected.append((frame,point))
    values=np.array([p for _,p in selected]);median=np.median(values,axis=0)
    return dict(table_xy_m=median.tolist(),frames=[i for i,_ in selected],
                maximum_spread_m=float(np.linalg.norm(values-median,axis=1).max()),
                support=len(selected),single_keyframe_only=len(selected)==1,
                selection='Earliest valid stable cluster; later displaced estimates excluded')


def mesh_in_body(model,data,geom):
    mesh=int(model.geom_dataid[geom]);start=int(model.mesh_vertadr[mesh]);n=int(model.mesh_vertnum[mesh])
    vertices=model.mesh_vert[start:start+n]
    body=int(model.geom_bodyid[geom])
    return (vertices@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]-data.xpos[body])@data.xmat[body].reshape(3,3)


def make_scenario(source,objects_csv,destination,coaster_size_mm=60.,coaster_thickness_mm=2.):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    model=mujoco.MjModel.from_xml_path(str(source));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    table=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'work_table_top')
    height=float(data.geom_xpos[table,2]+model.geom_size[table,2])
    if not np.isclose(height,ORIGIN[2],atol=1e-6):raise ValueError('Current table height differs from validated 0.832 m transform')
    cup=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'table_cup_visual')
    vertices=mesh_in_body(model,data,cup)
    dimensions=dict(height_m=float(np.ptp(vertices[:,2])),body_diameter_m=float(np.ptp(vertices[:,1])),
                    bottom_local_z_m=float(vertices[:,2].min()),handle_excluded_from_diameter=True)
    if abs(dimensions['height_m']-.098)>.001 or abs(dimensions['body_diameter_m']-.082)>.001:
        raise ValueError(f'Cup geometry no longer matches 41 mm radius/98 mm height: {dimensions}')
    with Path(objects_csv).open() as handle:rows=list(csv.DictReader(handle))
    objects={name:initial_cluster(rows,name) for name in ('cup','coaster')}
    if not all(objects.values()):
        raise ValueError('Cup and coaster each require at least one valid tabletop localization; missing positions are not guessed')
    if not np.isfinite([coaster_size_mm,coaster_thickness_mm]).all() or min(coaster_size_mm,coaster_thickness_mm)<=0:
        raise ValueError('Coaster dimensions must be positive and finite')
    # Preserve original numeric strings: MjSpec.to_xml rounds camera transforms.
    root=ET.parse(source)
    scene_include=next(node for node in root.findall('include') if Path(node.get('file')).name=='table_and_cup.xml')
    scene_path=source.parent/scene_include.get('file')
    scene=ET.parse(scene_path)
    for node in root.findall('include'):node.set('file',str((source.parent/node.get('file')).resolve()))
    compiler=root.find('compiler');compiler.set('meshdir',str(source.parent));compiler.set('texturedir',str(source.parent))
    if objects['cup']:
        p=ROTATION@np.r_[objects['cup']['table_xy_m'],0]+ORIGIN
        body=scene.find('.//body[@name="table_cup"]')
        rotation=Rotation.from_quat(np.fromstring(body.get('quat','1 0 0 0'),sep=' ')[[1,2,3,0]]).as_matrix()
        position=p-rotation@np.array([0,0,dimensions['bottom_local_z_m']])
        body.set('pos',' '.join(repr(float(v))for v in position))
        objects['cup']['body_world_m']=position.tolist()
    if objects['coaster']:
        p=ROTATION@np.r_[objects['coaster']['table_xy_m'],0]+ORIGIN
        existing=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b)for b in range(model.nbody)
                  if 'coaster' in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b)or'')]
        if existing:raise ValueError('Existing coaster model found; add its measured bottom offset explicitly rather than guessing')
        thickness=coaster_thickness_mm/1000;size=coaster_size_mm/1000
        position=p+[0,0,thickness/2]
        body=ET.SubElement(scene.find('worldbody'),'body',name='recorded_coaster',pos=' '.join(map(str,position)))
        ET.SubElement(body,'geom',name='recorded_coaster_visual',type='box',
                      size=f'{size/2} {size/2} {thickness/2}',rgba='.16 .62 .7 1',contype='0',conaffinity='0',density='0')
        objects['coaster'].update(body_world_m=position.tolist(),size_m=[size,size,thickness],
                                  geometry_note='User requested 60x60 mm square. Visual thickness defaults to 2 mm, not measured.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    scene_copy=destination.with_name('recorded_scene_objects.xml')
    ET.indent(scene,space='  ');scene.write(scene_copy,encoding='unicode')
    scene_include.set('file',str(scene_copy))
    ET.indent(root,space='  ');root.write(destination,encoding='unicode')
    verified=mujoco.MjModel.from_xml_path(str(destination));d=mujoco.MjData(verified);mujoco.mj_forward(verified,d)
    for name in ('g1_01_pelvis','work_table','stereo_camera1_frame','stereo_camera2_frame'):
        a=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name);b=mujoco.mj_name2id(verified,mujoco.mjtObj.mjOBJ_BODY,name)
        np.testing.assert_allclose(data.xpos[a],d.xpos[b],atol=1e-9)
    report=dict(transform_origin_world_m=ORIGIN.tolist(),yaw_degrees=-90,cup_dimensions=dimensions,
                objects=objects,static_initial_placement_only=True,source_model=str(source),scenario=str(destination),
                warning='Kinematic replay and static object estimates do not prove grasp/contact success. Mask-fit residual is not ground-truth placement accuracy.')
    destination.with_suffix('.objects.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
