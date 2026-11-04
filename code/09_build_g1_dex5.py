"""Generate a separate fixed-base G1 + two Dex5 hands; no source models changed."""
import argparse
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def add_hand(root, robot, side, assets):
    letter = 'R' if side == 'right' else 'L'
    relative = Path('dex5_1')/f'Dex5-URDF-{letter}'
    urdf_path = root/'models'/relative/f'Dex5-URDF-{letter}.urdf'
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    with tempfile.TemporaryDirectory() as temp:
        compiled = Path(temp)/'hand.xml'
        mujoco.mj_saveLastXML(str(compiled),model)
        hand = ET.parse(compiled).getroot()
    prefix = f'g1_01_dex5_{side}_'
    names = {node.get('name'):prefix+node.get('name') for node in hand.iter() if node.get('name')}
    for node in hand.iter():
        for attribute,value in list(node.attrib.items()):
            if attribute in ('name','mesh','material','joint','body','class') and value in names:
                node.set(attribute,names[value])
    for mesh in hand.findall('./asset/mesh'):
        mesh.set('file',(relative/mesh.get('file')).as_posix())
        assets.append(mesh)
    wrist = robot.find(f'.//body[@name="g1_01_{side}_wrist_yaw_link"]')
    rubber = wrist.find(f'body[@name="g1_01_{side}_rubber_hand"]')
    if rubber is None:
        raise ValueError(f'Expected original {side} rubber hand')
    wrist.remove(rubber)
    # Original palm-up alignment: fingers +Y -> G1 wrist +X.
    quat = '0.7071067812 0 0 -0.7071067812' if side == 'right' else '0 0.7071067812 0.7071067812 0'
    palm = ET.SubElement(wrist,'body',name=prefix+'palm',pos='0.0415 0 0',quat=quat)
    urdf = ET.parse(urdf_path).getroot()
    base_name = 'base_link00' if side == 'right' else 'base_link00L'
    inertial = urdf.find(f'./link[@name="{base_name}"]/inertial')
    if inertial is None:
        raise ValueError(f'Missing {side} palm inertia')
    origin = inertial.find('origin')
    inertia = inertial.find('inertia')
    rotation = Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
    xx,yy,zz,xy,xz,yz = [float(inertia.get(k)) for k in ('ixx','iyy','izz','ixy','ixz','iyz')]
    tensor = rotation @ np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]]) @ rotation.T
    ET.SubElement(palm,'inertial',pos=origin.get('xyz','0 0 0'),
                  mass=inertial.find('mass').get('value'),
                  fullinertia=' '.join(map(str,[tensor[0,0],tensor[1,1],tensor[2,2],tensor[0,1],tensor[0,2],tensor[1,2]])))
    for child in list(hand.find('worldbody')):
        palm.append(child)
    ET.SubElement(palm,'site',name=prefix+'frame',size='0.004',rgba='0 0.7 0.3 1')


def build(root):
    models = root/'models'
    robot = ET.parse(models/'robots/unitree_g1_29dof/unitree_g1_29dof_g1_01.xml').getroot()
    pelvis = robot.find('./worldbody/body')
    base_joint = pelvis.find('joint[@name="g1_01_floating_base_joint"]')
    if base_joint is None:
        raise ValueError('Expected G1 floating-base joint')
    pelvis.remove(base_joint)
    assets = ET.Element('asset')
    robot.insert(0,assets)
    for side in ('left','right'):
        add_hand(root,robot,side,assets)
    output = models/'robots/g1_dex5/g1_dex5_stationary.xml'
    output.parent.mkdir(parents=True,exist_ok=True)
    ET.indent(robot,space='  ')
    ET.ElementTree(robot).write(output,encoding='unicode',xml_declaration=False)
    scene = ET.Element('mujoco',model='g1_dex5_stationary_test')
    ET.SubElement(scene,'include',file='robots/unitree_g1_29dof/unitree_g1_29dof_assets.xml')
    ET.SubElement(scene,'include',file='robots/g1_dex5/g1_dex5_stationary.xml')
    ET.SubElement(scene,'compiler',angle='radian',meshdir='.',texturedir='.')
    ET.SubElement(scene,'option',timestep='0.002',gravity='0 0 -9.81',integrator='implicitfast')
    ET.SubElement(scene,'statistic',center='0 0 0.8',extent='2')
    visual = ET.SubElement(scene,'visual')
    ET.SubElement(visual,'headlight',ambient='0.5 0.5 0.5',diffuse='0.8 0.8 0.8')
    ET.SubElement(visual,'global',offwidth='1280',offheight='720')
    world = ET.SubElement(scene,'worldbody')
    ET.SubElement(world,'light',pos='1 -2 3',dir='-0.2 0.2 -1')
    ET.SubElement(world,'geom',name='floor',type='plane',size='4 4 0.1',rgba='0.18 0.22 0.24 1')
    ET.indent(scene,space='  ')
    scene_path = models/'scenario_g1_dex5_stationary.xml'
    ET.ElementTree(scene).write(scene_path,encoding='unicode',xml_declaration=False)
    result = mujoco.MjModel.from_xml_path(str(scene_path))
    if result.nq != 69:
        raise ValueError(f'Expected 29 G1 + 40 Dex5 joints, got nq={result.nq}')
    print(f'Built {scene_path}: {result.nq} joints; fixed pelvis, no walking.')
    return scene_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    build(parser.parse_args().root.resolve())
