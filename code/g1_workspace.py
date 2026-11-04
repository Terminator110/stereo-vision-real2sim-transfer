"""Shared v3 manipulation frame and discrete MuJoCo arm collision checks."""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from trajectory_io import load_xyz, sha256, write_xyz


class CollisionChecker:
    def __init__(self, model, clearance=.002, include_waist_yaw=False):
        self.model, self.clearance = model, float(clearance)
        self.data = mujoco.MjData(model)
        self.include_waist_yaw = bool(include_waist_yaw)
        root_name = 'g1_01_waist_yaw_link' if self.include_waist_yaw else 'g1_01_right_shoulder_pitch_link'
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
        if root < 0:
            raise ValueError('Missing right shoulder body')
        bodies = {root}
        for body in range(root+1, model.nbody):
            if int(model.body_parentid[body]) in bodies:
                bodies.add(body)
        active = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]
                  or mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) == 'table_cup_visual']
        self.pairs = []
        for i, a in enumerate(active):
            ba = int(model.geom_bodyid[a])
            for b in active[i+1:]:
                bb = int(model.geom_bodyid[b])
                if ba not in bodies and bb not in bodies:
                    continue
                if ba == bb or model.body_weldid[ba] == model.body_weldid[bb]:
                    continue
                if int(model.body_parentid[ba]) == bb or int(model.body_parentid[bb]) == ba:
                    continue
                self.pairs.append((a, b))
        self.baseline = {}
        self.data.qpos[:] = model.qpos0
        mujoco.mj_kinematics(model, self.data)
        for pair in self.pairs:
            distance = self.distance(*pair)
            if distance < -.0005:
                names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or '' for g in pair]
                sides = ('right', 'left') if self.include_waist_yaw else ('right',)
                known_hand_mount = any(all(f'dex5_{side}' in name or f'{side}_wrist' in name for name in names)
                                       for side in sides)
                if not known_hand_mount:
                    raise ValueError(f'Unresolved non-hand baseline penetration: {names}, {distance:.4f} m')
                self.baseline[pair] = distance

    def distance(self, a, b):
        m, d = self.model, self.data
        if m.geom_type[a] != mujoco.mjtGeom.mjGEOM_PLANE and m.geom_type[b] != mujoco.mjtGeom.mjGEOM_PLANE:
            lower_bound = np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b])-m.geom_rbound[a]-m.geom_rbound[b]
            if lower_bound > self.clearance:
                return self.clearance
        return float(mujoco.mj_geomDistance(m, d, a, b, self.clearance, None))

    def check(self, qpos):
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        failures = []
        for pair in self.pairs:
            distance = self.distance(*pair)
            limit = self.baseline[pair]-.001 if pair in self.baseline else self.clearance-1e-6
            if distance < limit:
                names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
                         or mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[g])) for g in pair]
                failures.append(dict(geoms=names, distance_m=distance, minimum_m=limit))
        return failures

    def transition(self, start, end, addresses, max_step=.025):
        count = max(1, int(np.ceil(np.max(np.abs(end[addresses]-start[addresses]))/max_step)))
        for weight in np.linspace(0, 1, count+1)[1:]:
            failures = self.check(start+(end-start)*weight)
            if failures:
                return failures
        return []

    def description(self):
        return dict(clearance_m=self.clearance, checked_geom_pairs=len(self.pairs),
                    includes_waist_yaw=self.include_waist_yaw,
                    moving_scope='Entire upper body including both arms' if self.include_waist_yaw else 'Right arm',
                    baseline_hand_mount_overlaps=len(self.baseline),
                    baseline_policy='Existing Dex5/wrist assembly overlaps may not deepen by more than 1 mm; other collisions require 2 mm clearance',
                    limitations='Discrete convex-geom distances; adjacent/welded links excluded. Cup convex hull included conservatively despite visual-only contacts. Not exact CAD clearance or dynamics.')


def load_workspace(path):
    profile = json.loads(Path(path).read_text())
    if (profile.get('mapping_mode') != 'shared_manipulation_workspace'
            or profile.get('scale') != 1 or profile.get('source_frame') != 'calibrated_table_v3'):
        raise ValueError('Expected an unscaled shared v3 workspace profile')
    origin = np.asarray(profile['origin_world_m'], dtype=float)
    yaw = float(profile['yaw_degrees'])
    if origin.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite(yaw):
        raise ValueError('Invalid workspace origin/yaw')
    return Rotation.from_euler('z', yaw, degrees=True).as_matrix(), origin, profile


def export_world(source, workspace, destination):
    trajectory = load_xyz(source)
    meta_path = Path(source).with_suffix('.json')
    metadata = json.loads(meta_path.read_text())
    if metadata.get('coordinate_frame') != 'calibrated_table_v3' or metadata.get('csv_sha256') != sha256(source):
        raise ValueError('Source trajectory provenance/frame does not match the reconstructed CSV')
    rotation, origin, profile = load_workspace(workspace)
    if not metadata.get('table_calibration_sha256') or metadata['table_calibration_sha256'] != profile.get('source_table_calibration_sha256'):
        raise ValueError('Recording uses a different v3 calibration origin/frame; do not reuse this workspace silently')
    write_xyz(destination, trajectory.times, trajectory.points @ rotation.T+origin)
    metadata.update(coordinate_frame='mujoco_world', mapping_mode='shared_manipulation_workspace',
                    csv_sha256=sha256(destination), workspace_profile=str(Path(workspace).resolve()),
                    workspace_sha256=sha256(workspace), source_table_csv=str(Path(source).resolve()),
                    source_table_sha256=sha256(source), origin_world_m=origin.tolist(),
                    yaw_degrees=profile['yaw_degrees'], scale=1, hand_base_offset_m=[0, 0, 0])
    Path(destination).with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata
