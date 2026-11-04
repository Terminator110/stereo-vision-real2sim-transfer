"""Strict single-tag CSV input and explicit rigid table-frame transformations."""
from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path

import mujoco
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def proper_rotation(value):
    rotation = np.asarray(value, dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-7)):
        raise ValueError('Expected a proper rigid rotation (no scale/reflection)')
    return rotation


def scene_table_frame(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'work_table_top')
    if geom < 0 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError('Scene must contain the box work_table_top')
    axes = data.geom_xmat[geom].reshape(3, 3)
    dimensions = 2*model.geom_size[geom]
    if not np.allclose(np.sort(dimensions[:2]), [.62, .75], atol=1e-6):
        raise ValueError('Expected a 0.75 x 0.62 m tabletop')
    up = axes[:, 2]
    if np.dot(up, [0, 0, 1]) < .9999:
        raise ValueError('Expected an upward-facing horizontal tabletop')
    x = axes[:, int(np.argmax(dimensions[:2]))]
    rotation = proper_rotation(np.column_stack([x, np.cross(up, x), up]))
    centre = data.geom_xpos[geom]+up*model.geom_size[geom, 2]
    return rotation, centre


def write_xyz(path, times, points):
    with Path(path).open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['time_s', 'side', 'x', 'y', 'z'])
        for stamp, point in zip(times, points):
            writer.writerow([float(stamp), 'right', *map(float, point)])


@dataclass
class Trajectory:
    times: np.ndarray
    points: np.ndarray

    def gaps(self, maximum):
        return np.flatnonzero(np.diff(self.times) > maximum+1e-9).tolist()

    def sample(self, stamp, maximum_gap):
        if stamp < self.times[0]-1e-9 or stamp > self.times[-1]+1e-9:
            return None
        index = int(np.searchsorted(self.times, stamp, side='left'))
        if index < len(self.times) and abs(self.times[index]-stamp) < 1e-8:
            return self.points[index].copy()
        if index == 0 or index == len(self.times):
            return None
        dt = self.times[index]-self.times[index-1]
        if dt > maximum_gap+1e-9:
            return None
        weight = (stamp-self.times[index-1])/dt
        return self.points[index-1]*(1-weight)+self.points[index]*weight


def load_xyz(path):
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('Missing or duplicated CSV headers')
        if not {'time_s', 'side', 'x', 'y', 'z'} <= set(reader.fieldnames):
            raise ValueError('CSV requires time_s,side,x,y,z')
        rows = list(reader)
    if len(rows) < 2 or any(row['side'] != 'right' for row in rows):
        raise ValueError('Supply at least two samples for right arm only')
    if 'tag_id' in reader.fieldnames and len({row['tag_id'] for row in rows}) != 1:
        raise ValueError('Never combine positions from different AprilTags')
    times = np.array([float(row['time_s']) for row in rows])
    points = np.array([[float(row[key]) for key in ('x', 'y', 'z')] for row in rows])
    if not np.isfinite(times).all() or not np.isfinite(points).all() or times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError('Require finite XYZ and strictly increasing nonnegative timestamps')
    return Trajectory(times, points)


