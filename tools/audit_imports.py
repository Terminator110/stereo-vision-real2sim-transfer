#!/usr/bin/env python3
"""List third-party imports in the active workflow, including dynamic entry files.

Run this in g1_lerobot to identify dependencies missing from requirements.txt.
It is static analysis: optional imports and native executables require manual review.
"""
from __future__ import annotations
import ast
import importlib.metadata as md
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
FILES = [
    'dex5_batch_pipeline_v2.py', 'dex5_sam3_video_masks.py',
    'annotate_phases.py', 'dex5_video_to_sim_side_by_side.py',
    'dex5_video_to_sim_wrist_smoothed.py', 'dex5_physics_replay.py',
    '07_retarget_dex5_right.py', '08_live_webcam_dex5.py',
    '10_g1_dex5_stationary.py', '12_stereo_g1_dex5.py',
    'thumb_retarget.py', 'stereo/record_three_cameras.py',
    'stereo/stereo_wrist_test.py', 'stereo/select_and_reconstruct.py',
]
MAP = {'cv2':'opencv-python', 'PIL':'Pillow', 'yaml':'PyYAML',
       'sklearn':'scikit-learn', 'skimage':'scikit-image',
       'huggingface_hub':'huggingface-hub'}
local = {p.stem for p in CODE.glob('*.py')} | {p.stem for p in (CODE/'stereo').glob('*.py')}
roots = set()
missing_files=[]
for name in FILES:
    p = CODE/name
    if not p.is_file():
        missing_files.append(name)
        continue
    try:
        tree = ast.parse(p.read_text())
    except (UnicodeError,SyntaxError) as e:
        print(f"PARSE ERROR: {name}: {e}")
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split('.')[0])

print('ACTIVE WORKFLOW IMPORT AUDIT')
print('Python:', sys.executable)
print('Missing active source files:', missing_files or 'none')
print('\nTHIRD-PARTY / UNKNOWN:')
unknown=[]
for name in sorted(roots):
    if name in local or name in sys.stdlib_module_names or name.startswith('_'):
        continue
    dist = MAP.get(name)
    if not dist:
        dist_names = md.packages_distributions().get(name) or []
        dist = dist_names[0] if dist_names else name
    try:
        print(f'  {name:24s} {dist}=={md.version(dist)}')
    except md.PackageNotFoundError:
        unknown.append((name, dist))
        print(f'  {name:24s} UNKNOWN / NOT INSTALLED: {dist}')
print('\nUnresolved imports (inspect before finalizing requirements):', unknown or 'none')
print('Caveat: package dependencies loaded via binaries, plugins, or FoundationStereo scripts are not resolved by this audit.')
