#!/usr/bin/env python3
"""Safely add an explicit trajectory input to the existing DEX5 physics script.

Changes only the CLI and two source-file lookups; no physics or cup logic changes.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import ast


def patch(source: str) -> str:
    flag = "    ap.add_argument('--trajectory', type=Path,"
    if flag not in source:
        # Insert before the output argument, even if the variable name differs.
        lines = source.splitlines(keepends=True)
        candidates = [i for i, line in enumerate(lines)
                      if "add_argument(" in line and "'--output'" in line]
        if len(candidates) != 1:
            raise RuntimeError('Expected one --output argparse definition; file unchanged')
        i = candidates[0]
        indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
        lines.insert(i, indent + "ap.add_argument('--trajectory', type=Path, "
                     "help='Explicit input sim_qpos.npz from kinematic replay')\n")
        source = ''.join(lines)

    old_run = "    src=root/'outputs'/f'smoothed_{take}'/'sim_qpos.npz'"
    new_run = "    src=(args.trajectory.expanduser().resolve() if args.trajectory else root/'outputs'/f'smoothed_{take}'/'sim_qpos.npz')"
    if old_run in source:
        source = source.replace(old_run, new_run, 1)
    elif new_run not in source:
        raise RuntimeError('Run trajectory source not recognized; file unchanged')

    old_view = '''        trajectory_file = (
            root / "outputs" / f"smoothed_{args.take}"
            / "sim_qpos.npz"
        )'''
    new_view = '''        trajectory_file = (
            args.trajectory.expanduser().resolve() if args.trajectory else
            root / "outputs" / f"smoothed_{args.take}" / "sim_qpos.npz"
        )'''
    if old_view in source:
        source = source.replace(old_view, new_view, 1)
    elif new_view not in source:
        raise RuntimeError('Video-view trajectory source not recognized; file unchanged')
    ast.parse(source)
    return source


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path('code/dex5_physics_replay.py'))
    ap.add_argument('--check', action='store_true', help='Check patch compatibility without editing')
    args = ap.parse_args()
    source = args.source.resolve()
    original = source.read_text()
    modified = patch(original)
    if args.check:
        print('PASS: trajectory patch is compatible; no files modified.')
        return
    if modified == original:
        print('Already patched:', source)
        return
    backup = source.with_name(source.name + '.bak_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    shutil.copy2(source, backup)
    source.write_text(modified)
    print('Patched:', source)
    print('Backup:', backup)
    print('Physics formulas, cup placement, and geometry unchanged.')


if __name__ == '__main__':
    main()
