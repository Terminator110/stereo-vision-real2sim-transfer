#!/usr/bin/env python3
import math
"""Batch runner for calibrated two-camera G1/Dex5 offline experiments.

Only orchestrates existing, tested programs. Does NOT claim full moving cup 6DoF.
Run in ~/physikal_workflow/dex-5/code with the three auxiliary scripts installed:
  dex5_sam3_video_masks.py, dex5_wristband_pipeline_v2.py,
  combine_wristband_stereo_pose.py, localize_objects_keyframes.py.

Use `record` for N distinct, Q-terminated takes, `process` for one or all takes,
`report` to summarize already generated files, and `replay` for a chosen take.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
FS = ROOT / 'FoundationStereo'
DEFAULT_CALIB = CODE / 'stereo/calibration/parallel_results_v3/stereo_calibration.npz'
DEFAULT_TABLE = CODE / 'stereo/calibration/parallel_results_v3/table_frame_multi.npz'
DEFAULT_RECORDINGS = CODE / 'stereo/recordings_parallel_v3'


def command(argv, *, dry_run=False, log=None):
    argv = [str(x) for x in argv]
    print('\n$ ' + ' '.join(shlex_quote(v) for v in argv), flush=True)
    if dry_run:
        return
    if log:
        log = Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w') as f:
            p = subprocess.Popen(argv, cwd=CODE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                print(line, end='', flush=True)
                f.write(line)
            rc = p.wait()
    else:
        rc = subprocess.run(argv, cwd=CODE).returncode
    if rc:
        raise RuntimeError(f'Command returned code {rc}: {argv[0]} {argv[1] if len(argv)>1 else ""}')


def shlex_quote(v):
    import shlex
    return shlex.quote(v)


def needs(path, desc):
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f'{desc} missing: {p}')
    return p.resolve()


def python_env(env, label):
    if not env:
        return Path(sys.executable)
    return needs(env, label)


def valid_take(path):
    p = Path(path).expanduser().resolve()
    for fn in ('camera1.avi', 'camera2.avi', 'timestamps.csv'):
        needs(p/fn, f'Take {p.name} {fn}')
    return p


def takes_from(args):
    if getattr(args, 'take', None):
        return [valid_take(args.take)]
    root = needs(args.recordings, 'Recordings directory')
    return [valid_take(p) for p in sorted(root.glob('take_*')) if
            p.is_dir() and (p/'camera1.avi').exists()]


def trial_for(take):
    # Preserve legacy prefix so this can coexist with previous experiments.
    return FS / ('wristband_batch_' + take.name)


def read_csv(path):
    if not Path(path).exists(): return []
    with open(path, newline='') as f: return list(csv.DictReader(f))


def longest_false(rows, key, success='VALID'):
    length = longest = 0
    for r in rows:
        if r.get(key) == success: length = 0
        else:
            length += 1
            longest = max(length, longest)
    return longest


def object_keyframes(take, interval, *, dry_run=False, force=False):
    """Extract synchronized ORIGINAL camera images, preserving saved cam2 rotation."""
    dst = take/'processed'/'object_keyframes'
    if not force and len(list(dst.glob('frame_*_camera1.png'))) >= 2:
        print('SKIP existing object keyframes', dst)
        return
    print(f'Extracting paired object keyframes every {interval:g}s -> {dst}')
    if dry_run: return
    import cv2
    dst.mkdir(parents=True, exist_ok=True)
    a = cv2.VideoCapture(str(take/'camera1.avi'))
    b = cv2.VideoCapture(str(take/'camera2.avi'))
    if not a.isOpened() or not b.isOpened():
        raise RuntimeError('Could not open recorded videos')
    fps = a.get(cv2.CAP_PROP_FPS) or 20
    times = {int(r['frame']): float(r['time_s']) for r in read_csv(take/'timestamps.csv')}
    idx = 0; last = -1e100; saved = 0
    while True:
        ok1, f1 = a.read(); ok2, f2 = b.read()
        if not (ok1 and ok2): break
        t = times.get(idx, idx/fps)
        if t-last >= interval - 1e-4:
            for cam, frame in ((1, f1), (2, f2)):
                fn = dst/f'frame_{idx:06d}_camera{cam}.png'
                if not cv2.imwrite(str(fn), frame):
                    raise IOError(f'Unable to write {fn}')
            saved += 1; last = t
        idx += 1
    a.release(); b.release()
    print(f'Extracted {saved} object keyframe pairs')


def stage(title, cmd, output, args, *, output_list=None):
    output = Path(output)
    targets = list(map(Path, output_list)) if output_list else [output]
    if all(p.exists() for p in targets) and not args.force:
        print('SKIP existing:', title, '->', ', '.join(str(x) for x in targets))
        return
    print('\n' + '='*20 + ' ' + title + ' ' + '='*20)
    command(cmd, dry_run=args.dry_run, log=output.parent/'logs'/(title.replace(' ', '_')+'.log'))
    if not args.dry_run:
        for p in targets: needs(p, title+' output')



def choose_prompt_frame(trial):
    import cv2

    frames = sorted((trial / "left_jpg").glob("*.jpg"))
    if not frames:
        raise RuntimeError("No prepared frames: " + str(trial))

    idx = 0
    name = "Choose wristband prompt frame"

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    try:
        while True:
            im = cv2.imread(str(frames[idx]))
            if im is None:
                raise RuntimeError("Cannot read " + str(frames[idx]))

            cv2.putText(
                im,
                f"Frame {idx}/{len(frames)-1} "
                "A/D:1 B/N:10 Enter:Select",
                (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (0, 255, 255),
                1,
            )

            cv2.imshow(name, im)
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("d"), 83):
                idx = min(len(frames)-1, idx+1)
            elif key in (ord("a"), 81):
                idx = max(0, idx-1)
            elif key == ord("n"):
                idx = min(len(frames)-1, idx+10)
            elif key == ord("b"):
                idx = max(0, idx-10)
            elif key in (13, 10, 32):
                return idx
            elif key in (27, ord("q")):
                raise RuntimeError("Prompt selection cancelled")
    finally:
        cv2.destroyAllWindows()


def selected_prompt(trial, args):
    saved = trial / "selected_prompt_frame.txt"

    if saved.exists() and not args.force:
        n = int(saved.read_text().strip())
        print("Reusing prompt frame:", n)
    elif args.prompt_frame is not None:
        n = args.prompt_frame
    elif args.choose_prompt:
        n = choose_prompt_frame(trial)
    else:
        n = 0

    total = len(list((trial / "left_jpg").glob("*.jpg")))

    if not 0 <= n < total:
        raise RuntimeError(
            f"Prompt frame {n} outside 0..{total-1}"
        )

    if not args.dry_run:
        saved.write_text(str(n) + "\n")

    return n


def process_one(take, args):
    trial = trial_for(take)
    print('\n'+'#'*65+'\nTAKE:', take, '\nTRIAL:', trial, '\n'+'#'*65, flush=True)
    calib = needs(args.calib, 'Calibration')
    table = needs(args.table, 'Table frame')
    g1 = python_env(args.g1_python, 'g1_lerobot Python')
    fs_py = python_env(args.fs_python, 'FoundationStereo Python')
    sam_py = python_env(args.sam_python or args.g1_python, 'SAM3 Python')
    object_py = python_env(args.object_python or args.sam_python or args.g1_python, 'Object SAM Python')
    weights = needs(args.sam_weights, 'SAM3 checkpoint')
    repo = needs(args.fs_repo, 'FoundationStereo repository')
    ckpt = needs(args.fs_ckpt, 'FoundationStereo checkpoint')
    for fn in ('dex5_sam3_video_masks.py', '12_stereo_g1_dex5.py',
               'combine_wristband_stereo_pose.py'):
        needs(CODE/fn, 'Project tool')
    needs(FS/'scripts/dex5_wristband_pipeline_v2.py', 'Original wristband processor')
    if args.objects:
        needs(CODE/'localize_objects_keyframes.py', 'Cup/coaster localizer')
    trial.mkdir(parents=True, exist_ok=True)

    stage('01 Prepare stereo', [fs_py, FS/'scripts/dex5_wristband_pipeline_v2.py',
          'prepare', '--take', take, '--calib', calib, '--out', trial, '--scale', args.scale],
          trial/'meta.json', args)
    sam_script = CODE / "dex5_sam3_video_masks.py"
    sam_source = sam_script.read_text()

    if (trial / "sam_tracking.csv").exists() and not args.force:
        print("SKIP existing SAM3 tracking:", trial)
    else:
        prompt = (
            selected_prompt(trial, args)
            if not args.dry_run
            else (args.prompt_frame or 0)
        )

        if prompt and "--prompt-frame" not in sam_source:
            raise RuntimeError(
                "SAM3 script does not support --prompt-frame"
            )

        sam_cmd = [
            sam_py,
            sam_script,
            "--frames-dir", trial / "left_jpg",
            "--weights", weights,
            "--out", trial,
            "--overlay-every", args.overlay_every,
        ]

        if "--prompt-frame" in sam_source:
            sam_cmd += ["--prompt-frame", prompt]

        stage(
            "02 SAM3 wrist masks",
            sam_cmd,
            trial / "sam_tracking.csv",
            args,
        )
    # Important: depth must never be run without real segmentation masks.
    stage('03 FoundationStereo depth', [str(Path(args.fs_python).expanduser()), FS/'scripts/dex5_wristband_pipeline_v2.py',
          'depth', '--out', trial, '--fs-repo', repo, '--fs-ckpt', ckpt,
          '--iters', args.iters, '--vis-every', args.overlay_every],
          trial/'wrist_xyz_camera1.csv', args)
    sdir = trial/'stereo_retarget'
    if args.force and sdir.exists() and not args.dry_run:
        print('FORCE: removing existing stereo output directory:', sdir)
        shutil.rmtree(sdir)
    # Resume-safe stereo retargeting.
    # The stereo script requires a completely new output directory.
    stereo_dir = trial / 'stereo_retarget'
    required_stereo = (
        stereo_dir / 'stereo_quality.csv',
        stereo_dir / 'stereo_landmarks_and_targets.npz',
        stereo_dir / 'report.json',
    )

    if stereo_dir.exists() and not all(f.is_file() for f in required_stereo):
        if not args.dry_run:
            backup = stereo_dir.with_name(
                'stereo_retarget_incomplete_' +
                datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            )
            stereo_dir.rename(backup)
            print('Moved incomplete stereo output to:', backup)

    stage('04 Stereo palm and fingers', [g1, CODE/'12_stereo_g1_dex5.py',
          '--left-video', take/'camera1.avi', '--right-video', take/'camera2.avi',
          '--calib', calib, '--table-frame', table,
          '--max-reproj', args.max_reproj, '--headless', '--output', sdir],
          sdir/'stereo_landmarks_and_targets.npz', args,
          output_list=[sdir/'stereo_landmarks_and_targets.npz',sdir/'stereo_quality.csv'])
    stage('05 Combine aligned pose and fingers',
          [g1, CODE/'combine_wristband_stereo_pose.py', '--trial', trial,
           '--stereo', sdir, '--table-frame', table],
          trial/'palm_pose_6dof.csv', args,
          output_list=[trial/'palm_pose_6dof.csv', trial/'joint_targets.csv'])

    if args.objects:
        object_keyframes(take, args.object_every_s,
                         dry_run=args.dry_run, force=args.force)
        stage('06 Cup coaster locations',
              [object_py, CODE/'localize_objects_keyframes.py', '--take', take,
               '--calib', calib, '--table', table, '--model', weights,
               '--cup-radius-mm', args.cup_radius_mm,
               '--cup-height-mm', args.cup_height_mm,
               '--coaster-top-mm', args.coaster_top_mm],
              take/'processed/object_localization/objects_keyframes.csv', args)
    if not args.dry_run:
        if args.objects:
            export_object_world_reference(take, args)
        report_one(take, trial)
    print('\nReplay command:')
    print('python dex5_batch_pipeline.py replay --take', shlex_quote(str(take)))



def export_object_world_reference(take, args):
    import statistics

    source = (
        take / "processed" / "object_localization"
        / "objects_keyframes.csv"
    )

    rows = read_csv(source)

    if not rows:
        print("No cup/coaster localization data.")
        return

    output = {
        "reference": "Static tabletop object locations",
        "world_origin_m": [0.385, -0.120, 0.832],
        "table_yaw_deg": -90,
        "cup_radius_assumed_m": args.cup_radius_mm / 1000,
        "cup_height_assumed_m": args.cup_height_mm / 1000,
        "coaster_top_assumed_m": args.coaster_top_mm / 1000,
        "objects": {},
    }

    for obj, status in [
        ("cup", "cup_valid_tabletop"),
        ("coaster", "coaster_valid"),
    ]:
        points = []
        frames = []

        for row in rows:
            if str(row.get(status, "")).strip() != "1":
                continue

            try:
                x = float(row[f"{obj}_x_table_m"])
                y = float(row[f"{obj}_y_table_m"])
                frame = int(row["frame"])

                if not (math.isfinite(x) and math.isfinite(y)):
                    continue

                points.append((x, y))
                frames.append(frame)

            except (ValueError, KeyError, TypeError):
                continue

        if not points:
            output["objects"][obj] = {
                "valid": False,
                "reason": "No valid keyframes",
            }
            continue

        x = statistics.median(p[0] for p in points)
        y = statistics.median(p[1] for p in points)

        # Rz(-90deg) @ [x,y,z] = [y,-x,z]
        world_x = 0.385 + y
        world_y = -0.120 - x

        if obj == "cup":
            z = 0.832 + args.cup_height_mm / 2000
            z_meaning = "Assumed upright cup centre"
        else:
            z = 0.832 + args.coaster_top_mm / 1000
            z_meaning = "Assumed coaster top surface"

        output["objects"][obj] = {
            "valid": True,
            "table_xy_m": [x, y],
            "world_xy_m": [world_x, world_y],
            "world_z_reference_m": z,
            "z_reference_meaning": z_meaning,
            "frames": frames,
            "valid_keyframes": len(points),
        }

    output["limitations"] = [
        "Static tabletop localization only.",
        "No lifted cup 6DoF pose.",
        "No measured cup handle yaw.",
        "No coaster yaw.",
        "World positions must be verified against MuJoCo geometry.",
    ]

    destination = source.parent / "object_world_reference.json"

    destination.write_text(
        json.dumps(output, indent=2) + "\n"
    )

    print("Object world reference:", destination)


def report_one(take, trial):
    mask = read_csv(trial/'sam_tracking.csv')
    depth = read_csv(trial/'wrist_xyz_camera1.csv')
    palm = read_csv(trial/'palm_pose_6dof.csv')
    qual = read_csv(trial/'stereo_retarget/stereo_quality.csv')
    objects = read_csv(take/'processed/object_localization/objects_keyframes.csv')
    def rate(rows,key,val):
        return [sum(r.get(key)==val for r in rows),len(rows)]
    info = {
        'take':str(take),'trial':str(trial),
        'sam_valid_total':rate(mask,'status','VALID'),
        'wrist_depth_valid_total':rate(depth,'status','VALID'),
        'depth_longest_invalid_frames':longest_false(depth,'status'),
        'palm_orientation_valid_total':rate(palm,'palm_status','VALID'),
        'stereo_complete_total':rate(qual,'stereo_status','STEREO_VALID'),
        'cup_valid_keyframes_total':rate(objects,'cup_valid_tabletop','1'),
        'coaster_valid_keyframes_total':rate(objects,'coaster_valid','1'),
        'output_paths':{
            'wrist_camera1':str(trial/'wrist_xyz_camera1.csv'),
            'combined_pose':str(trial/'palm_pose_6dof.csv'),
            'dex5_fingers':str(trial/'joint_targets.csv'),
            'stereo_npz':str(trial/'stereo_retarget/stereo_landmarks_and_targets.npz'),
            'cup_coaster_keyframes':str(take/'processed/object_localization/objects_keyframes.csv'),
        },
        'notes': ['Wrist XYZ is visible wristband surface, not anatomical wrist centre.',
                  'Cup and coaster are estimated only in keyframes with successful two-view masks.',
                  'Cup model assumes upright stationary cup resting on tabletop, given radius and height.',
                  'Coaster assumes planar visible top; neither object has dynamic 6DoF tracking.',
                  'Video FPS-derived timestamps may differ from capture timing.']
    }
    # Robust positional estimates across sufficiently valid keyframes, not a dynamic pose.
    if objects:
        import statistics
        for obj in ('cup','coaster'):
            status = 'cup_valid_tabletop' if obj=='cup' else 'coaster_valid'
            cols = (f'{obj}_x_table_m', f'{obj}_y_table_m')
            pts = []
            for row in objects:
                if row.get(status)!='1':continue
                try: pts.append([float(row[c]) for c in cols])
                except (ValueError, KeyError, TypeError): pass
            info[obj+'_static_xy_median_table_m'] = (
                [statistics.median(p[i] for p in pts) for i in (0,1)] if pts else None)
    trial.mkdir(parents=True, exist_ok=True)
    (trial/'batch_report.json').write_text(json.dumps(info,indent=2)+'\n')
    print('\n== QUALITY ==')
    for k in ('sam_valid_total','wrist_depth_valid_total','depth_longest_invalid_frames',
              'palm_orientation_valid_total','stereo_complete_total',
              'cup_valid_keyframes_total','coaster_valid_keyframes_total',
              'cup_static_xy_median_table_m','coaster_static_xy_median_table_m'):
        if k in info: print(f'{k}: {info[k]}')
    print('Report:',trial/'batch_report.json')


def replay(take,args):
    trial=trial_for(take)
    needs(trial/'palm_pose_6dof.csv','Combined pose')
    needs(trial/'joint_targets.csv','Dex5 finger targets')
    script = needs(CODE/'dex5_video_to_sim_wrist_smoothed.py','Smoothed MuJoCo replay')
    command([sys.executable, script, '--palm-csv', trial/'palm_pose_6dof.csv',
             '--finger-csv',trial/'joint_targets.csv','--video',take/'camera1.avi',
             '--ik-reference','wrist','--ik-mode','pose',
             '--table-yaw-deg','-90', '--table-origin-world','0.385','-0.120','0.832',
             '--xyz-window','9', '--orientation-window','7',
             '--orientation-max-gap-s','0.30', '--position-weight','10',
             '--pose-weight','0.5','--max-arm-speed','4',
             '--preposition-seconds','2','--robot-alpha','0.32',
             '--output',ROOT/'outputs'/('smoothed_'+take.name)],dry_run=args.dry_run)


def build_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sp=p.add_subparsers(dest='mode',required=True)
    rec=sp.add_parser('record',help='Record N new stereo takes (press Q for each)')
    rec.add_argument('--count',type=int,default=5)
    rec.add_argument('--recordings',type=Path,default=DEFAULT_RECORDINGS)
    rec.add_argument('--calib',type=Path,default=DEFAULT_CALIB)
    rec.add_argument('--dry-run',action='store_true')
    pro=sp.add_parser('process',help='Process --take or --all')
    pro.add_argument('--take',type=Path)
    pro.add_argument('--all',action='store_true')
    pro.add_argument('--recordings',type=Path,default=DEFAULT_RECORDINGS)
    pro.add_argument('--calib',type=Path,default=DEFAULT_CALIB)
    pro.add_argument('--table',type=Path,default=DEFAULT_TABLE)
    pro.add_argument('--sam-weights',type=Path,default=Path.home()/'physikal_workflow/sam3.pt')
    pro.add_argument('--g1-python',type=Path,default=None,
                     help='Python in g1_lerobot (default active python)')
    pro.add_argument('--sam-python',type=Path,default=None,
                     help='Python with Ultralytics SAM 3 (default g1-python)')
    pro.add_argument('--object-python',type=Path,default=None,
                     help='Python with SAM3SemanticPredictor (default sam-python)')
    pro.add_argument('--fs-python',type=Path,default=FS/'.venv/bin/python')
    pro.add_argument('--fs-repo',type=Path,default=FS)
    pro.add_argument('--fs-ckpt',type=Path,default=FS/'pretrained_models/11-33-40/model_best_bp2.pth')
    pro.add_argument('--objects',action='store_true',help='Estimate cup/coaster at keyframes')
    pro.add_argument('--object-every-s',type=float,default=1.0)
    pro.add_argument('--cup-radius-mm',type=float,default=41.)
    pro.add_argument('--cup-height-mm',type=float,default=98.)
    pro.add_argument('--coaster-top-mm',type=float,default=0.)
    pro.add_argument('--scale',type=float,default=.5)
    pro.add_argument('--iters',type=int,default=16)
    pro.add_argument('--max-reproj',type=float,default=8.)
    pro.add_argument('--overlay-every',type=int,default=5)
    pro.add_argument("--choose-prompt", action="store_true")
    pro.add_argument("--prompt-frame", type=int, default=None)
    pro.add_argument('--force',action='store_true',help='Recompute existing outputs; use carefully')
    pro.add_argument('--dry-run',action='store_true')
    rep=sp.add_parser('report')
    rep.add_argument('--take',type=Path)
    rep.add_argument('--recordings',type=Path,default=DEFAULT_RECORDINGS)
    rp=sp.add_parser('replay')
    rp.add_argument('--take',type=Path,required=True)
    rp.add_argument('--dry-run',action='store_true')
    return p


def main():
    p=build_parser(); args=p.parse_args()
    if args.mode=='record':
        if args.count<=0: p.error('--count must be positive')
        needs(CODE/'stereo/record_parallel_tags.py','Recorder')
        for i in range(args.count):
            print(f'\n===== TAKE {i+1}/{args.count} =====',flush=True)
            command([sys.executable,CODE/'stereo/record_parallel_tags.py',
                     '--cam1','2','--cam2','0','--rotate-cam2',
                     '--calib',args.calib,'--out-root',args.recordings,
                     '--width','1280','--height','720','--fps','20','--ids','999'],
                    dry_run=args.dry_run)
            if not args.dry_run and i+1<args.count:
                input('Press Enter to begin next take (Ctrl+C ends sequence)... ')
    elif args.mode=='process':
        if bool(args.take)==bool(args.all):p.error('Select exactly one of --take or --all')
        takes=takes_from(args)
        if not takes:p.error('No recorded takes found')
        for take in takes:
            try: process_one(take,args)
            except Exception as ex:
                print('ERROR processing', take.name, ':',repr(ex),file=sys.stderr)
                if not args.all:raise
                print('Continuing to next take. Resume later without --force.',flush=True)
    elif args.mode=='report':
        for take in takes_from(args): report_one(take,trial_for(take))
    else:
        replay(valid_take(args.take),args)

if __name__=='__main__':
    main()
