from pathlib import Path
import py_compile

src = Path("dex5_batch_pipeline.py")
dst = Path("dex5_batch_pipeline_v2.py")

if not src.exists():
    raise SystemExit("Missing dex5_batch_pipeline.py")

if dst.exists():
    raise SystemExit(
        "dex5_batch_pipeline_v2.py already exists. "
        "Existing file preserved."
    )

s = src.read_text()

def replace(old, new):
    global s
    if s.count(old) != 1:
        raise RuntimeError(
            f"Expected exactly one occurrence:\n{old[:160]}"
        )
    s = s.replace(old, new, 1)


# ---------------------------------------------------------
# Interactive wristband prompt-frame selector
# ---------------------------------------------------------

selector = '''
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
        saved.write_text(str(n) + "\\n")

    return n


'''

replace(
    "def process_one(take, args):",
    selector + "def process_one(take, args):"
)


# ---------------------------------------------------------
# SAM 3 stage with selectable starting frame
# ---------------------------------------------------------

old = '''    stage('02 SAM3 wrist masks', [sam_py, CODE/'dex5_sam3_video_masks.py',
          '--frames-dir', trial/'left_jpg', '--weights', weights,
          '--out', trial, '--overlay-every', args.overlay_every],
          trial/'sam_tracking.csv', args)'''

new = '''    sam_script = CODE / "dex5_sam3_video_masks.py"
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
        )'''

replace(old, new)


# ---------------------------------------------------------
# Avoid repeating existing completed stages
# ---------------------------------------------------------

replace(
    "    if not args.dry_run: report_one(take, trial)",
    '''    if not args.dry_run:
        if args.objects:
            export_object_world_reference(take, args)
        report_one(take, trial)'''
)


# ---------------------------------------------------------
# Convert cup/coaster table XY to world coordinates
# ---------------------------------------------------------

objects_code = '''
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
        json.dumps(output, indent=2) + "\\n"
    )

    print("Object world reference:", destination)


'''

replace(
    "def report_one(take, trial):",
    objects_code + "def report_one(take, trial):"
)


# ---------------------------------------------------------
# New command-line arguments
# ---------------------------------------------------------

replace(
    "    pro.add_argument('--force',action='store_true',help='Recompute existing outputs; use carefully')",
    '''    pro.add_argument("--choose-prompt", action="store_true")
    pro.add_argument("--prompt-frame", type=int, default=None)
    pro.add_argument('--force',action='store_true',help='Recompute existing outputs; use carefully')'''
)


# ---------------------------------------------------------
# Validate and create v2
# ---------------------------------------------------------

dst.write_text(s)
py_compile.compile(str(dst), doraise=True)

print("Created:", dst.resolve())
print("Original preserved:", src.resolve())
print("Python syntax check: PASS")
