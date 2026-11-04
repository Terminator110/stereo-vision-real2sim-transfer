from pathlib import Path
import py_compile

src = Path("dex5_live_stereo_to_wrist_ik.py")
dst = Path("dex5_live_sam3fs_local.py")

s = src.read_text()

def replace(old, new):
    global s
    if old not in s:
        raise RuntimeError(f"Expected code not found:\n{old[:120]}")
    s = s.replace(old, new, 1)

replace(
    "import argparse, csv, importlib.util, time",
    "import argparse, csv, importlib.util, time, sys, types"
)

replace(
    "ref = load_module('_dex5_live_wrist_ik_tools', HERE/'dex5_video_to_sim_wrist_site.py')",
    """ref = load_module('_dex5_live_wrist_ik_tools', HERE/'dex5_video_to_sim_wrist_site.py')
band = load_module('_dex5_fs_band',
                   ROOT/'FoundationStereo/scripts/dex5_wristband_pipeline_v2.py')

class BandFilter:
    def __init__(self, deadband_mm=1.5, cutoff_hz=1.2):
        self.xyz = None
        self.last = None
        self.deadband = deadband_mm / 1000
        self.cutoff = cutoff_hz

    def reset(self):
        self.xyz = None
        self.last = None

    def update(self, xyz, now):
        xyz = np.asarray(xyz, float)
        if not np.isfinite(xyz).all():
            return self.xyz, 'INVALID'

        if self.xyz is None:
            self.xyz = xyz.copy()
            self.last = now
            return self.xyz.copy(), 'ACQUIRED'

        dt = max(0.005, now-self.last)
        jump = np.linalg.norm(xyz-self.xyz)

        if jump > max(.025, 1.2*dt):
            return self.xyz.copy(), 'OUTLIER'

        self.last = now
        alpha = min(.4, 1-np.exp(-2*np.pi*self.cutoff*dt))
        new = self.xyz + alpha*(xyz-self.xyz)

        if np.linalg.norm(new-self.xyz) < self.deadband:
            return self.xyz.copy(), 'DEADBAND'

        self.xyz = new
        return self.xyz.copy(), 'TRACKING'
"""
)

replace(
    "    ap.add_argument('--cam1',type=int,default=2)",
    """    ap.add_argument('--sam-weights', type=Path,
                    default=ROOT.parent/'sam3.pt')
    ap.add_argument('--fs-repo', type=Path,
                    default=ROOT/'FoundationStereo')
    ap.add_argument('--fs-ckpt', type=Path,
                    default=ROOT/'FoundationStereo/pretrained_models/11-33-40/model_best_bp2.pth')
    ap.add_argument('--scale', type=float, default=.5)
    ap.add_argument('--iters', type=int, default=16)
    ap.add_argument('--deadband-mm', type=float, default=1.5)
    ap.add_argument('--smooth-cutoff-hz', type=float, default=1.2)
    ap.add_argument('--cam1',type=int,default=2)"""
)

replace(
    "    cal,size,Rct,tct=old.load_calibration(a.calib,a.table_frame)",
    """    if not a.sam_weights.is_file():
        ap.error('SAM3 checkpoint missing: ' + str(a.sam_weights))
    if not a.fs_ckpt.is_file():
        ap.error('FoundationStereo checkpoint missing: ' + str(a.fs_ckpt))
    if not 0 < a.scale <= 1:
        ap.error('--scale must be between 0 and 1')

    cal,size,Rct,tct=old.load_calibration(a.calib,a.table_frame)

    with np.load(a.calib) as c:
        K1,D1,R1,P1,K2,D2,R2,P2 = [
            c[k] for k in ('K1','D1','R1','P1','K2','D2','R2','P2')
        ]
        baseline = float(c['baseline_mm'])/1000

    W,H = size

    if P2[1,3]>=0 or abs(P2[0,3])>1e-4:
        raise RuntimeError('Expected vertical-baseline calibration')

    map1x,map1y = cv2.initUndistortRectifyMap(
        K1,D1,R1,P1[:,:3],size,cv2.CV_32FC1
    )
    map2x,map2y = cv2.initUndistortRectifyMap(
        K2,D2,R2,P2[:,:3],size,cv2.CV_32FC1
    )

    Kfs = np.array([
        [P1[1,1]*a.scale, 0, P1[1,2]*a.scale],
        [0, P1[0,0]*a.scale, (W-1-P1[0,2])*a.scale],
        [0,0,1.]
    ],float)

    fs,Padder,torch = band.load_fs(a)

    from ultralytics import SAM
    sam = SAM(str(a.sam_weights.resolve()))

    def prep(im,mapx,mapy):
        r=cv2.remap(im,mapx,mapy,cv2.INTER_LINEAR)
        r=cv2.rotate(r,cv2.ROTATE_90_COUNTERCLOCKWISE)

        if a.scale != 1:
            r=cv2.resize(
                r,None,fx=a.scale,fy=a.scale,
                interpolation=cv2.INTER_AREA
            )
        return r

    def disparity(l,r):
        lt=torch.from_numpy(
            np.ascontiguousarray(cv2.cvtColor(l,cv2.COLOR_BGR2RGB))
        ).cuda().float()[None].permute(0,3,1,2)

        rt=torch.from_numpy(
            np.ascontiguousarray(cv2.cvtColor(r,cv2.COLOR_BGR2RGB))
        ).cuda().float()[None].permute(0,3,1,2)

        pad=Padder(lt.shape,divis_by=32,force_square=False)
        lt,rt=pad.pad(lt,rt)

        with torch.inference_mode(),torch.autocast(
            'cuda',dtype=torch.float16
        ):
            dis=fs.forward(lt,rt,iters=a.iters,test_mode=True)

        return pad.unpad(dis.float()).detach().cpu().numpy().reshape(
            l.shape[:2]
        )

    filt=BandFilter(a.deadband_mm,a.smooth_cutoff_hz)
    roi=None
    last_orient_seen=None"""
)

replace(
    "            wrist=None;target=None;actual=None;err=float('nan');rerr=float('nan')\n"
    "            good_wrist=bool(p3 is not None and groups.get('wrist') and np.isfinite(p3[0]).all())\n"
    "            if good_wrist:\n"
    "                wrist=p3[0].copy()\n"
    "                target=yaw@wrist+origin\n"
    "                valid+=1",
    """            wrist=None
            target=None
            actual=None
            err=float('nan')
            rerr=float('nan')

            leftfs=prep(f1,map1x,map1y)
            rightfs=prep(f2,map2x,map2y)

            if roi is None:
                sel=cv2.selectROI(
                    'Select wristband on RECTIFIED LEFT',
                    leftfs,False,False
                )
                cv2.destroyWindow('Select wristband on RECTIFIED LEFT')
                xx,yy,ww,hh=map(int,sel)
                if ww>4 and hh>4:
                    roi=[xx,yy,xx+ww,yy+hh]

            band_state='NO_ROI'
            good_wrist=False

            if roi is not None:
                result_sam=sam.predict(
                    source=leftfs,
                    bboxes=roi,
                    imgsz=640,
                    device=0,
                    verbose=False,
                    save=False
                )[0]

                masks=result_sam.masks

                if masks is not None and len(masks.data):
                    mm=masks.data[0].detach().cpu().numpy()

                    mask=cv2.resize(
                        (mm>.5).astype(np.uint8),
                        (leftfs.shape[1],leftfs.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                    yy,xx=np.where(mask>0)

                    if len(xx)>=30:
                        bx=[
                            int(xx.min()),int(yy.min()),
                            int(xx.max()+1),int(yy.max()+1)
                        ]

                        oldmid=np.array([
                            (roi[0]+roi[2])/2,
                            (roi[1]+roi[3])/2
                        ])
                        newmid=np.array([
                            (bx[0]+bx[2])/2,
                            (bx[1]+bx[3])/2
                        ])

                        max_shift=max(
                            45.,
                            .8*np.hypot(
                                roi[2]-roi[0],
                                roi[3]-roi[1]
                            )
                        )

                        if np.linalg.norm(oldmid-newmid)<max_shift:
                            roi=[
                                max(0,bx[0]-12),
                                max(0,bx[1]-12),
                                min(leftfs.shape[1],bx[2]+12),
                                min(leftfs.shape[0],bx[3]+12)
                            ]

                            disp=disparity(leftfs,rightfs)

                            estimate,_,num,_,mad=band.sample_band_xyz(
                                mask,disp,Kfs,R1,baseline,
                                .35,2.5,3,20
                            )

                            if estimate is not None:
                                raw=Rct@estimate['cam1']+tct

                                wrist,band_state=filt.update(raw,now)

                                good_wrist=band_state not in (
                                    'OUTLIER','INVALID'
                                )

                                if good_wrist:
                                    target=yaw@wrist+origin
                                    valid+=1
                            else:
                                band_state='INVALID_DEPTH'
                        else:
                            band_state='ROI_JUMP'
                    else:
                        band_state='EMPTY_MASK'
                else:
                    band_state='SAM3_MISSING'"""
)

replace(
    "                        last_orientation=yaw@hand_R@site_to_anat.T",
    """                        desired=yaw@hand_R@site_to_anat.T

                        if last_orientation is None:
                            last_orientation=desired
                            last_orient_seen=now
                        else:
                            delta=Rotation.from_matrix(
                                desired@last_orientation.T
                            )

                            if np.degrees(delta.magnitude())<50:
                                correction=Rotation.from_rotvec(
                                    .25*delta.as_rotvec()
                                ).as_matrix()

                                last_orientation=correction@last_orientation
                                last_orient_seen=now"""
)

replace(
    "                old_q=qcur.copy()",
    """                if (last_orient_seen is None or
                        now-last_orient_seen>.30):
                    last_orientation=None

                old_q=qcur.copy()"""
)

# Remove finger updates from wrist-dependent IK block.
replace(
    """                if p3 is not None:
                    try:
                        finger_pose,fstate=retarget.update(p3,now)
                        d.qpos[finger_addrs]=finger_pose[finger_addrs]
                    except (FloatingPointError,ValueError):
                        pass
""",
    ""
)

replace(
    "                status='TRACKING'",
    """                status=(
                    'XYZ_ONLY_NO_ORIENTATION'
                    if a.ik_mode=='pose' and last_orientation is None
                    else 'TRACKING'
                )"""
)

replace(
    "            elif playing:\n                status='HOLD: INVALID STEREO'",
    """            elif playing:
                status='HOLD: INVALID WRISTBAND/DEPTH'"""
)

replace(
    "            # Draw the exact current target and actual wrist-joint site.",
    """            # Finger retargeting operates independently of wrist depth.
            if playing and p3 is not None:
                try:
                    finger_pose,fstate=retarget.update(p3,now)
                    d.qpos[finger_addrs]=finger_pose[finger_addrs]
                    mujoco.mj_forward(m,d)
                except (FloatingPointError,ValueError):
                    pass

            # Draw the exact current target and actual wrist-joint site."""
)

replace(
    """            if pts1 is not None:
                uv=tuple(np.rint(pts1[0]).astype(int))
                cv2.circle(vis,uv,9,(0,255,255),2)
                cv2.putText(vis,'MediaPipe WRIST (not band centroid)',(uv[0]+12,uv[1]-12),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),2)""",
    """            if wrist is not None:
                pcam=Rct.T@(wrist-tct)

                if pcam[2]>0:
                    uv,_=cv2.projectPoints(
                        pcam.reshape(1,1,3),
                        np.zeros(3),np.zeros(3),K1,D1
                    )
                    uv=tuple(np.rint(uv.reshape(2)).astype(int))

                    cv2.circle(vis,uv,10,(0,255,255),2)
                    cv2.putText(
                        vis,'SAM3+FS wristband',
                        (uv[0]+12,uv[1]-12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        .5,(0,255,255),2
                    )"""
)

replace(
    """            lines=[f'LIVE G1 wrist IK: {status} | {a.ik_mode} | {result["status"]}',""",
    """            lines=[f'LIVE SAM3+FS wrist: {band_state} | IK {status} | {a.ik_mode}',"""
)

replace(
    "            if key==ord('r'):\n                playing=False;status='REST';last_orientation=None",
    """            if key==ord('b'):
                roi=None
                filt.reset()
                last_orientation=None
                last_orient_seen=None

            if key==ord('r'):
                playing=False
                status='REST'
                last_orientation=None
                last_orient_seen=None
                filt.reset()"""
)

replace(
    """            if count%10==0:
                print(f'frame {count}: {status}; stereo={result["status"]};""",
    """            if count%10==0:
                print(f'frame {count}: {status}; band={band_state}; fingers={result["status"]};"""
)

dst.write_text(s)
py_compile.compile(str(dst),doraise=True)

print("Created:", dst)
print("Original script preserved:", src)
print("Python syntax check: PASS")
