"""Summarize the raised physical calibration without opening robot transport."""
import json
from pathlib import Path
import cv2
import numpy as np
from agilex_deploy.camera_calibration import anchor_points, fit_extrinsics, project, save_json

p=Path(__file__).resolve().parent
d=json.loads((p/'dataset_agent_v2.json').read_text())
r=json.loads((p/'results_agent_v2/candidate.json').read_text())
out=p/'results_agent_v2'
points,pixels,pose_ids=anchor_points(d)
predicted,_=project(points,r['T_camera_from_base'],d['camera'])
errors=np.linalg.norm(predicted-pixels,axis=1)
names=np.array([o['anchor_id'] for o in d['observations']])
families={}
for name in sorted(set(names)):
 mask=names==name
 row={'count':int(mask.sum()),'all_fit_rms_px':float(np.sqrt(np.mean(errors[mask]**2)))}
 try:
  fit=fit_extrinsics(points[~mask],pixels[~mask],d['camera'])
  uv,_=project(points[mask],fit['T_camera_from_base'],d['camera'])
  row['leave_one_landmark_out_rms_px']=float(np.sqrt(np.mean(np.sum((uv-pixels[mask])**2,axis=1))))
 except ValueError as e:row['error']=str(e)
 families[name]=row

# Image-only static-background check. This is not a precision extrinsic test.
reference=cv2.imread(str(p/d['poses'][1]['image']),cv2.IMREAD_GRAYSCALE)
mask=np.zeros_like(reference);mask[20:260,650:1240]=255
features=cv2.goodFeaturesToTrack(reference,150,.02,15,mask=mask)
background=[]
for i,pose in enumerate(d['poses']):
 gray=cv2.imread(str(p/pose['image']),cv2.IMREAD_GRAYSCALE)
 tracked,ok,_=cv2.calcOpticalFlowPyrLK(reference,gray,features,None)
 backward,back_ok,_=cv2.calcOpticalFlowPyrLK(gray,reference,tracked,None)
 valid=(ok.ravel()!=0)&(back_ok.ravel()!=0)&(np.linalg.norm(backward-features,axis=2).ravel()<.5)
 flow=(tracked-features).reshape(-1,2)[valid]
 background.append({'pose_number':i+1,'tracked_points':len(flow),
  'median_shift_px':np.median(flow,axis=0).tolist(),
  'p90_displacement_px':float(np.percentile(np.linalg.norm(flow,axis=1),90))})

report={
 'status':'review_required_not_applied','captures':len(d['poses']),
 'observations':len(pixels),'landmarks':len(set(names)),
 'stationary_captures_accepted':sum(x['stationary_capture']['accepted'] for x in d['poses']),
 'all_rms_px':r['all_rms_px'],'inlier_rms_px':r['inlier_rms_px'],
 'inliers':sum(r['inliers']),'threshold_px':r['threshold_px'],
 'worst_leave_one_pose_out_rms_px':max(x['rms_px'] for x in r['leave_one_pose_out']),
 'per_landmark':families,'background_tracking_relative_to_pose_2':background,
 'robot_commands_sent_during_analysis':False,'active_configuration_changed':False,
 'observations_per_pose':[int((pose_ids==pose['id']).sum()) for pose in d['poses']],
 'findings':[
  'The 15 stationary captures and image hashes passed dataset validation.',
  'Annotations are assistant visual selections; the user did not click points.',
  'Pose-one rail selection initially confused a background bracket with the rail endpoint; revision history preserves the correction.',
  'Rail end-cap front/side edge ambiguity affected several initial annotations.',
  'Window lower-left edge can be an occluding wing silhouette rather than the same rigid CAD corner; it remains provisional.',
  'Wrist barrel screw correspondence remains less consistent than wing and forearm screw observations.',
  'The 40-observation wrist subset is a diagnostic selected after feature review, not independent proof of calibration accuracy.',
  'No camera intrinsics, joint offsets, kinematic dimensions, or CAD landmark coordinates were optimized.',
  'Pixel residuals are not an independent millimeter accuracy measurement; this candidate is not ready to apply.',
 ],
 'next_step':'Resolve occlusion-dependent gap corners and wrist screw CAD correspondence using the existing captures before requesting additional robot motion.',
}
save_json(out/'review.json',report)

font=cv2.FONT_HERSHEY_SIMPLEX
def label(im,text,xy,scale=.55,color=(230,230,230)):
 if scale>=.5:
  (w,h),base=cv2.getTextSize(text,font,scale,1)
  cv2.rectangle(im,(xy[0]-2,xy[1]-h-3),(xy[0]+w+2,xy[1]+base+2),(25,25,25),-1)
 cv2.putText(im,text,xy,font,scale,color,1,cv2.LINE_AA)

canvas=np.full((90+5*290,3*500,3),25,np.uint8)
label(canvas,'15 physical captures | 132 observations | CANDIDATE - NOT APPLIED',(15,28),.75)
label(canvas,'Green: observed image feature   Red cross: projected CAD point   Yellow: residual',(15,57),.6)
for i,pose in enumerate(d['poses']):
 im=cv2.imread(str(p/pose['image']))
 sel=pose_ids==pose['id']
 for observed,fitted in zip(pixels[sel],predicted[sel]):
  a=tuple(np.rint(observed).astype(int));b=tuple(np.rint(fitted).astype(int))
  cv2.line(im,a,b,(0,200,255),1);cv2.circle(im,a,4,(30,255,30),1)
  cv2.drawMarker(im,b,(30,30,255),cv2.MARKER_CROSS,10,1)
 tile=cv2.resize(im,(500,281));rms=float(np.sqrt(np.mean(errors[sel]**2)))
 label(tile,f'{i+1:02d}  n={sel.sum()}  RMS={rms:.2f}px',(10,23),.58)
 y=90+(i//3)*290;x=(i%3)*500;canvas[y:y+281,x:x+500]=tile
cv2.imwrite(str(out/'all_15_annotated.jpg'),canvas)

detail=np.full((100+2*480,3*600,3),25,np.uint8)
label(detail,'Raised-camera calibration review | all points RMS 4.52 px | worst held-out pose RMS 7.01 px',(15,30),.75)
label(detail,'Green circle = observation; red cross = projection. Point labels identify distinct CAD landmarks.',(15,62),.65)
short={'rail_ypos_tip':'R1','rail_ypos_root':'R2','rail_yneg_tip':'R3','rail_yneg_root':'R4',
 'wing_screw':'W','white_screw':'F','hinge_screw':'H','window_a':'A','window_b':'B','window_c':'C','window_d':'D'}
for j,i in enumerate([0,3,5,8,9,13]):
 pose=d['poses'][i];im=cv2.imread(str(p/pose['image']))
 im=cv2.convertScaleAbs(im,alpha=1.45,beta=3)
 sel=np.flatnonzero(pose_ids==pose['id'])
 for k in sel:
  a=tuple(np.rint(pixels[k]).astype(int));b=tuple(np.rint(predicted[k]).astype(int))
  cv2.line(im,a,b,(0,200,255),1);cv2.circle(im,a,4,(30,255,30),1)
  cv2.drawMarker(im,b,(30,30,255),cv2.MARKER_CROSS,10,1)
  label(im,short[names[k]],(a[0]+5,a[1]-7),.36,(150,255,150))
 tile=cv2.resize(im[100:660,100:800],(600,480))
 label(tile,f'Pose {i+1:02d} | {len(sel)} observations',(10,25),.6)
 y=100+(j//3)*480;x=(j%3)*600;detail[y:y+480,x:x+600]=tile
cv2.imwrite(str(out/'validation_detail.jpg'),detail)

(out/'REVIEW.md').write_text('''# Raised physical camera calibration review

The latest run contains 15 stationary captures and 132 image observations on
11 CAD landmarks. The assistant selected the image points. No robot commands
were sent during this analysis and no active calibration was changed.

The all-observation reprojection RMS is **4.52 px**. Of 132 observations,
112 are within the 6 px RANSAC threshold; their RMS is **3.13 px**. The worst
leave-one-pose-out RMS is **7.01 px**. These are pixel consistency measures,
not independently established metric accuracy.

The initial 132-point review had 7.38 px RMS. Enlarged image inspection found
one background-bracket misidentification and several rail front/side corner
ambiguities. `annotation_revisions_v2.json` records every pixel revision.
The original captures and first annotation version are preserved.

All fixed factory intrinsics, measured joints, geometry, and landmark local
coordinates remain unchanged. The candidate remains **not applied** because
some gap corners depend on occlusion and the wrist barrel screw is less
consistent. The 40-point wrist-only diagnostic has 2.66 px RMS and 5.64 px
worst held-out-pose RMS; that post-review subset is not proof of accuracy.

Next, resolve the gap-corner and barrel-screw CAD correspondences using the
existing images. Additional robot motion is not needed for that investigation.
See `review.json` for per-landmark held-out checks and static-background flow.
''')
print(json.dumps({k:report[k] for k in ['captures','observations','all_rms_px','inlier_rms_px','worst_leave_one_pose_out_rms_px','observations_per_pose']},indent=2))
print('Maximum background p90:',max(x['p90_displacement_px'] for x in background))
