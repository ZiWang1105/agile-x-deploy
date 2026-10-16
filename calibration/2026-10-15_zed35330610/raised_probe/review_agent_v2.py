"""Document image-only review of rail face corners after diagnostic residuals.

No model parameters or CAD anchor coordinates are altered. Pixel revisions are
read from 2x original-image crops. Revision one remains available for audit.
The end-cap side faces make several root corners ambiguous; these annotations
remain a candidate, including their high residuals.
"""
import json
from pathlib import Path
import numpy as np
from agilex_deploy.camera_calibration import save_json, solve_dataset

p=Path(__file__).resolve().parent
d=json.loads((p/'dataset_agent_v1.json').read_text())
rails=[
 [[338,257],[332,283],[567,276],[563,301]],
 [[430,177],[425,201],[655,191],[652,216]],
 [[264,180],[258,199],[481,191],[475,216]],
 [[283,220],[277,252],[561,240],[557,273]],
 [[298,125],[290,154],[560,137],[558,173]],
 [[347,170],[338,193],[615,170],[612,196]],
 [[298,208],[297,234],[525,237],[523,258]],
 [[358,191],[362,215],[607,123],[607,151]],
 [[271,120],[255,146],[500,220],[486,248]],
 [[423,257],[417,283],[660,278],[656,305]],
 [[253,260],[248,282],[477,280],[473,303]],
 [[379,286],[374,305],[577,303],[574,325]],
 [[324,240],[318,265],[576,288],[573,314]],
 [[369,273],[364,292],[550,265],[549,285]],
 [[352,280],[353,311],[605,216],[610,244]],
]
names=['rail_ypos_tip','rail_ypos_root','rail_yneg_tip','rail_yneg_root']
changes=[]
for i,pose in enumerate(d['poses']):
 for o in d['observations']:
  if o['pose_id']!=pose['id'] or o['anchor_id'] not in names: continue
  new=rails[i][names.index(o['anchor_id'])]
  if new!=o['uv']:
   changes.append(dict(pose_id=pose['id'],anchor_id=o['anchor_id'],old_uv=o['uv'],new_uv=new))
   o['uv']=new
   o['selection']='Assistant visual review on 2x photo crop, original pixel coordinates; no projected point copied'
d['annotation_notes']['revision']='v2: correct pose-one background bracket confusion and review rail end-cap front versus side edges. Side-edge ambiguity remains.'
save_json(p/'annotation_revisions_v2.json',changes)
save_json(p/'dataset_agent_v2.json',d)
r=solve_dataset(p/'dataset_agent_v2.json',p/'results_agent_v2/candidate.json')
print('all RMS',r['all_rms_px'],'inlier RMS',r['inlier_rms_px'],'inliers',sum(r['inliers']),'/',len(r['inliers']))
print('Worst held-out pose RMS',max(x.get('rms_px',float('inf')) for x in r['leave_one_pose_out']))
for name in d['anchors']:
 errors=[e for e,o in zip(r['errors_px'],d['observations']) if o['anchor_id']==name]
 if errors:print(name,len(errors),float(np.sqrt(np.mean(np.array(errors)**2))))
