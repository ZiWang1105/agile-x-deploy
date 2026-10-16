"""Reproduce visual annotations of the 15 raised captures; no hardware access.

Pixels were read by the assistant from brightness-enhanced copies with an
original-pixel grid, before fitting. They are not projected model coordinates.
Rounded corners and dark screw centers have several-pixel uncertainty.
"""
import json
from pathlib import Path
import numpy as np
from agilex_deploy.camera_calibration import save_json, solve_dataset

ROOT = Path(__file__).resolve().parent
doc = json.loads((ROOT / 'dataset.json').read_text())
old = json.loads((ROOT.parent / 'physical/dataset_agent_v1.json').read_text())
doc['anchors'] = old['anchors']
candidates = json.loads((ROOT / 'cad_candidate_anchors.json').read_text())
doc['anchors'].update(candidates)
rail_names = ['rail_ypos_tip', 'rail_ypos_root', 'rail_yneg_tip', 'rail_yneg_root']
rails = [
 [[319,256],[313,291],[572,277],[570,309]],
 [[430,179],[423,210],[657,192],[656,226]],
 [[264,179],[258,204],[488,192],[487,226]],
 [[283,220],[277,252],[561,240],[560,279]],
 [[298,125],[290,154],[563,139],[562,178]],
 [[347,170],[338,193],[617,171],[617,214]],
 [[298,208],[297,234],[531,236],[530,262]],
 [[358,191],[362,215],[606,137],[613,162]],
 [[271,120],[255,146],[501,221],[496,260]],
 [[423,257],[417,283],[660,278],[658,311]],
 [[253,260],[248,282],[483,279],[482,310]],
 [[379,286],[374,305],[582,304],[580,329]],
 [[324,240],[318,265],[576,288],[573,314]],
 [[369,273],[364,292],[550,267],[563,297]],
 [[352,280],[353,311],[605,216],[612,250]],
]
# Wing-center screw, forearm white housing screw, and wrist barrel screw.
screws = [
 [[414,398],[367,509],[443,440]],
 [[511,306],[471,413],[543,345]],
 [None,[289,410],[351,337]],
 [[365,396],[309,521],[403,446]],
 [[372,281],[330,407],[411,327]],
 [None,[354,435],[450,356]],
 [None,[360,444],[420,379]],
 [[496,294],[472,413],[544,317]],
 [[266,298],[199,421],[277,356]],
 [[508,401],[460,509],[543,442]],
 [None,[269,506],[343,432]],
 [None,[413,493],[472,434]],
 [None,None,None],
 [None,[370,504],[424,428]],
 [[494,395],[468,519],[544,426]],
]
# Visible rectangular gap corners; diagnostic only until CAD identity is checked.
windows = {
 1:[[406,380],[410,370],[462,388],[465,376]],
 2:[[493,291],[497,283],[555,298],[558,287]],
 4:[[367,371],[371,357],[427,378],[430,364]],
 9:[[294,282],[303,271],[333,306],[339,295]],
 10:[[494,384],[496,373],[553,391],[556,381]],
 11:[[324,382],[326,372],[368,390],[371,376]],
 12:[[440,391],[441,383],[486,398],[488,389]],
 13:[[401,382],[403,371],[471,391],[474,379]],
 15:[[472,384],[469,372],[538,367],[535,359]],
}
doc['observations'] = []
for i, pose in enumerate(doc['poses']):
 pairs = list(zip(rail_names, rails[i])) + list(zip(
     ['wing_screw','white_screw','hinge_screw'], screws[i]))
 pairs += list(zip(['window_a','window_b','window_c','window_d'], windows.get(i+1, [])))
 for name, uv in pairs:
  if uv is not None:
   doc['observations'].append(dict(pose_id=pose['id'],anchor_id=name,uv=uv,
       selection='Assistant visual inspection of original-pixel grid; no model prediction used'))
doc['annotation_notes'] = {
 'method':'Assistant visual selection, not an automatic detector or subpixel measurement.',
 'omissions':'Occluded screw centers and gap corners omitted. Gap anchors are provisional CAD surface picks.',
 'status':'Diagnostic annotations; not a validated calibration.',
 'fixed':'Factory intrinsics, measured joints, robot geometry and landmark local coordinates are not fitted.',
}
save_json(ROOT/'dataset_agent_v1.json',doc)
result=solve_dataset(ROOT/'dataset_agent_v1.json',ROOT/'results_agent_v1/candidate.json')
errors=np.array(result['errors_px'])
print('Observations:', len(errors), 'All RMS:',result['all_rms_px'],
      'Inlier RMS:',result['inlier_rms_px'],'Inliers:',sum(result['inliers']))
for name in doc['anchors']:
 mask=np.array([r['anchor_id']==name for r in doc['observations']])
 if mask.any(): print(name,int(mask.sum()),'RMS',float(np.sqrt(np.mean(errors[mask]**2))))
print('Worst held-out pose:',max(r.get('rms_px',float('inf')) for r in result['leave_one_pose_out']))
