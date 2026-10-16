"""Audit exact policy requests against archived sensors; never runs a policy."""
import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np

from agilex_deploy.agent_sensors import RGBDFrame, ground_pixel


def audit(directory):
    directory = Path(directory)
    inputs = directory/'policy'
    count = 0
    for request in sorted(inputs.glob('request_*.json')):
        index = int(request.stem.split('_')[-1])
        payload = json.loads(request.read_text())
        content = payload['messages'][-1]['content']
        metadata = json.loads(content[0]['text'])
        expected = {'instruction','robot','pinch_poses','camera','depth_samples_uv_m','execution','available_arms','tool_geometry'}
        if set(metadata) != expected:
            raise ValueError('Unexpected policy input fields')
        robot_fields = {'observation_id','timestamp_s','joint_names','joint_positions_rad','joint_velocities_rad_s',
                        'gripper_position','gripper_states','left_tcp','right_tcp','mode','reason'}
        if set(metadata['robot']) != robot_fields:
            raise ValueError('Unexpected robot sensor fields')
        if 'Authorization' in payload or 'api_key' in payload:
            raise ValueError('Credential field in request archive')
        rgb = cv2.imread(str(inputs/f'rgb_{index:03}.png'))
        encoded = base64.b64decode(content[1]['image_url']['url'].split(',',1)[1])
        sent = cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(sent,rgb)
        depth = cv2.imread(str(inputs/f'depth_{index:03}.png'),cv2.IMREAD_UNCHANGED)/1000
        for u,v,z in metadata['depth_samples_uv_m']:
            if depth[v,u] != z:
                raise ValueError('Depth sample differs from archived metric frame')
        # Explicitly check privileged terms at the actual observation boundary.
        for forbidden in ('cube_position_m','cube_contacts','robot_environment_contacts','scene_config','task_cube'):
            if forbidden in json.dumps(metadata):
                raise ValueError('Privileged simulator information in model input')
        count += 1
    resolved = 0
    for turn in json.loads((directory/'turns.json').read_text()):
        grounding = turn['decision']['grounding']
        if grounding is None:
            continue
        index = turn['turn']
        camera = json.loads((inputs/f'camera_{index:03}.json').read_text())
        depth = cv2.imread(str(inputs/f'depth_{index:03}.png'),-1)/1000
        frame = RGBDFrame(np.empty((*depth.shape,3),np.uint8),depth,np.array(camera['K']),
                          np.array(camera['T_base_from_optical']),camera['timestamp_s'],camera['calibration'])
        point,surface = ground_pixel(frame,grounding['pixel_uv'],grounding['offset_base_m'])
        np.testing.assert_allclose(surface,grounding['surface_base_m'],atol=1e-12)
        np.testing.assert_allclose(point,grounding['pinch_base_m'],atol=1e-12)
        resolved += 1
    return {'requests_checked':count,'grounding_results_reproduced':resolved,'sensor_boundary_passed':True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    print(json.dumps(audit(parser.parse_args().directory),indent=2))
