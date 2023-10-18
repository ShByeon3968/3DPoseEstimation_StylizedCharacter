import json
import numpy as np

with open('./Human36M/annotations/Human36M_subject1_camera.json','rb') as f:
    cam = json.load(f)
with open('./Human36M/annotations/Human36M_subject1_data.json','rb') as f:
    data = json.load(f)
with open('./Human36M/annotations/Human36M_subject1_joint_3d.json','rb') as f:
    joint = json.load(f)
with open('./characters/dataset/meta_info/file_2.json','rb') as f:
    character = json.load(f)

# print(cam['1'])
# print(cam['1'])
# print(joint.keys())
# print(character['T'])

new_cam = {}
new_cam['R'] = character['R']
new_cam['t'] = character['T']
new_cam['f'] = [character['fx'],character['fy']]
new_cam['c'] = [character['cx'],character['cy']]

new_cam_dict = {}
new_cam_dict['1'] = new_cam
with open('./characters/dataset/annotations/character_subject1_camera.json','w') as f:
    json.dump(new_cam_dict,f)

new_joint = {}
for frame_idx in character['joint_3d']:
    joint_3d = character['joint_3d'][frame_idx]
    new_joint[frame_idx] = list(joint_3d.values())
with open('./characters/dataset/annotations/character_subject1_joint_3d.json','w') as f:
    json.dump(new_joint,f)