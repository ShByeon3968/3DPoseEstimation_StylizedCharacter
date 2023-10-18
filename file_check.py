import json

with open('./Human36M/annotations/Human36M_subject1_joint_3d.json','rb') as f:
    file = json.load(f)

print(file.keys)