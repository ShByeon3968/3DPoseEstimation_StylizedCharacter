import bpy
import argparse
import os
import json
from config import cfg
import numpy as np


if __name__ == '__main__':
    data_dict = {}
    bpy.ops.wm.read_factory_settings(use_empty=True)

    if bpy.data.objects.get("Cube"):
        bpy.data.objects.remove(bpy.data.objects["Cube"])

    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 1))
    new_camera = bpy.context.object

    new_camera.location = (-4.316697120666504, -5.400135040283203, 4.194668769836426)
    new_camera.rotation_euler = (1.1356264352798462, 2.513778269985778e-07, -0.6934781670570374)


    fbx_file_path = cfg.fbx_file

    bpy.ops.import_scene.fbx(filepath=fbx_file_path)

    camera = bpy.data.objects.get("Camera")

    if camera:

        sensor_width = camera.data.sensor_width
        sensor_height = camera.data.sensor_height

        resolution_x = bpy.context.scene.render.resolution_x
        resolution_y = bpy.context.scene.render.resolution_y

        focal_length_x = (resolution_x * camera.data.lens) / (2 * sensor_width)
        focal_length_y = (resolution_y * camera.data.lens) / (2 * sensor_height)

        principal_point_x = resolution_x / 2
        principal_point_y = resolution_y / 2

        print("Camera Intrinsic Parameters:")
        print("Focal Length (x-axis):", focal_length_x)
        print("Focal Length (y-axis):", focal_length_y)
        print("Principal Point (x-axis):", principal_point_x)
        print("Principal Point (y-axis):", principal_point_y)

        rotation_matrix = camera.rotation_euler.to_matrix().to_3x3()
        translation_vector = camera.location

        print("Camera Rotation Matrix:")
        print(rotation_matrix)
        print("Camera Translation Vector:")
        print(translation_vector)
    else:
        print("No active camera found.")

    data_dict['fx'] = focal_length_x
    data_dict['fy'] = focal_length_y
    data_dict['cx'] = principal_point_x
    data_dict['cy'] = principal_point_y
    data_dict['R'] = [list(mat) for mat in rotation_matrix]
    data_dict['T'] = list(translation_vector)

    armature = bpy.data.objects['Armature']

    if armature:
        frame_joint_coordinates = {}

        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')

        frame_start = bpy.context.scene.frame_start
        frame_end = bpy.context.scene.frame_end

        for frame in range(frame_start, frame_end + 1):

            bpy.context.scene.frame_set(frame)

            frame_coordinates = {}
            for bone in armature.pose.bones:
                # world_matrix = armature.matrix_world @ bone.matrix
                frame_coordinates[bone.name] = list(bone.head)

            frame_joint_coordinates[frame] = frame_coordinates
        
        data_dict['joint_3d'] = frame_joint_coordinates
        anno_name = os.path.join(cfg.output_dir,cfg.json_name)

        with open(anno_name,'w') as f:
            json.dump(data_dict,f)
        bpy.ops.wm.save_as_mainfile(filepath=cfg.scene_path)
    else:
        print("No armature object found.")