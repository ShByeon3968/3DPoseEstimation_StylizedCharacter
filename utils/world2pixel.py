import bpy
from mathutils import Matrix
import numpy as np

def world_to_pixel(world_coords):
    world_coords = np.array(world_coords)
    transformed_coords = np.dot(R, world_coords) + T
    pixel_x = (transformed_coords[0] / transformed_coords[2]) * fx + cx
    pixel_y = (transformed_coords[1] / transformed_coords[2]) * fy + cy
    return pixel_x, pixel_y

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord
def cam2pixel(cam_coord, fx,fy, cx,cy):
    x = cam_coord[:, 0] / (cam_coord[:, 2]) * fx + cx
    y = cam_coord[:, 1] / (cam_coord[:, 2]) * fy + cy
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord


if __name__ == '__main__':
    # Get the armature object
    armature = bpy.data.objects['Armature']

    # Get the camera object
    camera = bpy.data.objects['Camera']

    joint_coordinates = []
    for bone in armature.pose.bones:
        joint_coordinates.append(bone.head)

    data_dict = {}
    if camera:

        sensor_width = camera.data.sensor_width
        sensor_height = camera.data.sensor_height

        resolution_x = bpy.context.scene.render.resolution_x
        resolution_y = bpy.context.scene.render.resolution_y

        fx = (resolution_x * camera.data.lens) / (2 * sensor_width)
        fy = (resolution_y * camera.data.lens) / (2 * sensor_height)

        cx = resolution_x / 2
        cy = resolution_y / 2

        print("Camera Intrinsic Parameters:")
        print("Focal Length (x-axis):", fx)
        print("Focal Length (y-axis):", fy)
        print("Principal Point (x-axis):", cx)
        print("Principal Point (y-axis):", cy)

        rotation_matrix = camera.rotation_euler.to_matrix().to_3x3()
        translation_vector = camera.location
    else:
        print("No active camera found.")

    data_dict['fx'] = fx
    data_dict['fy'] = fy
    data_dict['cx'] = cx
    data_dict['cy'] = cy
    data_dict['R'] = [list(mat) for mat in rotation_matrix]
    data_dict['T'] = list(translation_vector)

    cam_coord = world2cam(np.array(joint_coordinates),data_dict['R'],data_dict['T'])