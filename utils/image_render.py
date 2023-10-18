'''
import bpy
from config import cfg

blend_file_path = cfg.scene_path

bpy.ops.wm.open_mainfile(filepath=blend_file_path)

scene = bpy.context.scene
camera = bpy.data.objects.get("Camera") 
if camera:
    scene.camera = camera
    
    bpy.context.scene.camera = camera  # Set the active camera
    bpy.context.scene.camera.rotation_euler = camera.rotation_euler
    bpy.context.scene.camera.location = camera.location

# Set rendering settings (adjust these according to your needs)
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.resolution_x = 1920  # Width of the image
scene.render.resolution_y = 1080  # Height of the image

# Set the output directory for the rendered frames
output_directory = cfg.render_dir
scene.render.filepath = output_directory

# Set the frame range for rendering (adjust these according to your needs)
scene.frame_start = 1
scene.frame_end = 250  # Example: Render frames 1 to 250

# Render and export image sequences
bpy.ops.render.render(animation=True)

# Print a message when rendering is complete
print("Rendering complete!")

# Optionally, you can save the .blend file with any changes
# bpy.ops.wm.save_mainfile(filepath=blend_file_path)
'''
import bpy
from config import cfg

# Set the path to your HDRI image
hdri_image_path = cfg.hdri_image_path

blend_file_path = cfg.scene_path

bpy.ops.wm.open_mainfile(filepath=blend_file_path)

scene = bpy.context.scene
camera = bpy.data.objects.get("Camera") 
if camera:
    scene.camera = camera
    
    bpy.context.scene.camera = camera  # Set the active camera
    bpy.context.scene.camera.rotation_euler = camera.rotation_euler
    bpy.context.scene.camera.location = camera.location

# Set rendering settings (adjust these according to your needs)
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.resolution_x = 1920  # Width of the image
scene.render.resolution_y = 1080  # Height of the image

# Set the output directory for the rendered frames
output_directory = cfg.render_dir
scene.render.filepath = output_directory

# Set the frame range for rendering (adjust these according to your needs)
scene.frame_start = 1
scene.frame_end = 250  # Example: Render frames 1 to 250

# Create a new world shader node tree
world = bpy.data.worlds.new('World')
scene.world = world
world.use_nodes = True
shader_tree = world.node_tree
shader_tree.nodes.clear()

# Add an Environment Texture node and load the HDRI image
environment_texture_node = shader_tree.nodes.new(type='ShaderNodeTexEnvironment')
environment_texture_node.location = (0, 0)

# Load the HDRI image
environment_texture_node.image = bpy.data.images.load(hdri_image_path)

# Create an Output node for the shader tree
output_node = shader_tree.nodes.new(type='ShaderNodeOutputWorld')
output_node.location = (400, 0)

# Connect the Environment Texture node to the Output node
shader_tree.links.new(environment_texture_node.outputs['Color'], output_node.inputs['Surface'])

# Render both RGB and Depth Map
scene.use_nodes = True
scene.view_layers["View Layer"].use_pass_combined = True
scene.view_layers["View Layer"].use_pass_z = True

# Render and export image sequences
bpy.ops.render.render(animation=True)

# Print a message when rendering is complete
print("Rendering complete!")

# Optionally, you can save the .blend file with any changes
# bpy.ops.wm.save_mainfile(filepath=blend_file_path)
