from easydict import EasyDict as edict

cfg = edict()

cfg.output_dir = './characters/dataset/meta_info/'
cfg.json_name = 'Amy_1.json'

cfg.fbx_file = './characters/character_fbxes/Amy/Chapa-Giratoria.fbx'
cfg.scene_path = './characters/dataset/Scene/Amy/Chapa-Giratoria.blend'
cfg.render_dir = './characters/dataset/images/1/'

cfg.hdri_image_path = './small_empty_room_1_4k.exr'
cfg.render_dir = './characters/render_img/'
