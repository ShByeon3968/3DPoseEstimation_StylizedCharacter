import os.path as osp
import numpy as np
import math
import torch
import json
import copy
import scipy.sparse
import cv2
from pycocotools.coco import COCO

from utils.noise_utils import synthesize_pose

from utils.coord_utils import world2cam, cam2pixel, process_bbox, rigid_align, get_bbox
from utils.aug_utils import affine_transform, j2d_processing, augm_params, j3d_processing, flip_2d_joint
from utils.noise_stats import error_distribution




class Chracter(torch.utils.data.Dataset):
    def __init__(self, data_dir,mode):
        dataset_name = 'character'
        self.debug = None
        self.aug_flip = True
        self.MODEL_input_shape = (384, 288)
        self.data_dir = data_dir
        self.data_split = mode
        self.use_gt_input = True

        self.img_dir = osp.join(self.data_dir, 'images')
        self.annot_path = osp.join(self.data_dir, 'annotations')
        self.protocol = 1

        # character joint set
        self.character_joint_num = 16
        self.character_joints_name = (
        'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Head',
        'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        # 수정 필요 밑에
        self.character_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (13, 10))
        self.character_skeleton = (
        (0, 7), (7, 8), (8, 9), (8, 10), (10, 11), (11, 12), (8, 13), (13, 14), (14, 15), (0, 1), (1, 2),
        (2, 3), (0, 4), (4, 5), (5, 6))
        self.character_root_joint_idx = self.character_joints_name.index('Pelvis')
        self.character_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15)

        self.input_joint_name = 'character'
        self.joint_num, self.skeleton, self.flip_pairs = self.get_joint_setting(self.input_joint_name)

        self.datalist = self.load_data()
        # if self.data_split == 'test':
        #     det_2d_data_path = osp.join(self.data_dir, dataset_name, 'absnet_output_on_testset.json')
        #     self.datalist_pose2d_det = self.load_pose2d_det(det_2d_data_path, skip_img_path)
        #     print("Check lengths of annotation and detection output: ", len(self.datalist), len(self.datalist_pose2d_det))


    def load_pose2d_det(self, data_path, skip_list):
        pose_list = []
        with open(data_path) as f:
            data = json.load(f)
            for img_path, pose2d in data.items():
                pose2d = np.array(pose2d, dtype=np.float32)
                if img_path in skip_list:
                    continue
                pose_list.append({'img_name': img_path, 'pose2d': pose2d})
        pose_list = sorted(pose_list, key=lambda x: x['img_name'])
        return pose_list

    def get_joint_setting(self, joint_category='character'):
        joint_num = eval(f'self.{joint_category}_joint_num')
        skeleton = eval(f'self.{joint_category}_skeleton')
        flip_pairs = eval(f'self.{joint_category}_flip_pairs')

        return joint_num, skeleton, flip_pairs

    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 5  # 50
        elif self.data_split == 'test':
            return 50 #
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            if self.protocol == 1:
                # subject = [1, 5, 6, 7, 8, 9]
                subject = [1]
            elif self.protocol == 2:
                subject = [1, 5, 6, 7, 8]
        elif self.data_split == 'test':
            if self.protocol == 1:
                subject = [11]
            elif self.protocol == 2:
                subject = [9, 11]
        else:
            assert 0, print("Unknown subset")

        if self.debug:
            subject = subject[0:1]

        return subject

    def load_data(self):
        print('Load annotations of Character Protocol ' + str(self.protocol))
        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()

        # aggregate annotations from each subject
        cameras = {}
        joints = {}
        datalist = []
        for subject in subject_list:
            # camera load
            with open(osp.join(self.annot_path, 'character_subject' + str(subject) + '_camera.json'), 'r') as f:
                cameras = json.load(f)
            # joint coordinate load
            with open(osp.join(self.annot_path, 'character_subject' + str(subject) + '_joint_3d.json'), 'r') as f:
                joints = json.load(f)

        cam_param = cameras['1']
        R, t, f, c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'],
                                                                            dtype=np.float32), np.array(
            cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
        cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}

        # project world coordinate to cam, image coordinate space
        for frame_idx,value in enumerate(joints):
            joint_world = np.array(joints[str(frame_idx+1)],
                                    dtype=np.float32)
            print(joint_world.shape)
            joint_cam = world2cam(joint_world, R, t)
            joint_img = cam2pixel(joint_cam, f, c)
            joint_vis = np.ones((self.character_joint_num, 1))


            datalist.append({
                'joint_img': joint_img,  # [x_img, y_img, z_cam]
                'joint_cam': joint_cam,  # [X, Y, Z] in camera coordinate
                'joint_vis': joint_vis,
                'cam_param': cam_param})

        return datalist


    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        flip, rot = augm_params(is_train=(self.data_split == 'train'),aug_filp=self.aug_flip)

        # h36m joints from datasets
        joint_cam_h36m, joint_img_h36m = data['joint_cam'], data['joint_img'][:, :2]

        joint_cam_h36m = joint_cam_h36m - joint_cam_h36m[:1]

        joint_img, joint_cam = joint_img_h36m, joint_cam_h36m

        if flip:
            joint_img = flip_2d_joint(joint_img, self.MODEL_input_shape[1], self.flip_pairs)
        joint_cam = j3d_processing(joint_cam, rot, flip, self.flip_pairs)


        #  -> 0~1
        joint_img = joint_img[:, :2]
        joint_img /= np.array([[self.MODEL_input_shape[1], self.MODEL_input_shape[0]]])

        # normalize loc&scale
        mean, std = np.mean(joint_img, axis=0), np.std(joint_img, axis=0)
        joint_img = (joint_img.copy() - mean) / std

        joint_img_n = np.hstack((joint_img,np.reshape(joint_cam[:,2],(16,1))))

        # default valid
        joint_valid = np.ones((len(joint_cam), 1), dtype=np.float32)
        return joint_img, joint_cam, joint_valid


    def evaluate_joint(self, outs):
        print('Evaluation start...')
        annots = self.datalist
        assert len(annots) == len(outs)
        sample_num = len(annots)

        mpjpe = np.zeros((sample_num, len(self.human36_eval_joint)))
        pampjpe = np.zeros((sample_num, len(self.human36_eval_joint)))
        for n in range(sample_num):
            out = outs[n]
            annot = annots[n]

            # render materials
            pose_coord_out, pose_coord_gt = out['joint_coord'], annot['joint_cam']

            # root joint alignment
            pose_coord_out, pose_coord_gt = pose_coord_out - pose_coord_out[:1], pose_coord_gt - pose_coord_gt[:1]
            # sample eval joitns
            pose_coord_out, pose_coord_gt = pose_coord_out[self.human36_eval_joint, :], pose_coord_gt[self.human36_eval_joint, :]

            # pose error calculate
            mpjpe[n] = np.sqrt(np.sum((pose_coord_out - pose_coord_gt) ** 2, 1))
            # perform rigid alignment
            pose_coord_out = rigid_align(pose_coord_out, pose_coord_gt)
            pampjpe[n] = np.sqrt(np.sum((pose_coord_out - pose_coord_gt) ** 2, 1))

        # total pose error
        tot_err = np.mean(mpjpe)
        eval_summary = 'MPJPE (mm)    >> tot: %.2f\n' % (tot_err)
        print(eval_summary)

        tot_err = np.mean(pampjpe)
        eval_summary = 'PA-MPJPE (mm) >> tot: %.2f\n' % (tot_err)
        print(eval_summary)


dataset = Chracter('./characters/dataset','train')

print(dataset[0][0])
print(dataset[0][1])