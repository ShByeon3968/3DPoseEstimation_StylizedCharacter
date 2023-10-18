import os
import sys
import time
import math
import numpy as np
import cv2
import shutil
from collections import OrderedDict

import torch
import torch.optim as optim
import matplotlib.pyplot as plt



def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()


def stop():
    sys.exit()


def check_data_pararell(train_weight):
    new_state_dict = OrderedDict()
    for k, v in train_weight.items():
        name = k[7:]  if k.startswith('module') else k  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer(model,optimizer_type,lr=0.001,momentum=0.09):
    optimizer = None
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum = momentum
        )
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=lr
        )
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr
        )

    return optimizer


# def get_scheduler(optimizer,scheduler:str):
#     scheduler = None
#     if scheduler == 'step':
#         scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.lr_step, gamma=cfg.TRAIN.lr_factor)
#     elif scheduler == 'platue':
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.TRAIN.lr_factor, patience=10, min_lr=1e-5)

#     return scheduler


# def save_checkpoint(states, epoch, is_best=None):
#     file_name = f'checkpoint{epoch}.pth.tar'
#     output_dir = cfg.checkpoint_dir
#     if states['epoch'] == cfg.TRAIN.end_epoch:
#         file_name = 'final.pth.tar'
#     torch.save(states, os.path.join(output_dir, file_name))

#     if is_best:
#         torch.save(states, os.path.join(output_dir, 'best.pth.tar'))


def load_checkpoint(load_dir, epoch=0, pick_best=False):
    try:
        print(f"Fetch model weight from {load_dir}")
        checkpoint = torch.load(load_dir, map_location='cuda')
        return checkpoint
    except Exception as e:
        raise ValueError("No checkpoint exists!\n", e)


