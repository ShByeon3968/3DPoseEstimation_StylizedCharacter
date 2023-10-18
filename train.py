from dataset import Chracter
from posenet import LinearModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import tqdm
import argparse
from utils.funcs_utils import get_optimizer

class CoordLoss(nn.Module):
    def __init__(self, has_valid=False):
        super(CoordLoss, self).__init__()

        self.has_valid = has_valid
        self.criterion = nn.L1Loss(reduction='mean')

    def forward(self, pred, target, target_valid):
        if self.has_valid:
            pred, target = pred * target_valid, target * target_valid

        loss = self.criterion(pred, target)

        return loss

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--epoch',type=int,default=50)
parser.add_argument('--num_joint',type=int)
args = parser.parse_args()


if __name__ == '__main__':
    device = 'cuda'
    print_freq = 10
    criterion = CoordLoss()
    model = LinearModel(args.num_joint)
    model.to(device)
    optimizer = get_optimizer(model,'rmsprop')
    dataset = Chracter(data_dir='./',mode='train')

    data_loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
    batch_generator = tqdm.tqdm(data_loader)

    for ep in range(args.epoch):
        for i, (img_joint, cam_joint, joint_valid) in enumerate(batch_generator):
            optimizer.zero_grad()
            img_joint, cam_joint = img_joint.to(device).float() , cam_joint.to(device).float()
            joint_valid = joint_valid.to(device).float()

            img_joint = img_joint.view(len(img_joint), -1)  # batch x (num_joint*2)
            pred_joint = model(img_joint)
            pred_joint = pred_joint.view(-1, args.num_joint, 3)

            loss = criterion(pred_joint,cam_joint,joint_valid)
            loss.backward()
            optimizer.step()
            
            if i % print_freq ==0:
                batch_generator.set_description(f'Epoch{args.epoch}_({i}/{len(batch_generator)}) => '       
                                                f'total loss: {loss.detach():.4f} ')
    print('\n')
    print('--model save--')
    torch.save(model.state_dict(),'./checkpoint/pose_model.pth')
    print('Done')
