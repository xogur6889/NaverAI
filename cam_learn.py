"""
A demo for Gen-LaneNet with new anchor extension. It predicts 3D lanes from a single image.

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: March, 2020
"""
import math
import random

import cv2
import numpy as np
import torch

from tools.utils import *
import json
import pandas
from skimage.metrics import structural_similarity as ssim
import torch.nn as nn
from networks.LaneNet3D_ext import VggEncoder, RoadPlanePredHead
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms
import time

torch.set_num_threads(2)


class CamNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Define network
        self.im_encoder = VggEncoder(batch_norm=True)
        self.road_plane_pred_head = RoadPlanePredHead(540, 960, batch_norm=False)

    def forward(self, rgb_tensor):
        _, _, _, x4 = self.im_encoder(rgb_tensor)
        pred_cam = self.road_plane_pred_head(x4)
        # cam_height = 1 + pred_cam[:, 0]
        cam_pitch = pred_cam[:, 0]
        return cam_pitch


if __name__ == '__main__':
    ivh_data_path = r'C:\Users\na\Desktop\iVH_data'
    prob_data_path = r'C:\Users\na\Desktop\New Unity Project\cam_learn\prob'
    gt_data_path = r'C:\Users\na\Desktop\New Unity Project\cam_learn\gt'
    df = pandas.read_csv(os.path.join(ivh_data_path, 'GT_data.csv'))
    ego_x_s = [df['Ego_Y_position[m]'][0] - x for x in df['Ego_Y_position[m]']]
    ego_z_s = [z - df[' Ego_X_position[m]'][0] for z in df[' Ego_X_position[m]']]

    try:
        model = torch.load('cam_model.pt', map_location='cuda:0')
    except Exception as e:
        print(e)
        model = CamNet()

    model = model.cuda()

    # 5*1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
    model.train()
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    pitch_range = range(-10, 11)
    loss = 0
    for i in range(100000):
        select_pitch = random.choice(pitch_range)
        # select_pitch = 10
        learning_info = {'folder_name': '1 ' + str(select_pitch), 'ego_x_s': ego_x_s, 'ego_z_s': ego_z_s}
        folder_path = os.path.join(ivh_data_path, learning_info['folder_name'])
        start_img = random.choice(os.listdir(folder_path)[:-31])
        # start_img = os.listdir(folder_path)[0]
        learning_info['start_img'] = start_img[:-4]
        img_path = os.path.join(folder_path, start_img)
        with open(img_path, 'rb') as f:
            image = (Image.open(f).convert('RGB'))

        image = F.resize(image, size=[540, 960], interpolation=Image.BILINEAR)
        image = to_tensor(image)
        image = normalize(image)
        image = image.cuda(non_blocking=True)
        image = image.contiguous().float()
        # cv2.imshow('zz', image.permute(1, 2, 0).cpu().numpy())
        # cv2.waitKey(0)
        image = image.unsqueeze(0)

        rgb = cv2.imread(img_path)
        p_cam_pitch = model(image)
        degree = p_cam_pitch * 180 / math.pi

        learning_info['cam_height'] = 1
        learning_info['cam_pitch'] = degree.item()
        learning_info['gt_pitch'] = select_pitch

        # with open(os.path.join(ivh_data_path, 'LearningInfo', str(i)) + '.json', 'w') as json_file:
        #     json.dump(learning_info, json_file)
        #
        # prob_rgb = cv2.imread(os.path.join(prob_data_path, str(i) + '.jpg'))
        # gt_rgb = cv2.imread(os.path.join(gt_data_path, str(i) + '.jpg'))
        # while prob_rgb is None or gt_rgb is None:
        #     prob_rgb = cv2.imread(os.path.join(prob_data_path, str(i) + '.jpg'))
        #     gt_rgb = cv2.imread(os.path.join(gt_data_path, str(i) + '.jpg'))
        # prob_rgb[prob_rgb > 0] = 1
        # gt_rgb[gt_rgb > 0] = 1
        # loss += torch.abs(degree - select_pitch) + torch.tensor(math.sqrt(np.sum(prob_rgb - gt_rgb)))
        loss += (degree.squeeze(0) - select_pitch) ** 2
        if i == 0:
            continue
        if i % 3 == 0:
            print('epoch : ', i, 'loss : ', loss.item())
            # print('cam_pitch :', degree.item())
            optimizer.zero_grad()
            # loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            loss = 0
        if i % 10000 == 0:
            torch.save(model, 'cam_model.pt')
