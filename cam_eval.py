"""
A demo for Gen-LaneNet with new anchor extension. It predicts 3D lanes from a single image.

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: March, 2020
"""
import sys
import json
import math
import random

import pandas
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms

from networks.LaneNet3D_ext import VggEncoder, RoadPlanePredHead
from tools.utils import *


class CamNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Define network
        self.im_encoder = VggEncoder(batch_norm=True)
        self.road_plane_pred_head = RoadPlanePredHead(540, 960, batch_norm=False)

    def forward(self, rgb_tensor):
        _, _, _, x4 = self.im_encoder(rgb_tensor)
        pred_cam = self.road_plane_pred_head(x4)
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
        model = torch.load('C:/Users/na/Desktop/Pytorch_Generalized_3D_Lane_Detection-master/cam_model.pt', map_location='cuda:0')
    except Exception as e:
        print(e)
        model = CamNet()

    model = model.cuda()

    # 5*1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
    model.eval()
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    pitch_range = range(-10, 11)
    loss = 0
    with torch.no_grad():
        for _ in range(10):
            select_pitch = random.choice(pitch_range)
            learning_info = {'folder_name': '1 ' + str(select_pitch), 'ego_x_s': ego_x_s, 'ego_z_s': ego_z_s}
            folder_path = os.path.join(ivh_data_path, learning_info['folder_name'])
            degree = 0
            test_data_set = os.listdir(folder_path)[-31:]
            # test_data_set = os.listdir(folder_path)
            for test_img in test_data_set:
                # start_img = random.choice(os.listdir(folder_path)[:-31])
                learning_info['start_img'] = test_img[:-4]
                img_path = os.path.join(folder_path, test_img)
                with open(img_path, 'rb') as f:
                    image = (Image.open(f).convert('RGB'))

                image = F.resize(image, size=[540, 960], interpolation=Image.BILINEAR)
                image = to_tensor(image)
                image = normalize(image)
                image = image.cuda(non_blocking=True)
                image = image.contiguous().float()
                image = image.unsqueeze(0)

                rgb = cv2.imread(img_path)
                p_cam_pitch = model(image)
                degree += p_cam_pitch * 180 / math.pi
            degree /= len(test_data_set)
            print('gt pitch : ', select_pitch)
            print('pr pitch : ', degree.item())
            print(flush=True)
