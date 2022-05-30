"""
A demo for Gen-LaneNet with new anchor extension. It predicts 3D lanes from a single image.

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: March, 2020
"""
import math

import cv2
import numpy as np
import torch
import torch.optim
import glob
from tqdm import tqdm
from dataloader.Load_Data_3DLane_ext import *
from networks import GeoNet3D_ext, erfnet
from tools.utils import *
from tools.visualize_pred import lane_visualizer
import json
from model.lanenet.LaneNet import LaneNet

from networks.LaneNet3D_ext import VggEncoder, RoadPlanePredHead
import torch.nn as nn

fx = 2015.0  # focal length x (width)
fy = 2015.0  # focal length y (height)
cx = 960  # optical center x
cy = 540  # optical center y

torch.set_num_threads(2)


class CamNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Define network
        self.im_encoder = VggEncoder(batch_norm=True)
        # self.road_plane_pred_head = RoadPlanePredHead(256, 512, batch_norm=False)
        self.road_plane_pred_head = RoadPlanePredHead(540, 960, batch_norm=False)

    def forward(self, rgb_tensor):
        _, _, _, x4 = self.im_encoder(rgb_tensor)
        pred_cam = self.road_plane_pred_head(x4)
        # cam_height = 1 + pred_cam[:, 0]
        cam_pitch = pred_cam[:, 0]
        return cam_pitch


class RNNAgent(nn.Module):
    def __init__(self, input_shape, rnn_hidden_dim):
        super(RNNAgent, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, input_shape)
        self.hidden_state = None

    def init_hidden(self):
        # make hidden states on same device as model
        self.hidden_state = self.fc1.weight.new(self.rnn_hidden_dim).zero_()

    def forward(self, inputs):
        x = self.fc1(inputs)
        self.hidden_state = self.rnn(x.unsqueeze(0), self.hidden_state.reshape(-1, self.rnn_hidden_dim))
        q = self.fc2(self.hidden_state)
        return q


if __name__ == '__main__':
    # video_name = '0001'
    video_name = sys.argv[-1]
    th_avg_speed = 100
    base_path = r'C:\Users\na\Desktop\Pytorch_Generalized_3D_Lane_Detection-master'
    try:
        model = torch.load(os.path.join(base_path, 'cam_model.pt'), map_location='cuda:0')
    except Exception as e:
        print(e)
        model = CamNet()

    model = model.cuda()
    model.eval()
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ego car 포지션 읽기
    f = open(os.path.join(base_path, 'other_model_result', 'ego_track_' + video_name + '.txt'), 'r')
    ego_pos_data = f.readlines()
    f.close()

    ego_pos_xs = []
    ego_pos_zs = []
    for ego_pos in ego_pos_data:
        one_line_ego_data = ego_pos.split()
        ego_x, ego_z = one_line_ego_data[1], one_line_ego_data[3]
        ego_pos_xs.append(float(ego_x))
        ego_pos_zs.append(float(ego_z))

    # other_pos_data = []
    # other car 포지션 읽기
    f = open(os.path.join(base_path, 'other_model_result', video_name + '.txt'), 'r')
    other_pos_data = f.readlines()
    f.close()

    formatted_other_pos_data = []
    # 데이터 정리
    for other_pos_data_readline in other_pos_data:
        line_info = other_pos_data_readline.split()
        if line_info[2] != 'pedestrian':
            obj_type = 0
            if line_info[2] == 'car':
                obj_type = 0
            elif line_info[2] == 'bus':
                obj_type = 1
            frame_idx, obj_id, x, z, width, height, length = int(line_info[0]), int(line_info[1]), float(line_info[13]), float(line_info[15]), float(line_info[10]), float(line_info[11]), float(line_info[12])
            # other position 좀 크다.. 그대로 쓸지 말지
            x, z = x*0.6, z*0.6
            formatted_other_pos_data.append([frame_idx, obj_id, obj_type, x, z, width, height, length])

    lab_2d_result_path = os.path.join(base_path, 'result_2d', video_name)
    lab_2d_files = os.listdir(lab_2d_result_path)

    data = {'frames': []}
    og_x = 0.0
    og_z = 0.0

    model_name = 'C:/Users/na/Desktop/Pytorch_Generalized_3D_Lane_Detection-master/traffic_light_gru.pt'
    rnn_agent = torch.load(model_name, map_location='cpu')
    rnn_agent.init_hidden()
    max_signal_pixel_height = 20.0
    max_signal_pixel_height_threshold = 20.0
    max_signal_x = 0.0
    max_signal_z = 0.0
    max_signal_data_idx = 0
    odds_happen_sometime = 0

    degree = 0
    for file_name in lab_2d_files:
        with open(os.path.join(lab_2d_result_path, file_name), 'r') as f:
            frame_json = json.load(f)
        index = frame_json['info']['frame_index']
        format_index = "%05d" % index
        frame_rate = frame_json['info']['frame_rate']
        width, height = frame_json['info']['video_size'].split('x')
        width, height = float(width), float(height)
        with open(os.path.join(base_path, 'data/rgb', video_name, format_index + '.png'), 'rb') as f:
            image = (Image.open(f).convert('RGB'))

        image = F.resize(image, size=[540, 960], interpolation=Image.BILINEAR)
        image = to_tensor(image)
        image = normalize(image)
        image = image.cuda(non_blocking=True)
        image = image.contiguous().float()
        image = image.unsqueeze(0)
        with torch.no_grad():
            p_cam_pitch = model(image)
        degree += p_cam_pitch * 180 / math.pi

        # "single_white": 흰색 실선
        # "dashed_white": 흰색 점선
        # "single_yellow":황색 실선
        # "dashed_yellow":황색 점선
        # "single_blue":청색 실선
        # "dashed_blue":청색 점선
        # t - 0: 실선, 1: 점선
        # c - 0: 흰색, 1: 황색, 3: 청색
        screen_ratio = 0.7
        min_repeat_num = 15
        start_height = height * screen_ratio
        lines = []
        for lane in frame_json['2D']['static']['roads']['lanes']:
            t, c = lane['type'].split('_')
            if t == 'single':
                t = 0
            elif t == 'dashed':
                t = 1
            else:
                print('에러')

            if c == 'white':
                c = 0
            elif c == 'yellow':
                c = 1
            elif c == 'blue':
                c = 2
            else:
                print('에러')

            lane_xs = np.array(lane['lane_pos'])[:, 0]
            lane_ys = np.array(lane['lane_pos'])[:, 1]
            min_y_idx = np.argmin(lane_ys)
            points = []
            # v_xs = []
            # v_ys = []
            for j in range(min_repeat_num, min_repeat_num+5):
                for i in range(len(lane_ys)):
                    if i % 2 == 0:
                        if start_height - j < lane_ys[i] < start_height + j:
                            points.append({'x': int(lane_xs[i]), 'y': int(height - lane_ys[i])})
                            # v_xs.append(lane_xs[i])
                            # v_ys.append(height - lane_ys[i])
                if len(points) > 0:
                    break

            # if v_xs:
            #     points.append({'x': int(np.mean(v_xs)), 'y': int(np.mean(v_ys))})
            lines.append({'t': t, 'c': c, 'points': points})

        driveways = []
        for lane in frame_json['2D']['static']['roads']['driveways']:
            points = []
            for j in range(min_repeat_num, min_repeat_num+5):
                for seg in lane['segment']:
                    for point in seg:
                        if start_height - j < point[1] < start_height + j:
                            points.append({'x': int(point[0]), 'y': int(height - point[1])})
                if len(points) > 0:
                    break
            driveways.append({'points': points})

        sidewalks = []
        for lane in frame_json['2D']['static']['roads']['sidewalks']:
            points = []
            for j in range(min_repeat_num, min_repeat_num+5):
                for seg in lane['segment']:
                    for point in seg:
                        if start_height - j < point[1] < start_height + j:
                            points.append({'x': int(point[0]), 'y': int(height - point[1])})
                if len(points) > 0:
                    break
            sidewalks.append({'points': points})

        others = []
        cur_frame_ids = []
        for _, obj_id, _, _, _, _, _, _ in [d for d in formatted_other_pos_data if d[0] == index]:
            cur_frame_ids.append(obj_id)
        other_id_type_x_z_pairs = {}
        start_index = 0 if index-frame_rate < 0 else index-frame_rate
        end_index = len(lab_2d_files) if index+frame_rate+1 > len(lab_2d_files) else index+frame_rate+1
        for i in range(int(start_index), int(end_index)):
            others_one_frame_data = [d for d in formatted_other_pos_data if d[0] == i]
            for _, obj_id, obj_type, other_x, other_z, other_width, other_height, other_length in others_one_frame_data:
                # 현재 프레임에 있는 차량들만
                if obj_id in cur_frame_ids:
                    if obj_id not in other_id_type_x_z_pairs:
                        other_id_type_x_z_pairs[obj_id] = []
                    other_id_type_x_z_pairs[obj_id].append([obj_type, other_x, other_z, other_width, other_height, other_length])
        for obj_id in other_id_type_x_z_pairs:
            other_type_x_z_arr = np.array(other_id_type_x_z_pairs[obj_id])
            others.append({'id': obj_id, 't': other_type_x_z_arr[0, 0], 'x': other_type_x_z_arr[:, 1].mean(), 'y': other_type_x_z_arr[:, 2].mean(), 'w': other_width, 'h': other_height, 'l': other_length})

        signals = []
        check_signal_height_count = len(frame_json['2D']['static']['objects']['signals'])
        for signal in frame_json['2D']['static']['objects']['signals']:
            center_x = width * 0.5
            x1, x2, y1, y2 = signal['position']['x1'], signal['position']['x2'], signal['position']['y1'], signal['position']['y2']
            y_diff = y2 - y1
            if max_signal_pixel_height < y_diff:
                max_signal_pixel_height = y_diff
                max_signal_data_idx = index
                y1, y2 = (rnn_agent(torch.FloatTensor([y1 / height, y2 / height])) * height).squeeze(0)

                # 실제 신호등 세로 길이 : 1341.7322835 pixel = 355 mm
                # 물체 까지의 거리(pixel) = 실제 신호등 세로 길이(1341) * 초점 거리(f) / 이미지 상의 신호등 세로 길이
                real_sign_vert = 1341.7322835
                f = 1500
                real_dist_pixel = real_sign_vert * f / y2 - y1
                # 1pixel = 0.2645833333 mm
                # mm -> meter = * 0.001 (나누기 1000)
                z = real_dist_pixel * 0.2645833333 * 0.001
                # 실제 물체 x 위치(중점 으로 부터 pixel 거리) = 실제 거리(pixel) * 이미지 상 물체 x 위치(중점 으로 부터 pixel 거리) / 초점 거리(f)
                img_obj_x = ((x1 + x2) / 2) - center_x
                real_x_pixel = real_dist_pixel * img_obj_x / f
                x = real_x_pixel * 0.2645833333 * 0.001
                max_signal_x = x.item()
                max_signal_z = z.item()
            else:
                check_signal_height_count -= 1
        if check_signal_height_count == 0:
            # 가끔 탐지 누락 되는 걸 넘어 가게 해주기 위해서.
            odds_happen_sometime += 1
            if odds_happen_sometime > 10 and max_signal_pixel_height > max_signal_pixel_height_threshold:
                data['frames'][max_signal_data_idx - 1]['signals'] = [{'id': 0, 't': 0, 'x': max_signal_x, 'y': max_signal_z}]
                max_signal_pixel_height = max_signal_pixel_height_threshold
                odds_happen_sometime = 0
        # 검출이 됐으면 계속 우연 방지 초기화
        else:
            odds_happen_sometime = 0

        # 비율이 제 멋대로네.. 하.. + 후진 없다. + 조금이라도 차이나게
        new_x = np.mean(ego_pos_xs[int(start_index):int(end_index)]) * th_avg_speed
        new_z = np.mean(ego_pos_zs[int(start_index):int(end_index)]) * th_avg_speed
        # new_x = ego_pos_xs[index] * th_avg_speed
        # new_z = ego_pos_zs[index] * th_avg_speed
        if new_z < og_z:
            new_x = og_x
            new_z = og_z + 0.1
        data['frames'].append({'x': new_x, 'y': new_z, 'lines': lines, 'driveways': driveways, 'sidewalks': sidewalks, 'others': others, 'signals': signals})
        og_x = new_x
        og_z = new_z

    degree /= len(lab_2d_files)
    data['img_width'] = width
    data['img_height'] = height
    data['pitch'] = degree.item()
    with open(os.path.join(base_path, video_name + '.json'), 'w') as json_file:
        json.dump(data, json_file)
    print(video_name, ' 3D visual data generated')
