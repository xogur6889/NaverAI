import json
import os
import sys

if __name__ == '__main__':
    # video_name = '0001'
    video_name = sys.argv[-1]
    base_path = r'C:\Users\na\Desktop\Pytorch_Generalized_3D_Lane_Detection-master'
    world_pos_result_path = os.path.join(base_path, 'world_pos_result', video_name + '.json')

    with open(world_pos_result_path, 'r') as f:
        world_pos_result = json.load(f)
        world_pos_result = world_pos_result['frames']

    # Record each car start and disappear frame
    id_strart_end = {}
    for i in range(len(world_pos_result)):
        for other in world_pos_result[i]['others']:
            j = other['id']
            if j not in id_strart_end.keys():
                id_strart_end[j] = {}
                id_strart_end[j]['start'] = i
                id_strart_end[j]['end'] = i
            id_strart_end[j]['end'] = i

    lab_2d_result_path = os.path.join(base_path, 'result_2d', video_name)
    lab_2d_files = os.listdir(lab_2d_result_path)
    # jsonfile.sort(key=lambda x: int(x.split("\\")[-1].split('_')[1]))

    for file_name in lab_2d_files:
        with open(os.path.join(lab_2d_result_path, file_name), 'r') as f:
            frame_json = json.load(f)
        frame_json['info']['3D_version'] = '3D_release_22-05-30'
        frame_json['info']['4D_version'] = 'N/A'
        i = frame_json['info']['frame_index']
        info = {'frame_rate': frame_json['info']['frame_rate'], 'frame_index': frame_json['info']['frame_index'], 'video_size': frame_json['info']['video_size'],
                'metadata_version': frame_json['info']['metadata_version'], 'clip_version': frame_json['info']['clip_version'], '2D_version': frame_json['info']['2D_version'],
                '3D_version': frame_json['info']['3D_version'], '4D_version': frame_json['info']['4D_version']}
        frame_json['info'] = info

        frame_json['3D'] = {}
        json_3d = frame_json['3D']
        json_3d['ego'] = {}
        json_3d_ego = json_3d['ego']
        json_3d_ego['base_world_position'] = {'x': world_pos_result[i]['y'], 'y': world_pos_result[i]['x'], 'z': 0}

        before_5sec = i - int(frame_json['info']['frame_rate']) * 5
        after_5sec = i + int(frame_json['info']['frame_rate']) * 5 + 1
        json_3d_ego['base_world_trajectory'] = []

        for traj_idx in range(0 if before_5sec < 0 else before_5sec, after_5sec if after_5sec < len(lab_2d_files) else len(lab_2d_files)):
            json_3d_ego['base_world_trajectory'].append({'frame_index': traj_idx, 'position': {'x': world_pos_result[traj_idx]['y'], 'y': world_pos_result[traj_idx]['x'], 'z': 0}})

        json_3d['static'] = {}
        json_3d_static = json_3d['static']
        json_3d_static['roads'] = {}
        json_3d_static_road = json_3d_static['roads']
        json_3d_static_road['driveways'] = []
        json_3d_static_road['lanes'] = []
        json_3d_static_road['signs'] = []
        json_3d_static_road['sidewalks'] = []

        line_idx = 0
        for line in world_pos_result[i]['driveways']:
            lane_position = []
            for point in line['points']:
                lane_position.append({'x': point['y'], 'y': point['x'], 'z': 0})
            json_3d_static_road['driveways'].append({'id': '3Dr' + '{0:03d}'.format(line_idx), 'position': lane_position})
            line_idx += 1

        line_idx = 0
        for line in world_pos_result[i]['sidewalks']:
            lane_position = []
            for point in line['points']:
                lane_position.append({'x': point['y'], 'y': point['x'], 'z': 0})
            json_3d_static_road['sidewalks'].append({'id': '3Dr' + '{0:03d}'.format(line_idx), 'position': lane_position})
            line_idx += 1

        line_idx = 0
        for line in world_pos_result[i]['lines']:
            # "single_white": 흰색 실선
            # "dashed_white": 흰색 점선
            # "single_yellow":황색 실선
            # "dashed_yellow":황색 점선
            # "single_blue":청색 실선
            # "dashed_blue":청색 점선
            # t - 0: 실선, 1: 점선
            # c - 0: 흰색, 1: 황색, 2: 청색
            line_type = ''
            if line['t'] == 0:
                line_type += 'single_'
            elif line['t'] == 1:
                line_type += 'dashed_'
            else:
                print('라인 종류 에러')

            if line['c'] == 0:
                line_type += 'white'
            elif line['c'] == 1:
                line_type += 'yellow'
            elif line['c'] == 2:
                line_type += 'blue'
            else:
                print('라인 색상 에러')
            lane_position = []
            for point in line['points']:
                lane_position.append({'x': point['y'], 'y': point['x'], 'z': 0})
            json_3d_static_road['lanes'].append({'id': '3Dr' + '{0:03d}'.format(line_idx), 'type': line_type, 'position': lane_position})

            line_idx += 1

        json_3d_static['objects'] = {}
        json_3d_static_objects = json_3d_static['objects']
        json_3d_static_objects['objects'] = []

        json_3d_static_objects['signals'] = []
        for signal in world_pos_result[i]['signals']:
            signal_data = {'id': '3Ds' + '{0:03d}'.format(signal['id']), 'type': 'traffic_signal', 'position': {'x': signal['y'], 'y': signal['x'], 'z': 4.5}, 'crashed': False}
            json_3d_static_objects['signals'].append(signal_data)

        json_3d['dynamic'] = []
        json_3d_dynamic = json_3d['dynamic']

        for other in world_pos_result[i]['others']:
            other_data = {'id': '3Dd' + '{0:03d}'.format(other['id']), 'frame': {'creating_frame': id_strart_end[other['id']]['start'], 'destroying_frame': id_strart_end[other['id']]['end']},
                          'color': 'red', 'type': 'Unknown'}
            if other['t'] == 0:
                other_data['type'] = 'car'
            elif other['t'] == 1:
                other_data['type'] = 'bus'
            other_data['speed'] = 0
            other_data['position'] = {'x': other['y'], 'y': other['x'], 'z': 0}
            other_data['trajectory'] = []

            for traj_idx in range(0 if before_5sec < 0 else before_5sec, after_5sec if after_5sec < len(lab_2d_files) else len(lab_2d_files)):
                # 다른 차량 중에서 지금 찾는게 있는지
                for n_other in world_pos_result[traj_idx]['others']:
                    if other['id'] == n_other['id']:
                        other_data['trajectory'].append(
                            {'frame_index': traj_idx, 'position': {'x': n_other['y'], 'y': n_other['x'], 'z': 0}})

            other_data['volume'] = {'height': 0, 'width': 0, 'length': 0}
            other_data['crashed'] = False
            json_3d_dynamic.append(other_data)

        # eye car 3D location
        frame_json['accidents'] = {}

        # write json file
        # frame_json = json.dumps(frame_json, indent=4, ensure_ascii=False)
        frame_json = json.dumps(frame_json, indent=4)
        save_path = os.path.join(base_path, 'result_3d', video_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, video_name + "_{0:05d}_3D".format(i) + ".json"), "w") as f:
            f.write(frame_json)
