import re
import os

import torch
from plyfile import PlyData, PlyElement
import numpy as np

from tqdm import tqdm

frame_data_path = '/data/fengbh/data/fengbh/DPM-AAAI/dataset/KITTI360'
label_path = '/data/fengbh/data/fengbh/DPM-AAAI/dataset/KITTI360label/NEW_KITTI360label'
output_path = '/data/fengbh/data/fengbh/DPM-AAAI/dataset/KITTI360_SS'

calib_cam_to_velo = torch.tensor(
[[0.04307104361, -0.08829286498, 0.995162929, 0.8043914418,],
 [-0.999004371, 0.007784614041, 0.04392796942, 0.2993489574,],
 [-0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824,],
 [0, 0, 0, 1]])
calib_velo_to_cam = calib_cam_to_velo.inverse()

calib_cam_to_pose = torch.tensor(
[[0.0371783278, -0.0986182135, 0.9944306009, 1.5752681039,],
 [0.9992675562, -0.0053553387, -0.0378902567, 0.0043914093,],
 [0.0090621821, 0.9951109327, 0.0983468786, -0.6500000000, ],
 [0, 0, 0, 1]])

calib_pose_to_velo = calib_cam_to_velo @ calib_cam_to_pose.inverse()

tr_imu_to_world = torch.tensor([
    [0.6804458798, -0.7314378405, 0.0446328361, 3639.236961,],
    [-0.7298356009, -0.6819080306, -0.0483883609, 1708.205377,],
    [0.0658285676, 0.000351028, -0.9978308857, 122.056814,],
    [0, 0, 0, 1]
])
tr_cam_to_world = torch.tensor([
    [-0.705111, -2.610952e-02, 0.708616, 3.640277e+03,],
    [-0.709097, 2.470034e-02, -0.704678, 1.707084e+03,],
    [0.000896, -9.993541e-01, -0.035930, 1.228091e+02,],
    [0.000000, 0, 0.000000, 1,]
])

global_offical = tr_imu_to_world @ calib_cam_to_pose @ calib_cam_to_velo.inverse()
global_mine = tr_cam_to_world @ calib_cam_to_velo.inverse()


torch.cuda.set_device(0)


def validate_label_filename(filename):
    pattern = r'^\d{10}_\d{10}\.ply$'  # 文件名格式：0000000834_0000001286.ply
    if re.match(pattern, filename):
        return True
    else:
        return False


def split_start_end_frame_id(filename):
    pattern = r'(\d{10})_(\d{10})\.ply$'
    match = re.search(pattern, filename)
    if match:
        number1 = int(match.group(1))  # 第一个捕获组
        number2 = int(match.group(2))
        return number1, number2
    else:
        return None, None


def collect_npz_file_num():
    total_num = 0
    for root, dirs, files in os.walk(frame_data_path):
        for dir_name in dirs:
            sub_dir_path = os.path.join(root, dir_name)  # 2013_05_28_drive_0000_sync
            npz_data_path = os.path.join(sub_dir_path, '0')
            for _, _, cur_files in os.walk(npz_data_path):
                for file in cur_files:
                    if file.lower().endswith('.npz'):
                        total_num += 1
    return total_num


def main():
    total_npz_num = collect_npz_file_num()

    with tqdm(total=total_npz_num, desc="总进度") as pbar:
        current_progress = 0

        for root, dirs, files in os.walk(frame_data_path):
            dirs.remove('2013_05_28_drive_0008_sync')
            dirs.remove('2013_05_28_drive_0018_sync')
            dirs.remove('_check')
            for dir_name in dirs:
                sub_dir_path = os.path.join(root, dir_name)  # 2013_05_28_drive_0000_sync
                npz_data_path = os.path.join(sub_dir_path, '0')

                dynamic_label_path = os.path.join(label_path, dir_name, 'dynamic')
                static_label_path = os.path.join(label_path, dir_name, 'static')

                new_npz_data_path = os.path.join(output_path, dir_name, '0')  # 保存新npz文件的路径，与frame_data_path结构要相同
                if not os.path.exists(new_npz_data_path):
                    os.makedirs(new_npz_data_path, exist_ok=True)

                for _, _, static_files in os.walk(static_label_path):
                    for static_file in static_files:
                        if validate_label_filename(static_file):
                            start_frame, end_frame = split_start_end_frame_id(static_file)

                            # region step 1：载入序列label
                            static_label_file = os.path.join(static_label_path, static_file)
                            dynamic_label_file = os.path.join(dynamic_label_path, static_file)
                            # region step 1.1：载入static
                            ply_static_data = PlyData.read(static_label_file)
                            label_static_vertices = ply_static_data['vertex'].data.copy().tolist()
                            # endregion

                            # region step 1.2：载入dynamic
                            ply_dynamic_data = PlyData.read(dynamic_label_file)
                            label_dynamic_vertices = ply_dynamic_data['vertex'].data.copy().tolist()
                            # endregion

                            # region step 2：遍历帧数据
                            for frame in range(start_frame, end_frame + 1):
                                npz_file = os.path.join(npz_data_path, str(frame) + '.npz')
                                new_npz_file = os.path.join(new_npz_data_path, str(frame) + '.npz')
                                if not os.path.exists(npz_file):
                                    continue  # 不存在的文件直接跳过

                                # region step 2.1：将static和当前帧内的dynamic合并作为当前点集合
                                vertices = label_static_vertices.copy()
                                for dynamic_vertice in label_dynamic_vertices:
                                    timestamp = dynamic_vertice[9]
                                    if timestamp == frame:
                                        vertices.append((dynamic_vertice[0],
                                                         dynamic_vertice[1],
                                                         dynamic_vertice[2],
                                                         dynamic_vertice[3],
                                                         dynamic_vertice[4],
                                                         dynamic_vertice[5],
                                                         dynamic_vertice[6],
                                                         dynamic_vertice[7],
                                                         dynamic_vertice[8],
                                                         dynamic_vertice[10]))
                                # endregion

                                # region step 2.2：遍历npz中的所有点，在label点集合中进行匹配
                                npz_file = np.load(npz_file)
                                npz_data = {key: value for key, value in npz_file.items()}
                                global_xyz = (npz_data['ego_rotation'] @ npz_data['lidar_pcd'].T + npz_data['ego_translation']).T  # 此时已转换为世界坐标的点集合了

                                npz_data['ego_label'] = []  # 记录语义标签

                                # region ----------------------------------开始在GPU中处理------------------------------------
                                npz_tensor = torch.tensor(global_xyz, dtype=torch.float32)  # 单帧npz文件数据
                                map_tensor = torch.tensor(vertices, dtype=torch.float32)  # 地图
                                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                npz_tensor = npz_tensor.to(device)
                                map_tensor = map_tensor.to(device)
                                map_tensor_pos = map_tensor[:, :3]

                                # region 这块方案会爆显存
                                # distances = torch.norm(npz_tensor.unsqueeze(1) - map_tensor_pos.unsqueeze(0), dim=2)
                                # min_indices = torch.argmin(distances, dim=1)  # 找到每行（即每个 npz_tensor 元素）中最小距离的索引
                                # results = map_tensor[min_indices.unsqueeze(1), 6].squeeze(1)
                                # endregion

                                results = torch.zeros(npz_tensor.size(0), 1, dtype=torch.int, device=device)
                                for i in tqdm(range(npz_tensor.size(0)), desc="单个npz文件"):
                                    distances = torch.norm(npz_tensor[i:i + 1] - map_tensor_pos, dim=1)  # 计算当前 data1 元素与 data2_subset 所有元素之间的欧几里得距离
                                    min_index = torch.argmin(distances)  # 找到最小距离的索引
                                    results[i] = int(map_tensor[min_index, 6])

                                npz_data['ego_label'] = results.cpu()
                                # endregion  ----------------------------------开始在GPU中处理------------------------------------

                                # endregion

                                # region step 2.3：写入新数据
                                np.savez_compressed(new_npz_file, **npz_data)
                                # endregion

                                current_progress += 1
                                pbar.update(1)
                        # endregion

                        # endregion


if __name__ == '__main__':
    main()
