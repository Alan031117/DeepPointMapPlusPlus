import os
import pickle
import colorlog as logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import math
import torch
import open3d as o3d
from typing import Dict, List, Tuple, Union
from tqdm import tqdm
import sys
sys.path.insert(1, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from system.modules.utils import PoseTool
from utils.pose import rt_global_to_relative_np


def batch_icp(src: List, dst: List, init_R: np.ndarray, init_T: np.ndarray, return_SE3: bool
              ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    icp修正刚体变换矩阵

    :param src: [pcd1, pcd2 ,..., pcdn]
    :param dst: [pcd1, pcd2 ,..., pcdn]
    :param init_R: (B, 3, 3)
    :param init_T: (B, 3, 1)
    :param return_SE3: True for return SE3, False for return (R, T)
    :return: (B, 3, 3), (B, 3, 1)
    """
    B = init_R.shape[0]
    num_src, num_dst = len(src), len(dst)
    init_SE3_list = np.repeat(np.eye(4)[np.newaxis, :, :], axis=0, repeats=B)
    init_SE3_list[:, :3, :3] = init_R
    init_SE3_list[:, :3, 3:] = init_T

    if num_src == 1 and num_dst > 1:
        src = src * num_dst
    elif num_dst == 1 and num_src > 1:
        dst = dst * num_src

    SE3_refinement = []
    for src_pcd, dst_pcd, init_SE3 in tqdm(zip(src, dst, init_SE3_list), total=B, leave=False, dynamic_ncols=True, desc='icp'):
        icp = o3d.pipelines.registration.registration_icp(
            source=src_pcd, target=dst_pcd, max_correspondence_distance=1.0, init=init_SE3,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
        icp_SE3 = icp.transformation
        delta_pose = np.linalg.inv(icp_SE3) @ init_SE3
        delta_R, delta_T = PoseTool.Rt(delta_pose)
        delta_angle = np.arccos((np.trace(delta_R) - 1) / 2).item() * 180 / math.pi
        delta_translation = np.linalg.norm(delta_T).item()

        _DEBUG = delta_angle > 5 or delta_translation > 2 if sys.platform == 'darwin' else False
        if _DEBUG:
            from utils.visualization import show_pcd
            src_xyz = np.asarray(src_pcd.points)
            dst_xyz = np.asarray(dst_pcd.points)
            src_init = (init_SE3[:3, :3] @ src_xyz.T + init_SE3[:3, 3:]).T
            src_icp = (icp_SE3[:3, :3] @ src_xyz.T + icp_SE3[:3, 3:]).T
            show_pcd([src_xyz, dst_xyz], [[1, 0, 0], [0, 1, 0]], window_name='origin')
            show_pcd([src_init, dst_xyz], [[1, 0, 0], [0, 1, 0]], window_name='gt')
            show_pcd([src_icp, dst_xyz], [[1, 0, 0], [0, 1, 0]],
                     window_name=f'icp: delta_angle={delta_angle:.2f}, delta_translation={delta_translation:.2f}')

        if delta_angle > 5 or delta_translation > 2:
            # SE3_refinement.append(init_SE3)
            # logger.warning('A suspected failed icp refinement has been discarded')
            SE3_refinement.append(icp_SE3)
            logger.warning('A suspected failed icp refinement has been used')
        else:
            SE3_refinement.append(icp_SE3)
    SE3_refinement = np.stack(SE3_refinement, axis=0).astype(np.float32)
    if return_SE3:
        return SE3_refinement
    else:
        R, T = SE3_refinement[:, :3, :3], SE3_refinement[:, :3, 3:]
        return R, T


def load_pcd(file_path):
    with np.load(file_path, allow_pickle=True) as npz:
        npz_keys = npz.files
        assert 'lidar_pcd' in npz_keys, 'pcd file must contains \'lidar_pcd\''
        xyz = npz['lidar_pcd']  # (N, 3), f32
        rotation = npz['ego_rotation'] if 'ego_rotation' in npz_keys else None  # (3, 3), f32
        translation = npz['ego_translation'] if 'ego_translation' in npz_keys else None  # (3, 1), f32

    dist = np.linalg.norm(xyz, axis=-1)
    mask = (dist > 5) & (dist < 100)
    xyz = xyz[mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd = pcd.voxel_down_sample(voxel_size=0.3)
    pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=3.0)

    return pcd, rotation, translation


def scene_refinement(file_list: List[str], frame_dis: np.ndarray, max_dist: float) -> Dict[Tuple[int, int], np.ndarray]:
    """
    获取一个场景内邻近帧之间修正后的相对位姿SE3

    :param file_list: 一个场景内的文件列表
    :param frame_dis: 一个场景内各帧之间的相对距离
    :param max_dist: 判定邻近帧的最近距离
    :return: Dict[(int, int), np.ndarray]
    """
    SE3_dict = dict()
    pcd_list = [None] * len(file_list)
    loop = tqdm(zip(file_list, frame_dis), total=len(file_list), leave=True, dynamic_ncols=True)
    for i, (file_path, dists) in enumerate(loop):
        if pcd_list[i] is None:
            pcd_center = load_pcd(file_path)
            pcd_list[i] = pcd_center
        else:
            pcd_center = pcd_list[i]
        id_others = np.nonzero(dists <= max_dist)[0].astype(np.int32)
        if len(id_others) <= 1:
            logger.warning(f'Found a frame(index={i}) without adjacent frames')
            continue
        id_others = id_others[id_others > i]  # 只计算j > i的情况，另一方向直接取逆
        if len(id_others) == 0:
            continue
        pcd_others = []
        for j in tqdm(id_others, leave=False, dynamic_ncols=True, desc='load pcd'):
            if pcd_list[j] is None:
                pcd_other = load_pcd(file_list[j])
                pcd_list[j] = pcd_other
            else:
                pcd_other = pcd_list[j]
            pcd_others.append(pcd_other)
        # 计算gt相对位姿，然后icp优化，并写入字典
        center_R, center_T = pcd_center[1][np.newaxis, :, :], pcd_center[2][np.newaxis, :, :]
        other_R = np.stack([pcd[1] for pcd in pcd_others], axis=0)
        other_T = np.stack([pcd[2] for pcd in pcd_others], axis=0)
        relative_R, relative_T = \
            rt_global_to_relative_np(center_R=center_R, center_T=center_T, other_R=other_R, other_T=other_T)
        center_pcd = [pcd_center[0]]
        other_pcd = [pcd[0] for pcd in pcd_others]
        refined_SE3 = batch_icp(src=other_pcd, dst=center_pcd, init_R=relative_R, init_T=relative_T, return_SE3=True)
        for j, SE3 in zip(id_others, refined_SE3):
            SE3_dict[(i, j)] = SE3
    return SE3_dict


def get_refined_SE3(dataset_list: List, frame_distance: List[List[torch.Tensor]], max_dist: float
                    ) -> List[List[Dict[Tuple[int, int], np.ndarray]]]:
    """
    获取所有场景内邻近帧之间修正后的相对位姿SE3

    :param dataset_list: 各数据集
    :param frame_distance: 各帧之间的相对距离
    :param max_dist: 判定邻近帧的最近距离
    :return: List[List[Dict[(int, int), np.ndarray]]]
    """
    datasets_refined_SE3 = []
    for i, dataset in enumerate(dataset_list):
        datasets_refined_SE3.append([])
        if 'carla' in dataset.name.lower():
            continue  # CARLA系列数据集为模拟数据，标签为绝对真值，不需要修正
        for j, scene in enumerate(dataset.scene_list):
            frame_files = []
            for agent in scene.agent_list:
                frame_files += agent.file_list
            # 检查场景目录下有无refined_SE3文件，若有则读取，否则计算后写入
            refined_SE3_file = os.path.join(scene.root, 'refined_SE3.pkl')
            if os.path.exists(refined_SE3_file):
                with open(refined_SE3_file, 'rb') as f:
                    refined_SE3: Dict[Tuple[int, int], np.ndarray] = pickle.load(f)
            else:
                refined_SE3 = scene_refinement(frame_files, frame_distance[i][j].numpy(), max_dist)
                with open(refined_SE3_file, 'wb+') as f:
                    pickle.dump(refined_SE3, f)
                logger.info(f'File \'refined_SE3\' has been saved in {refined_SE3_file}')

            # 测试icp优化结果
            # from utils.visualization import show_pcd
            # import random
            # for _ in range(10):
            #     i = random.randint(0, len(refined_SE3) - 1)
            #     key, value = list(refined_SE3.items())[i]
            #     src = load_pcd(frame_files[key[0]])
            #     dst = load_pcd(frame_files[key[1]])
            #     icp_SE3 = value
            #     src_xyz = np.asarray(src[0].points)
            #     dst_xyz = np.asarray(dst[0].points)
            #     relative_R, relative_T = \
            #         rt_global_to_relative_np(center_R=src[1], center_T=src[2], other_R=dst[1], other_T=dst[2])
            #     gt_SE3 = np.eye(4)
            #     gt_SE3[:3, :3] = relative_R
            #     gt_SE3[:3, 3:] = relative_T
            #     gt_dist = np.linalg.norm(relative_T.squeeze(axis=1))
            #     delta_pose = np.linalg.inv(icp_SE3) @ gt_SE3
            #     delta_R, delta_T = PoseTool.Rt(delta_pose)
            #     delta_angle = np.arccos((np.trace(delta_R) - 1) / 2).item() * 180 / math.pi
            #     delta_translation = np.linalg.norm(delta_T).item()
            #     dst_gt = (relative_R @ dst_xyz.T + relative_T).T
            #     dst_icp = (icp_SE3[:3, :3] @ dst_xyz.T + icp_SE3[:3, 3:]).T
            #     show_pcd([src_xyz, dst_xyz], [[1, 0, 0], [0, 1, 0]],
            #              window_name=f'origin: {key[0]} - {key[1]}, in {scene.root}')
            #     show_pcd([src_xyz, dst_gt], [[1, 0, 0], [0, 1, 0]], window_name=f'gt: dist={gt_dist:.2f}')
            #     show_pcd([src_xyz, dst_icp], [[1, 0, 0], [0, 1, 0]],
            #              window_name=f'icp: delta_angle={delta_angle:.2f}, delta_translation={delta_translation:.2f}')

            datasets_refined_SE3[-1].append(refined_SE3)
    return datasets_refined_SE3


if __name__ == '__main__':
    from dataloader.body import SlamDatasets
    from pipeline.parameters import *
    import yaml
    args = parser.parse_args()
    yaml_file = args.yaml_file
    # yaml_file = r'configs/train/__DeepPointMap_B.yaml'
    # yaml_file = r'configs/train/DeepPointMap_B+_mkSE3.yaml'
    with open(yaml_file, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, yaml.FullLoader)
    args = update_args(args, cfg)
    dataset = SlamDatasets(args)
    print(dataset)
    get_refined_SE3(dataset.dataset_list, dataset.frame_distance, 20)
