import os
import json
import argparse
import numpy as np
from typing import Literal
from glob import glob

import warnings  # 导入warnings模块，用于控制警告信息的显示
warnings.filterwarnings("ignore")  # 忽略所有的警告信息


parser = argparse.ArgumentParser(description='SLAM metric')  # 解析命令行参数
parser.add_argument('--src_root', '-src', type=str)
parser.add_argument('--kitti360_only', default=False, action='store_true')


def parse_evo(evo_response: str) -> dict:
    """从evo的返回值中解析ape指标"""
    result = dict()
    for line in evo_response.splitlines():
        line_strip = line.strip()
        if line_strip.startswith("mean"):
            result['mean'] = f'{line.split()[1]:>10s}'
        elif line_strip.startswith("max"):
            result['max'] = f'{line.split()[1]:>10s}'
        elif line_strip.startswith("min"):
            result['min'] = f'{line.split()[1]:>10s}'
        elif line_strip.startswith("median"):
            result['median'] = f'{line.split()[1]:>10s}'
        elif line_strip.startswith("rmse"):
            result['rmse'] = f'{line.split()[1]:>10s}'
        elif line_strip.startswith("sse"):
            result['sse'] = f'{line.split()[1]:>10s}'
        elif line_strip.startswith("std"):
            result['std'] = f'{line.split()[1]:>10s}'
    return result


def eval_ape(gt_file_path: str, pred_file_path: str, step_file_path: str,
             dataset_type: Literal['kitti', 'kitti360', 'mulran'],
             calib_file_path: str = '', out_dir: str = '') -> dict:
    """
    评估APE指标

    :param gt_file_path: 真实位姿文件路径
    :param pred_file_path: 预测位姿文件路径
    :param step_file_path: 预测位姿的序号
    :param dataset_type: 数据集类型，支持kitti与kitti360的位姿组织方式
    :param calib_file_path: 标定矩阵文件路径
    :param out_dir: 评估结果输出目录
    :return: 指标
    """
    # 读取位姿
    with open(gt_file_path, 'r') as f:
        gt_pose = f.readlines()
        gt_pose = np.array(list(filter(lambda x: len(x) > 0, [i.strip().split() for i in gt_pose])), dtype=np.float)
    with open(pred_file_path, 'r') as f:
        pred_pose = f.readlines()
        pred_pose = np.array(list(filter(lambda x: len(x) > 0, [i.strip().split() for i in pred_pose])), dtype=np.float)
    with open(step_file_path, 'r') as f:
        pred_step = f.readlines()
        pred_step = np.array(list(filter(lambda x: len(x) > 0, [i.strip().split() for i in pred_step])), dtype=np.int)
        pred_step = pred_step.flatten()

    # 对齐预测位姿与gt位姿的序号
    if dataset_type == 'kitti360':
        gt_step_file_name = os.path.basename(gt_file_path).replace('gt', 'idx')
        gt_step_file_path = os.path.join(os.path.dirname(gt_file_path), gt_step_file_name)
        with open(gt_step_file_path, 'r') as f:
            gt_step = f.readlines()
            gt_step = np.array(list(filter(lambda x: len(x) > 0, [i.strip().split() for i in gt_step])), dtype=np.int)
            gt_step = gt_step.flatten()
        inter_step = np.intersect1d(gt_step, pred_step)
        gt_mask = np.zeros_like(gt_step).astype(np.bool)
        pred_mask = np.zeros_like(pred_step).astype(np.bool)
        gt_i, pred_i = 0, 0
        for i in inter_step:
            while gt_step[gt_i] != i:
                gt_i += 1
            gt_mask[gt_i] = True
            while pred_step[pred_i] != i:
                pred_i += 1
            pred_mask[pred_i] = True
        gt_pose = gt_pose[gt_mask]
        pred_pose = pred_pose[pred_mask]
    else:
        gt_pose = gt_pose[pred_step]

    padding = np.array([[0, 0, 0, 1]], dtype=np.float)
    gt_pose_se3 = np.concatenate([gt_pose, padding.repeat(gt_pose.shape[0], axis=0)], axis=1).reshape(-1, 4, 4)
    pred_pose_se3 = np.concatenate([pred_pose, padding.repeat(pred_pose.shape[0], axis=0)], axis=1).reshape(-1, 4, 4)

    # 利用标定矩阵变换坐标系（如果需要）
    if calib_file_path != '':
        with open(calib_file_path, 'r') as f:
            calib = f.readlines()
        if dataset_type == 'kitti':
            calib = np.array(calib[-1][3:].strip().split(), dtype=np.float)
            calib = np.concatenate([calib.reshape(-1, 4), padding])
            left = np.einsum("...ij,...jk->...ik", np.linalg.inv(calib), gt_pose_se3)
            gt_pose_se3 = np.einsum("...ij,...jk->...ik", left, calib)
        elif dataset_type == 'kitti360':
            calib = np.array(list(filter(lambda x: len(x) > 0, [i.strip().split() for i in calib])), dtype=np.float)
            calib = np.concatenate([calib.reshape(-1, 4), padding])
            gt_pose_se3 = gt_pose_se3 @ np.linalg.inv(calib)[np.newaxis, :, :]
        else:
            raise ValueError

    # 首帧对齐
    pred_pose_se3 = (gt_pose_se3[:1] @ np.linalg.inv(pred_pose_se3[:1])) @ pred_pose_se3

    gt_pose = gt_pose_se3.reshape(-1, 16)[:, :12]
    pred_pose = pred_pose_se3.reshape(-1, 16)[:, :12]
    temp_gt_file_path = os.path.join(os.path.dirname(pred_file_path), '__temp_gt_file.txt')
    temp_pred_file_path = os.path.join(os.path.dirname(pred_file_path), '__temp_pred_file.txt')
    with open(temp_gt_file_path, 'w+') as f:
        for pose in gt_pose.tolist():
            f.write(' '.join([f'{i:.10f}' for i in pose]) + '\n')
    with open(temp_pred_file_path, 'w+') as f:
        for pose in pred_pose.tolist():
            f.write(' '.join([f'{i:.10f}' for i in pose]) + '\n')

    # 使用evo计算ape
    if out_dir == '':
        out_dir = os.path.dirname(pred_file_path)
    save_plot = os.path.join(out_dir, 'ape.jpg')
    ape_map = os.path.join(out_dir, 'ape_map.jpg')
    ape_raw = os.path.join(out_dir, 'ape_raw.jpg')
    out_file = os.path.join(out_dir, 'ape.txt')
    for file_path in [ape_map, ape_raw, out_file]:
        if os.path.exists(file_path):
            os.remove(file_path)  # delete plot file if exist
    command = f'evo_ape kitti -a --plot_mode xy {temp_gt_file_path} {temp_pred_file_path} --save_plot {save_plot}'
    ape_result = parse_evo(os.popen(command).read())
    with open(out_file, 'w+') as f:
        f.write(json.dumps(ape_result, indent=4))

    # 删除临时文件
    os.remove(temp_gt_file_path)
    os.remove(temp_pred_file_path)
    return ape_result


def main():
    args = parser.parse_args()
    pred_pose_root = args.src_root
    assert os.path.exists(pred_pose_root)
    # pred_pose_root = r'/data/fengbh/data/fengbh/DPM++/log_infer/kitti.250102.1'

    gt_pose_root_kitti = r'/data/fengbh/data/fengbh/DPM-AAAI/dataset/GroundTruthTraj/SemanticKITTI_poses_Lidar_SUMA'
    seq_name_kitti = ['08']

    seq_list = sorted(list(glob(os.path.join(pred_pose_root, '*'))))

    # KITTI
    for i, seq in enumerate(seq_list):
        gt_file_path = os.path.join(gt_pose_root_kitti, f'pose_lidar_{seq_name_kitti[i]}.gt.txt')
        pred_file_path = os.path.join(seq, r'trajectory.allframes.txt')
        step_file_path = os.path.join(seq, r'trajectory.allsteps.txt')
        results = eval_ape(gt_file_path, pred_file_path, step_file_path, dataset_type='kitti')
        print(f'KITTI {seq_name_kitti[i]}: {results}')


if __name__ == "__main__":  # 主程序入口
    main()  # 调用主函数

