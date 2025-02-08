import os
import numpy as np
import os
import numpy as np
from Metrics.KittiOdometryMetric import KittiOdometryMetric
from Metrics.EvoMetric import eval_evo
from Metrics.RegistrationMetic import eval_reg

import argparse
parser = argparse.ArgumentParser(description='SLAM Metric')
parser.add_argument('--source', '-src',         type=str)
parser.add_argument('--ground_truth', '-gt',    type=str)
args = parser.parse_args()


def read_pose_txt(file_path):
    with open(file_path, 'r') as f:
        poses = f.readlines()
    poses_list = [np.concatenate([np.fromstring(trans, sep=' ').reshape(3, 4), np.array([0.0, 0.0, 0.0, 1.0]).reshape(1, 4)], axis=0) for trans in poses]
    return poses_list


def write_pose_txt(pose_list, file_path):
    with open(file_path, 'w+') as f:
        for pose in pose_list:
            pose_str = ' '.join([f'{float(i):.10f}' for i in pose[:3, :].flatten()]) + '\n'
            f.write(pose_str)
    return file_path


def main():
    gt_path = args.ground_truth
    pred_path = os.path.join(args.source, 'trajectory.allframes.txt')
    step_path_gt = None
    step_path_pred = os.path.join(args.source, 'trajectory.allsteps.txt')

    kitti_odometry_metric = KittiOdometryMetric(dataset='kitti')

    # metric_names = [
    #     'Seq', '多尺度分段平均误差(m/100m)', 'kitti_r_err(deg/100m)'
    # ]
    # print(', '.join([f'{i:>15s}' for i in metric_names]))

    cache_root = '__cache'
    os.makedirs(cache_root, exist_ok=True)

    case_name = pred_path.replace('\\', '/').split('/')[-3]

    poses_list_gt = [i for i in read_pose_txt(gt_path)]
    if (step_path_gt is None):
        poses_index_gt = list(range(len(poses_list_gt)))
    else:
        poses_index_gt = [int(i.strip()) for i in open(step_path_gt, 'r').readlines()]

    assert (len(poses_list_gt) == len(poses_index_gt))
    gt_step_pose = {k: v for k, v in zip(poses_index_gt, poses_list_gt)}

    poses_list_pred = [i for i in read_pose_txt(pred_path)]

    if (step_path_pred is None):
        poses_index_pred = list(range(len(poses_list_pred)))
    else:
        poses_index_pred = [int(i.strip()) for i in open(step_path_pred, 'r').readlines()]
    assert (len(poses_index_pred) == len(poses_list_pred))
    pred_step_pose = {k: v for k, v in zip(poses_index_pred, poses_list_pred)}

    valid_steps = sorted(list(set(gt_step_pose.keys()) & set(pred_step_pose.keys())))

    init_T = gt_step_pose[valid_steps[0]] @ np.linalg.inv(pred_step_pose[valid_steps[0]])

    cache_path = os.path.join(cache_root, case_name)

    pred_file_path = cache_path + '.pred.txt'
    gt_file_path = cache_path + '.gt.txt'

    write_pose_txt([i[1] for i in sorted(gt_step_pose.items(), key=lambda x: x[0]) if i[0] in valid_steps],
                   gt_file_path)
    write_pose_txt(
        [init_T @ i[1] for i in sorted(pred_step_pose.items(), key=lambda x: x[0]) if i[0] in valid_steps],
        pred_file_path)

    kitti_t_err, kitti_r_err, kitti_ate, kitti_rpe_t, kitti_rpe_r = kitti_odometry_metric.eval(gt_file=gt_file_path,
                                                                                               pred_file=pred_file_path,
                                                                                               fig_path=cache_root,
                                                                                               fig_name=case_name,
                                                                                               alignment=None)
    print('\n=======================================================\n')
    print(f'  多尺度分段平均误差(m/100m)：{kitti_t_err:.4f}')
    print(f'  同时定位与建图(SLAM)准确率为：{1-kitti_t_err/100:.2%}')
    print('\n=======================================================\n')
    # metric_vals = [
    #     kitti_t_err, kitti_r_err,
    # ]
    # assert len(metric_names) == len(metric_vals) + 1
    # print(f'{case_name[-12:]:>15s}, ' + ', '.join([f'{i:>15.2f}' for i in metric_vals]))


if __name__ == '__main__':
    main()


