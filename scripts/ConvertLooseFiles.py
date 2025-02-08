import open3d as o3d
from glob import glob
import numpy as np
from tqdm import tqdm
import os
import struct

# src_dir = input('src dir (e.g. /root/dataset/*.pcd): ')
# dst_dir = input('dst dir (e.g. /root/dataset/Scene0/0/): ')
# file_type = input('Dataset type: (pcd, kitti, nuscenes): ').lower()

file_type = 'kitti'
src_dirs = [f'/root/dataset/original_KITTI_Odometry/dataset/sequences/{i}/velodyne/*.bin' for i in range(11, 22)]
dst_dirs = [f'/root/dataset/TestSets/KITTI/Scene{i}/0/' for i in range(11, 22)]


def read_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file)
    xyz = np.array(pcd.points)
    return xyz  # N, 3


def read_nuscenes(file_path):
    pc_list = []
    with open(file_path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('fffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)  # N, 3


def read_kitti(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]  # (N,3)
    return points


if __name__ == '__main__':
    for src_dir, dst_dir in zip(src_dirs, dst_dirs):
        os.makedirs(dst_dir, exist_ok=True)
        for file in tqdm(glob(src_dir)):
            if (file_type == 'pcd'):
                xyz = read_pcd(file)
            elif (file_type == 'nuscenes'):
                xyz = read_nuscenes(file)
            elif (file_type == 'kitti'):
                xyz = read_kitti(file)
            else:
                raise NotImplementedError(f'unknown file type {file_type}')
            path, name = os.path.split(file)
            new_file = os.path.join(dst_dir, os.path.splitext(name)[0])
            np.save(new_file, xyz)  # (N, 3)
