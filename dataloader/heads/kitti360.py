import os
import numpy as np
from dataloader.heads.basic import PointCloudReader


class Kitti360Reader(PointCloudReader):

    def __init__(self):
        super().__init__()

    def _load_pcd(self, file_path):
        """从源文件读取"""
        raise NotImplementedError

        xyz = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]  # (N, 3)
        rotation = None
        translation = None
        norm = None
        label = None
        image = None
        uvd = None

        return xyz, rotation, translation, norm, label, image, uvd