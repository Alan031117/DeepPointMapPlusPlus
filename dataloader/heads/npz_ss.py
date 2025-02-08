import os  # 导入os模块，用于文件路径操作
import numpy as np  # 导入numpy库并命名为np，用于数值计算和数组操作
from dataloader.heads.basic import PointCloudReader  # 从dataloader.heads.basic模块中导入PointCloudReader类，这是一个点云读取器的基类

class NPZReader_W_SS(PointCloudReader):  # 定义一个继承自PointCloudReader的类NPZReader，用于读取npz格式的点云数据
    optional_type = ['npz']  # 定义一个类变量optional_type，指定可以处理的文件类型为'npz'

    def __init__(self, label_mapping: dict = None):  # 定义初始化方法
        super().__init__()  # 调用父类PointCloudReader的初始化方法
        if label_mapping is not None:
            label_mapping_keys = np.array([int(i) for i in label_mapping.keys()])
            label_max = max(label_mapping_keys)
            label_min = min(label_mapping_keys)
            self.label_mapping = np.ones(shape=(label_max - label_min + 100, ), dtype=np.int64) * -1
            self.label_mapping[label_mapping_keys - label_min] = list(label_mapping.values())
        else:
            self.label_mapping = None

    def _load_pcd(self, file_path):
        """从文件读取点云数据"""
        file_type = os.path.splitext(file_path)[-1][1:]  # 位于 file_path 文件的扩展名
        assert file_type in self.optional_type, f'Only type of the file in {self.optional_type} is optional, ' \
                                                f'not \'{file_type}\''  # 文件类型必须符合要求
        with np.load(file_path, allow_pickle=True) as npz:  # 使用 numpy.load 加载npz文件
            npz_keys = npz.files  # 获取 .npz 文件中所有数组名（键）
            assert 'lidar_pcd' in npz_keys, 'pcd file must contains \'lidar_pcd\''  # 'lidar_pcd'键必须存在
            xyz = npz['lidar_pcd']
            # 读取点云坐标，对应键为 'lidar_pcd'
            rotation = npz['ego_rotation'] if 'ego_rotation' in npz_keys else None
            # 读取自车旋转矩阵，键为'ego_rotation'
            translation = npz['ego_translation'] if 'ego_translation' in npz_keys else None
            # 读取自车平移向量，键为'ego_translation'
            norm = npz['lidar_norm'] if 'lidar_norm' in npz_keys else None
            # 读取点云法线数据，键为'lidar_norm'
            if 'lidar_seg' in npz_keys:
                label = npz['lidar_seg']
                if self.label_mapping is not None:
                    label = self.label_mapping[label]
            else:
                label = np.ones(shape=(xyz.shape[0],), dtype=np.int64) * -1
            # 读取点云分割标签，键为'lidar_seg'
            image = npz['image'] if 'image' in npz_keys else None
            # 读取图像数据，键为'image'
            uvd = npz['lidar_proj'] if 'lidar_proj' in npz_keys else None
            # 读取点云投影数据，键为'lidar_proj'

        return xyz, rotation, translation, norm, label, image, uvd  # 返回从npz文件中读取的所有数据
