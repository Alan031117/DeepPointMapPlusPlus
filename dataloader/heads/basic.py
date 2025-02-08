import os  # 导入os模块，用于文件路径操作
import numpy as np  # 导入numpy库并命名为np，用于数值计算和数组操作
from dataloader.transforms import PointCloud  # 从dataloader.transforms模块中导入PointCloud类，用于表示点云数据

class PointCloudReader:  # 定义一个名为PointCloudReader的类，用于读取不同格式的点云文件
    optional_type = ['npz', 'npy', 'bin', 'pcd']  # 定义一个类变量optional_type，列出支持的文件类型

    def __init__(self):  # 定义初始化方法
        pass  # 占位符语句，不执行任何操作

    def __call__(self, file_path: str) -> PointCloud:  # 定义__call__方法，使实例可以像函数一样被调用
        """
        读取源文件
        :param file_path: 文件路径
        """
        # 从文件中加载点云数据，包括坐标、旋转、平移、法线、标签、图像和投影信息
        xyz, rotation, translation, norm, label, image, uvd = self._load_pcd(file_path)
        # 使用加载的数据初始化PointCloud对象
        pcd = PointCloud(xyz=xyz, rotation=rotation, translation=translation,
                         norm=norm, label=label, image=image, uvd=uvd)
        return pcd  # 返回PointCloud对象

    def _load_pcd(self, file_path):
        """从源文件读取点云数据"""
        file_type = os.path.splitext(file_path)[-1][1:]  # 获取文件的扩展名
        assert file_type in self.optional_type, f'Only type of the file in {self.optional_type} is optional, ' \
                                                f'not \'{file_type}\''  # 断言文件类型在optional_type列表中，如果不在，抛出带有错误信息的异常
        if file_type == 'npy':  # 如果文件类型是npy
            xyz = np.load(file_path)  # 加载npy文件中的点云数据xyz，形状为(N, 3)
            rotation = None  # npy文件不包含旋转矩阵，因此将rotation设置为None
            translation = None  # npy文件不包含平移向量，因此将translation设置为None
            norm = None  # npy文件不包含法线信息，因此将norm设置为None
            label = None  # npy文件不包含标签信息，因此将label设置为None
            image = None  # npy文件不包含图像信息，因此将image设置为None
            uvd = None  # npy文件不包含投影信息，因此将uvd设置为None

        elif file_type == 'npz':  # npz文件
            with np.load(file_path, allow_pickle=True) as npz:  # 使用numpy的load函数加载npz文件，并允许文件内对象被pickle序列化
                npz_keys = npz.files  # 获取npz文件中的所有键名
                assert 'lidar_pcd' in npz_keys, 'pcd file must contains \'lidar_pcd\''  # 断言'lidar_pcd'键必须存在，否则抛出异常
                xyz = npz['lidar_pcd']  # 从npz文件中读取点云数据xyz，对应键为'lidar_pcd'，其形状为(N, 3)，数据类型为f32（32位浮点数）
                rotation = npz['ego_rotation'] if 'ego_rotation' in npz_keys else None  # 从npz文件中读取自车旋转矩阵rotation，键为'ego_rotation'，如果键不存在则为None，形状为(3, 3)，f32
                translation = npz['ego_translation'] if 'ego_translation' in npz_keys else None  # 从npz文件中读取自车平移向量translation，键为'ego_translation'，如果键不存在则为None，形状为(3, 1)，f32
                norm = npz['lidar_norm'] if 'lidar_norm' in npz_keys else None  # 从npz文件中读取点云法线数据norm，键为'lidar_norm'，如果键不存在则为None，形状为(N, 3)，f32
                label = npz['lidar_seg'] if 'lidar_seg' in npz_keys else None  # 从npz文件中读取点云分割标签label，键为'lidar_seg'，如果键不存在则为None，形状为(N, 3)，f32
                image = None  # npz文件不包含图像信息，因此将image设置为None
                uvd = None  # npz文件不包含投影信息，因此将uvd设置为None

        elif file_type == 'bin':  # 如果文件类型是bin
            xyz = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]  # 从bin文件中加载点云数据xyz，并将其形状调整为(N, 3)
            rotation = None  # bin文件不包含旋转矩阵，因此将rotation设置为None
            translation = None  # bin文件不包含平移向量，因此将translation设置为None
            norm = None  # bin文件不包含法线信息，因此将norm设置为None
            label = None  # bin文件不包含标签信息，因此将label设置为None
            image = None  # bin文件不包含图像信息，因此将image设置为None
            uvd = None  # bin文件不包含投影信息，因此将uvd设置为None
        elif file_type == 'pcd':  # 如果文件类型是pcd
            import open3d  # 导入open3d库，用于处理pcd文件
            pcd = open3d.io.read_point_cloud(file_path)  # 使用open3d读取pcd文件中的点云数据
            xyz = np.asarray(pcd.points)  # 将pcd文件中的点云数据转换为numpy数组
            xyz = xyz[np.sum(np.isnan(xyz), axis=-1) == 0]  # 过滤掉包含NaN值的点，保留有效点
            rotation = None  # pcd文件不包含旋转矩阵，因此将rotation设置为None
            translation = None  # pcd文件不包含平移向量，因此将translation设置为None
            norm = None  # pcd文件不包含法线信息，因此将norm设置为None
            label = None  # pcd文件不包含标签信息，因此将label设置为None
            image = None  # pcd文件不包含图像信息，因此将image设置为None
            uvd = None  # pcd文件不包含投影信息，因此将uvd设置为None
        else:  # 如果文件类型不在optional_type列表中
            raise ValueError  # 抛出一个ValueError异常，表示不支持该文件类型

        return xyz, rotation, translation, norm, label, image, uvd  # 返回加载的点云数据和其他相关信息


