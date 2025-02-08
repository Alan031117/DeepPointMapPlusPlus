import random  # 导入 random 模块，用于生成随机数
import math  # 导入 math 模块，提供数学函数
import torch  # 导入 PyTorch，用于张量操作和深度学习
from torch import Tensor as Tensor  # 从 PyTorch 导入 Tensor，并将其重命名为 Tensor，方便后续使用
import numpy as np  # 导入 numpy，用于数值计算
import open3d as o3d  # 导入 open3d 库，用于处理 3D 数据
import scipy.linalg as linalg  # 导入 scipy.linalg，用于线性代数计算
from scipy.spatial.transform import Rotation  # 从 scipy.spatial.transform 导入 Rotation 类，用于旋转矩阵和四元数转换
from typing import List, Sequence, Literal, Union  # 从 typing 模块导入类型注解工具

# 尝试从 pytorch3d 库中导入 knn_points、sample_farthest_points、ball_query、knn_gather 函数
try:
    from pytorch3d.ops import knn_points, sample_farthest_points, ball_query, knn_gather
    has_t3d = True  # 如果导入成功，设置 has_t3d 为 True，表示可以使用 pytorch3d 的相关操作
except:  # 如果导入失败
    has_t3d = False  # 设置 has_t3d 为 False，表示无法使用 pytorch3d 的相关操作


class PointCloud:  # 定义 PointCloud 类，用于表示点云数据（这个类的实例是一类点的集合）
    """点云类"""

    def __init__(
        self,
        xyz: Union[Tensor, np.ndarray],  # 初始化函数，接收点云的各种数据参数
        rotation: Union[Tensor, np.ndarray] = None,
        translation: Union[Tensor, np.ndarray] = None,
        norm: Union[Tensor, np.ndarray] = None,
        label: Union[Tensor, np.ndarray] = None,
        image: Union[Tensor, np.ndarray] = None,
        uvd: Union[Tensor, np.ndarray] = None,
    ) -> None:
        """
        :param xyz: (N, 3) 点云坐标
        :param rotation: (3, 3) 点云真实旋转矩阵
        :param translation: (3, 1) 点云真实平移矩阵
        :param norm: (N, 3) 法向量
        :param label: (N, 3) 分割标签
        :param image: 与点云匹配的图像
        :param uvd: (N, 3) 与图像的像素级对应关系
        """
        # 将所有输入参数保存到一个列表中，以便后续处理
        input_args = [xyz, rotation, translation, norm, label, image, uvd]
        for i in range(len(input_args)):  # 遍历每个输入参数
            arg = input_args[i]
            if arg is not None:  # 如果参数不为 None
                arg = torch.from_numpy(arg)  # 将 numpy 数组转换为 PyTorch 张量
                if arg.dtype == torch.float64:  # 如果张量的数据类型为双精度浮点数
                    arg = arg.float()  # 将双精度浮点数转换为单精度浮点数
                input_args[i] = arg  # 将转换后的张量保存回列表中
        xyz, rotation, translation, norm, label, image, uvd = input_args  # 将处理后的参数重新赋值给对应的变量

        self.xyz = xyz  # 保存点云的坐标
        self.nbr_point = xyz.shape[0]  # 获取点云中的点的数量
        self.device = self.xyz.device  # 获取张量所在的设备（CPU 或 GPU）

        # 如果旋转矩阵为 None，使用单位矩阵；否则使用提供的旋转矩阵
        self.R = rotation if rotation is not None else torch.eye(3, dtype=torch.float32, device=self.device)
        # 如果平移矩阵为 None，使用零矩阵；否则使用提供的平移矩阵
        self.T = translation if translation is not None else torch.zeros(size=(3, 1), dtype=torch.float32, device=self.device)

        self.calib = torch.eye(4, dtype=torch.float32, device=self.device)  # 初始化校准矩阵为 4x4 的单位矩阵

        self.norm = norm  # 保存法向量
        if norm is not None:  # 如果法向量不为 None
            self.has_norm = True  # 设置标志，表示点云具有法向量
        else:
            self.has_norm = False  # 设置标志，表示点云不具有法向量

        self.label = label  # 保存分割标签
        if label is not None:  # 如果分割标签不为 None
            self.has_label = True  # 设置标志，表示点云具有分割标签
        else:
            self.has_label = False  # 设置标志，表示点云不具有分割标签

        self.image = image  # 保存与点云匹配的图像
        if image is not None:  # 如果图像不为 None
            self.has_image = True  # 设置标志，表示点云具有匹配的图像
        else:
            self.has_image = False  # 设置标志，表示点云不具有匹配的图像

        self.uvd = uvd  # 保存与图像的像素级对应关系
        if uvd is not None:  # 如果 uvd 不为 None
            self.has_uvd = True  # 设置标志，表示点云具有像素级对应关系
        else:
            self.has_uvd = False  # 设置标志，表示点云不具有像素级对应关系

    # 以下方法都是变换函数内可能用到的工具
    def to_tensor(self, use_norm: bool = False, use_uvd: bool = False, use_label: bool = False, use_image: bool = False, use_calib: bool = False, padding_to: int = -1):
        """
        将点云数据和相关信息转换为张量格式，并根据需求返回不同的组合。

        :param use_norm: 是否使用点云的法线信息。如果为 True，则将法线信息包含在返回的张量中。
        :param use_uvd: 是否使用点云与图像的对应位置数据。如果为 True，则将 UVD 信息包含在返回的张量中。
        :param use_image: 是否返回与点云匹配的图像。如果为 True，则在返回结果中包含图像数据。
        :param use_calib: 是否返回校准矩阵。如果为 True，则在返回结果中包含校准矩阵。
        :param padding_to: 是否将点云数据填充到固定长度。如果值大于 0，则表示需要填充到指定长度，否则不进行填充。
        :return: 返回包含点云数据、旋转矩阵、平移矩阵、图像（可选）、填充掩码、校准矩阵（可选）等的元组。
        """
        constitution = [self.xyz]  # 初始化列表，将点云坐标 (xyz) 添加到列表中
        if use_norm and self.has_norm:  # 如果指定使用法线信息且点云包含法线信息
            constitution.append(self.norm)  # 将法线信息添加到 constitution 列表中
        if use_uvd and self.has_uvd:  # 如果指定使用 UVD 信息且点云包含 UVD 信息
            constitution.append(self.uvd)  # 将 UVD 信息添加到 constitution 列表中
        if use_label and self.has_label:  # 如果指定使用标签信息
            label = self.label.unsqueeze(1)
            constitution.append(label)  # 将标签信息添加到 constitution 列表中

        # constitution[1] = constitution[1].squeeze(-1)
        pcd = torch.concat(constitution, dim=1)  # 将列表中的张量在第 1 维（特征维度）上拼接，形成完整的点云数据

        if padding_to > 0:  # 如果指定了填充长度
            if self.nbr_point > padding_to:  # 检查点云的点数量是否超过了指定的填充长度
                raise RuntimeError(f'The number of Point Cloud ({self.nbr_point}) is greater than `padding_to` ({padding_to})')  # 如果超过，抛出异常
            padding = torch.zeros(size=(padding_to - self.nbr_point, pcd.shape[1]), device=self.device)  # 创建一个零填充张量，大小为 (填充点数, 特征数)
            pcd = torch.cat((pcd, padding), dim=0)  # 将填充张量与原始点云数据拼接，形成填充后的点云数据
            padding_mask = torch.zeros(size=(padding_to, ), dtype=torch.bool, device=self.device)  # 初始化填充掩码，全为 False
            padding_mask[self.nbr_point:] = True  # 将填充部分的掩码设置为 True
        else:  # 如果不需要填充
            padding_mask = torch.zeros(self.nbr_point, dtype=torch.bool, device=self.device)  # 创建一个与点云数量相同的掩码，全为 False

        if use_image:  # 如果指定要返回图像
            if self.has_image:  # 检查点云是否包含图像信息
                image = self.image  # 如果包含，使用点云中的图像
            else:
                image = torch.zeros(size=(1, ), device=self.device)  # 如果不包含，返回一个大小为 1 的零张量
            return pcd.T, self.R, self.T, image, padding_mask  # 返回点云数据（转置）、旋转矩阵、平移矩阵、图像和填充掩码
        elif use_calib:  # 如果指定要返回校准矩阵
            return pcd.T, self.R, self.T, padding_mask, self.calib  # 返回点云数据（转置）、旋转矩阵、平移矩阵、填充掩码和校准矩阵
        else:  # 如果不需要返回图像或校准矩阵
            return pcd.T, self.R, self.T, padding_mask  # 只返回点云数据（转置）、旋转矩阵、平移矩阵和填充掩码

    def apply_index(self, mask):  # 定义 apply_index 方法，基于给定的掩码（索引）筛选点云数据
        scalable_args = [self.xyz, self.norm, self.label, self.uvd]  # 创建一个包含点云坐标、法线、标签、UVD 信息的列表
        for i in range(len(scalable_args)):  # 遍历列表中的每个元素
            arg = scalable_args[i]
            if arg is not None:  # 如果当前元素不为 None
                scalable_args[i] = arg[mask]  # 使用掩码筛选该元素中的数据
        self.xyz, self.norm, self.label, self.uvd = scalable_args  # 将筛选后的数据赋值回类的属性
        self.nbr_point = self.xyz.shape[0]  # 更新点云中点的数量

    def to_gpu(self):  # 定义 to_gpu 方法，将所有相关数据转移到 GPU 上
        self.xyz, self.R, self.T, self.calib = self.xyz.cuda(), self.R.cuda(), self.T.cuda(), self.calib.cuda()  # 将点云坐标、旋转矩阵、平移矩阵和校准矩阵移动到 GPU
        if self.has_norm:  # 如果点云包含法线信息
            self.norm = self.norm.cuda()  # 将法线信息移动到 GPU
        if self.has_label:  # 如果点云包含标签信息
            self.label = self.label.cuda()  # 将标签信息移动到 GPU
        if self.has_image:  # 如果点云包含图像信息
            self.image = self.image.cuda()  # 将图像信息移动到 GPU
        if self.has_uvd:  # 如果点云包含 UVD 信息
            self.uvd = self.uvd.cuda()  # 将 UVD 信息移动到 GPU
        self.device = self.xyz.device  # 更新设备信息为 GPU

    def to_cpu(self):  # 定义 to_cpu 方法，将所有相关数据转移到 CPU 上
        self.xyz, self.R, self.T, self.calib = self.xyz.cpu(), self.R.cpu(), self.T.cpu(), self.calib.cpu()  # 将点云坐标、旋转矩阵、平移矩阵和校准矩阵移动到 CPU
        if self.has_norm:  # 如果点云包含法线信息
            self.norm = self.norm.cpu()  # 将法线信息移动到 CPU
        if self.has_label:  # 如果点云包含标签信息
            self.label = self.label.cpu()  # 将标签信息移动到 CPU
        if self.has_image:  # 如果点云包含图像信息
            self.image = self.image.cpu()  # 将图像信息移动到 CPU
        if self.has_uvd:  # 如果点云包含 UVD 信息
            self.uvd = self.uvd.cpu()  # 将 UVD 信息移动到 CPU
        self.device = self.xyz.device  # 更新设备信息为 CPU


class Compose:  # 定义一个类 Compose，用于组合和顺序执行多个点云数据的变换操作
    """组合多种变换"""

    def __init__(self, transforms: List):  # 初始化函数，接收一个变换列表作为参数
        self.transforms = transforms  # 将传入的变换列表保存为类的属性

    def __call__(self, pcd: PointCloud) -> PointCloud:  # 定义 __call__ 方法，使得 Compose 对象可以像函数一样被调用
        for t in self.transforms:  # 遍历所有的变换操作
            pcd = t(pcd)  # 将当前的点云对象 pcd 依次传入每一个变换操作中，更新 pcd
        return pcd  # 返回经过所有变换后的点云对象

    def __repr__(self) -> str:  # 定义 __repr__ 方法，返回该对象的字符串表示，便于打印输出和调试
        format_string = self.__class__.__name__ + "("  # 获取类名，并以字符串形式开始格式化输出
        for t in self.transforms:  # 遍历所有的变换操作
            format_string += "\n"  # 每个变换操作之前添加一个换行符
            format_string += f"    {t}"  # 将变换操作的字符串表示添加到输出字符串中，缩进4个空格以便阅读
        format_string += "\n)"  # 结束格式化字符串的构建
        return format_string  # 返回最终构建的字符串表示


class RandomChoice:  # 定义一个类 RandomChoice，用于从给定的变换列表中随机选择一个变换并应用到点云数据上
    """从给定的变换列表中随机应用一个"""

    def __init__(self, transforms, p=None):  # 初始化函数，接收变换列表和每个变换的权重（概率）
        if p is not None and not isinstance(p, Sequence):  # 如果提供了权重 p，但它不是一个序列（例如列表或元组）
            raise TypeError("Argument p should be a sequence")  # 抛出类型错误，提示权重 p 应该是一个序列
        self.transforms = transforms  # 将传入的变换列表保存为类的属性
        self.p = p  # 将传入的权重（概率）保存为类的属性

    def __call__(self, pcd: PointCloud) -> PointCloud:  # 定义 __call__ 方法，使得 RandomChoice 对象可以像函数一样被调用
        t = random.choices(self.transforms, weights=self.p)[0]  # 使用 random.choices 根据权重从变换列表中随机选择一个变换
        return t(pcd)  # 将点云数据 pcd 传递给选择的变换，并返回变换后的点云

    def __repr__(self) -> str:  # 定义 __repr__ 方法，返回该对象的字符串表示，便于打印输出和调试
        format_string = self.__class__.__name__ + "("  # 获取类名，并以字符串形式开始格式化输出
        for t in self.transforms:  # 遍历所有的变换操作
            format_string += "\n"  # 每个变换操作之前添加一个换行符
            format_string += f"    {t}"  # 将变换操作的字符串表示添加到输出字符串中，缩进4个空格以便阅读
        format_string += f"\n)(p={self.p})"  # 添加权重信息 p 的表示，结束格式化字符串的构建
        return format_string  # 返回最终构建的字符串表示


class GroundFilter:  # 定义一个类 GroundFilter，用于地面点的滤波
    """地面点滤波"""

    def __init__(self, img_len: int, img_width: int, grid_width: float, ground_height: float, preserve_sparse_ground: bool = True):  # 初始化函数，接收网格大小、地面高度阈值等参数
        """
        :param img_len: 网格长度方向数量
        :param img_width: 网格宽度方向数量
        :param grid_width: 单个网格的边长
        :param ground_height: 地面厚度阈值
        :param preserve_sparse_ground: 是否保留稀疏的地面点
        """
        self.img_len = img_len  # 保存网格长度方向的数量
        self.img_width = img_width  # 保存网格宽度方向的数量
        self.grid_width = grid_width  # 保存单个网格的边长
        self.ground_height = ground_height  # 保存地面高度的阈值
        self.preserve_sparse_ground = preserve_sparse_ground  # 保存是否保留稀疏地面点的选项

    def __call__(self, pcd: PointCloud) -> PointCloud:  # 定义 __call__ 方法，使得 GroundFilter 对象可以像函数一样被调用
        if self.ground_height <= 0:  # 如果地面高度阈值小于等于 0，则直接返回原始点云
            return pcd

        pointCloudsIn = pcd.xyz.cpu().clone().numpy()  # 将点云数据从 GPU 转移到 CPU，克隆并转换为 numpy 数组

        # 计算每个点所处的网格 id，去除不属于网格范围的点
        row_id = (pointCloudsIn[:, 0] / self.grid_width + self.img_len / 2).astype(np.int32)  # 计算点在行方向的网格 id
        col_id = (pointCloudsIn[:, 1] / self.grid_width + self.img_width / 2).astype(np.int32)  # 计算点在列方向的网格 id
        grid_id = row_id * self.img_width + col_id  # 将行列 id 转换为一维的网格 id
        dis_mask = (row_id >= 0) & (row_id < self.img_len) & (col_id >= 0) & (col_id < self.img_width)  # 生成一个掩码，过滤掉超出网格范围的点
        pointCloudsIn = pointCloudsIn[dis_mask]  # 过滤点云数据，保留在网格范围内的点
        grid_id = grid_id[dis_mask]  # 过滤网格 id
        remained_ids = np.nonzero(dis_mask)[0]  # 获取保留的点的索引

        # 按照 grid id 排序，相同 grid id 的点会被聚集在一起
        order = np.argsort(grid_id)  # 对 grid id 进行排序，获取排序索引
        pointCloudsIn = pointCloudsIn[order]  # 按排序索引对点云数据排序
        grid_id = grid_id[order]  # 按排序索引对网格 id 排序
        remained_ids = remained_ids[order]  # 按排序索引对保留的点的索引排序

        all_grid_id, all_grid_cnt = np.unique(grid_id, return_counts=True)  # 获取每个唯一的 grid id 以及对应的点数量
        grid_slices = np.cumsum(all_grid_cnt, axis=-1)  # 通过累加点数量，计算每个 grid 的范围
        non_ground_ids = []  # 初始化列表，用于保存非地面点的索引
        sparse_ground_ids = []  # 初始化列表，用于保存稀疏地面点的索引

        end = 0  # 初始化索引
        for grid_slice in grid_slices:  # 遍历每个 grid 的范围
            begin = end  # 记录当前 grid 的起始索引
            end = grid_slice  # 记录当前 grid 的结束索引
            if end - begin < 3:  # 如果当前 grid 的点数量小于 3，忽略该 grid
                continue
            grid_pcd = pointCloudsIn[begin:end, :]  # 获取当前 grid 的所有点
            grid_ids = remained_ids[begin:end]  # 获取当前 grid 的点对应的原始索引
            height_diff = grid_pcd[:, 2].max() - grid_pcd[:, 2].min()  # 计算当前 grid 内点的最大高度差

            # 如果高度差大于阈值，则认为该 grid 是非地面区域，将其索引保存到 non_ground_ids 列表中
            # 否则认为是地面区域，如果保留稀疏地面点选项开启，将该 grid 中的第一个点保存到 sparse_ground_ids 列表中
            if height_diff > self.ground_height:
                non_ground_ids.append(grid_ids)
            elif self.preserve_sparse_ground:
                sparse_ground_ids.append(grid_ids[:1])

        # 将所有的非地面点和稀疏地面点的索引合并为一个数组
        remained_ids = np.concatenate(non_ground_ids + sparse_ground_ids, axis=0)
        remained_ids = torch.from_numpy(remained_ids).to(pcd.device)  # 将 numpy 数组转换为 PyTorch 张量，并转移到点云数据所在的设备上
        pcd.apply_index(mask=remained_ids)  # 使用筛选出的索引过滤点云数据，保留非地面点和稀疏地面点
        return pcd  # 返回经过滤波后的点云数据


class OutlierFilter:  # 定义一个类 OutlierFilter，用于对点云数据进行离群点滤波
    """离群点滤波"""

    def __init__(self, nb_neighbors: int, std_ratio: float):  # 初始化函数，接收最近邻点的数量和标准差倍率上限
        """
        :param nb_neighbors: 考虑的最近邻点数量
        :param std_ratio: 滤波的标准差倍率上限
        """
        self.nb_neighbors = nb_neighbors  # 保存最近邻点数量，用于计算离群点
        self.std_ratio = std_ratio  # 保存标准差倍率上限，用于判断离群点

    def __call__(self, pcd: PointCloud) -> PointCloud:  # 定义 __call__ 方法，使得 OutlierFilter 对象可以像函数一样被调用
        if str(pcd.device).startswith('cuda') and has_t3d:  # 如果点云数据在 GPU 上且 pytorch3d 库可用
            # CUDA加速的torch3d实现具有最快速度
            pcd_xyz = pcd.xyz.unsqueeze(0)  # 将点云坐标增加一个维度，以匹配 knn_points 函数的输入要求
            knn = knn_points(p1=pcd_xyz, p2=pcd_xyz, K=self.nb_neighbors + 1, return_sorted=True, return_nn=False)  # 计算每个点的最近邻点
            dists = torch.sqrt(knn.dists.squeeze(0)[:, 1:])  # 计算最近邻点的欧几里得距离，并去除自身距离
            points_dist = dists.mean(1)  # 计算每个点到其最近邻点的平均距离
            mean = points_dist.mean()  # 计算所有点的平均距离
            std = points_dist.std()  # 计算所有点距离的标准差
            outlier_dist = mean + self.std_ratio * std  # 根据标准差倍率上限计算离群点的距离阈值
            mask = points_dist <= outlier_dist  # 生成一个掩码，标记出距离小于等于阈值的点
            pcd.apply_index(mask)  # 使用掩码过滤离群点，保留正常点
        else:  # 如果点云数据不在 GPU 上或者 pytorch3d 不可用
            # 非CUDA加速时使用open3d api
            pcd_o3d = o3d.geometry.PointCloud()  # 创建一个 Open3D 的 PointCloud 对象
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd.xyz.numpy())  # 将点云数据转换为 Open3D 格式
            mask = pcd_o3d.remove_statistical_outlier(nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio)[1]  # 使用 Open3D 的统计滤波方法计算离群点掩码
            pcd.apply_index(mask)  # 使用掩码过滤离群点，保留正常点

        return pcd  # 返回经过滤波后的点云数据


class LowPassFilter:  # 定义一个类 LowPassFilter，用于对点云数据进行低通滤波，保留简单的几何结构
    """低通滤波，保留简单几何结构"""

    def __init__(self, normals_radius: float, normals_num: int, filter_std: float, flux: int = 2, max_remain: int = -1):  # 初始化函数，接收用于法线估计、滤波的参数
        """
        :param normals_radius: 估计法线的邻域半径
        :param normals_num: 邻域内考虑的法线数量上限
        :param filter_std: 滤波的标准差倍率上限
        :param flux: 考虑法线相似度的法线通量
        :param max_remain: 最大保留点数
        """
        assert has_t3d  # 断言 pytorch3d 可用
        self.normals_radius = normals_radius  # 保存用于法线估计的邻域半径
        self.normals_num = normals_num  # 保存用于法线估计的邻域内法线数量上限
        self.filter_std = filter_std  # 保存滤波时使用的标准差倍率上限
        self.flux = flux  # 保存法线相似度计算时考虑的法线通量
        self.max_remain = max_remain  # 保存最大保留点数

    def __call__(self, pcd: PointCloud) -> PointCloud:  # 定义 __call__ 方法，使得 LowPassFilter 对象可以像函数一样被调用
        # 法线估计
        pcd_o3d = o3d.geometry.PointCloud()  # 创建一个 Open3D 的 PointCloud 对象
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd.xyz.cpu().numpy())  # 将点云数据从 GPU 转移到 CPU，并转换为 Open3D 格式
        pcd_o3d.estimate_normals(search_param=(o3d.geometry.KDTreeSearchParamRadius(radius=self.normals_radius)))  # 使用给定的半径参数估计法线
        normals = torch.tensor(np.asarray(pcd_o3d.normals), dtype=torch.float, device=pcd.device)  # 将估计的法线转换为 PyTorch 张量，并转移到与点云相同的设备上

        # 邻域内法线查询
        xyz = pcd.xyz  # 获取点云的坐标
        K = self.normals_num  # 获取邻域内法线数量上限
        result = knn_points(p1=xyz[None], p2=xyz[None], K=K + 1)  # 使用 knn_points 函数查找每个点的 K+1 个最近邻点
        grouped_indices = result.idx[..., 1:]  # 去除自身的索引，保留最近邻点的索引
        grouped_normals = knn_gather(x=normals[None], idx=grouped_indices)[0]  # 使用 knn_gather 函数获取邻域内的法线

        # 法线分布低通滤波 (N, K, 3) @ (N, 3, 1) => (N, K)
        normals_similarity = (grouped_normals @ normals.unsqueeze(-1)).squeeze(-1).abs()  # 计算邻域内法线与当前法线的相似度
        sim, _ = torch.topk(normals_similarity, k=self.flux, dim=-1)  # 取出法线相似度前 flux 个最大值
        sim = sim.sum(1)  # 将前 flux 个相似度值求和
        mask = sim > (sim.mean() - self.filter_std * sim.std())  # 根据标准差倍率生成掩码，保留高相似度的点
        if 0 < self.max_remain < mask.sum():  # 如果设置了最大保留点数且掩码中选中的点数超过该数
            _, mask = torch.topk(sim, k=self.max_remain)  # 则只保留相似度最高的 max_remain 个点

        pcd.apply_index(mask)  # 使用掩码过滤点云数据，只保留符合条件的点

        import sys  # 导入 sys 模块，用于获取平台信息
        if sys.platform == 'darwin' and False:  # 如果平台是 macOS 且特定条件为真，执行可视化（条件被设置为 False，表示不会执行）
            from utils.visualization import show_pcd  # 从 utils.visualization 导入 show_pcd 函数，用于显示点云
            show_pcd([xyz, pcd.xyz], [[1, 0, 0], [0, 1, 0]], window_name=f'{self}')  # 可视化过滤前后的点云
            show_pcd([pcd.xyz])  # 单独显示过滤后的点云
            import matplotlib.pyplot as plt  # 导入 matplotlib，用于绘制图形
            plt.figure(figsize=(10, 10), dpi=300)  # 创建一个图形窗口，设置大小和分辨率
            plt.title(f'{self}')  # 设置图形的标题
            plt.axis('equal')  # 设置坐标轴比例相等
            points = xyz.cpu().numpy()  # 获取原始点云坐标
            color = sim.cpu().numpy()  # 获取法线相似度值
            color = (color - color.min()) / (color.max() - color.min())  # 将相似度值归一化为 [0, 1] 区间
            plt.scatter(x=points[:, 0], y=points[:, 1], c=color, s=0.1, cmap='rainbow')  # 绘制散点图，使用颜色映射相似度
            plt.colorbar()  # 添加颜色条
            plt.show()  # 显示图形

        return pcd  # 返回经过滤波后的点云数据

    def __repr__(self):  # 定义 __repr__ 方法，返回对象的字符串表示，便于打印输出和调试
        s = f'{self.__class__.__name__}(normals_radius={self.normals_radius}, normals_num={self.normals_num}, ' \
            f'filter_std={self.filter_std}, flux={self.flux}, max_remain={self.max_remain})'
        return s  # 返回格式化后的字符串，包含类名和参数

    def __str__(self):  # 定义 __str__ 方法，返回对象的字符串表示
        return self.__repr__()  # 调用 __repr__ 方法，返回相同的字符串表示


class VerticalCorrect:  # 定义一个类 VerticalCorrect，用于垂直矫正点云，使所有点向 z 轴正方向旋转一定角度
    """垂直矫正，所有点向z轴正方向旋转一定角度"""

    def __init__(self, angle: float):  # 初始化函数，接收矫正的角度
        """
        :param angle: 矫正角度 (角度制)
        """
        self.angle = angle  # 将传入的矫正角度保存为类的属性

    def __call__(self, pcd: PointCloud) -> PointCloud:  # 定义 __call__ 方法，使得 VerticalCorrect 对象可以像函数一样被调用
        if self.angle == 0:  # 如果矫正角度为 0，则不进行任何操作，直接返回原始点云
            return pcd

        xyz = pcd.xyz.cpu().numpy()  # 将点云坐标从 GPU 转移到 CPU，并转换为 numpy 数组

        # 每点的旋转轴为点云向量与 z 轴所成平面的法向量，旋转角为 angle
        rotation_axis = np.cross(xyz, [0, 0, 1])  # 计算每个点与 z 轴的叉积，得到旋转轴的方向

        # 根据上述旋转轴和旋转角，获得每个点的旋转矩阵
        rotation_axis = rotation_axis / linalg.norm(rotation_axis, axis=1, keepdims=True)  # 对旋转轴进行归一化处理
        r = Rotation.from_rotvec(rotation_axis * self.angle, degrees=True)  # 根据旋转轴和角度生成旋转向量，并转换为旋转矩阵
        rotation_matrix = r.as_matrix().astype(np.float32)  # 将旋转矩阵转换为 numpy 数组，数据类型为 float32

        corrected_xyz = (rotation_matrix @ xyz[:, :, np.newaxis]).squeeze(-1)  # 将旋转矩阵应用到点云坐标上，得到矫正后的坐标
        pcd.xyz = torch.from_numpy(corrected_xyz)  # 将矫正后的坐标转换为 PyTorch 张量，并赋值给点云对象的坐标属性
        return pcd  # 返回矫正后的点云对象


class VoxelSample:  # 定义一个类 VoxelSample，用于对点云数据进行体素化下采样
    """体素化下采样"""

    def __init__(self, voxel_size: float, retention: Literal['first', 'center'] = 'center', num: int = None):
        """
        :param voxel_size: 体素边长
        :param retention: 每个体素保留点云的规则，'first' 为索引序最靠前的，'center' 为最接近中心的
        :param num: 采样点数量，为None时不约束，否则采样后点云数量不超过该值
        """
        assert retention in ['first', 'center'], f'\'{retention}\' is not a supported retention method, ' \
                                                 f'please use \'first\' or \'center\''  # 确保保留方法是 'first' 或 'center'
        self.voxel_size = voxel_size  # 保存体素的边长
        self.retention = retention  # 保存保留点的规则
        self.num = num  # 保存采样点的数量上限

    def __call__(self, pcd: PointCloud) -> PointCloud:  # 定义 __call__ 方法，使得 VoxelSample 对象可以像函数一样被调用
        device = pcd.device  # 获取点云数据所在的设备（CPU 或 GPU）
        pcd_xyz = pcd.xyz.cpu().numpy()  # 将点云坐标从 GPU 转移到 CPU，并转换为 numpy 数组

        # 根据点云范围确定voxel数量
        xyz_min = np.min(pcd_xyz, axis=0)  # 计算点云的最小边界
        xyz_max = np.max(pcd_xyz, axis=0)  # 计算点云的最大边界
        X, Y, Z = ((xyz_max - xyz_min) / self.voxel_size).astype(np.int32) + 1  # 根据体素边长计算每个方向上的体素数量

        # 计算每个点云所在voxel的xyz索引和总索引
        relative_xyz = pcd_xyz - xyz_min  # 计算每个点相对于点云最小边界的相对位置
        voxel_xyz = (relative_xyz / self.voxel_size).astype(np.int32)  # 计算每个点所在的体素的索引
        voxel_id = (voxel_xyz[:, 0] + voxel_xyz[:, 1] * X + voxel_xyz[:, 2] * X * Y).astype(np.int32)  # 将体素索引转换为一维索引，便于去重

        # 每个voxel仅保留索引序最前的点，因此不做任何调整即为 'first' 方法
        if self.retention == 'center':  # 如果选择保留最接近体素中心的点
            # 每个voxel仅保留最接近中心点的点云，预先根据点云距离voxel中心的距离由近到远进行排序，使得每个voxel最接近中心点的索引序最前
            dis = np.sum((relative_xyz - voxel_xyz * self.voxel_size - self.voxel_size / 2)**2, axis=-1)  # 计算每个点到体素中心的距离平方
            sorted_id = np.argsort(dis)  # 根据距离对点云排序
            voxel_id = voxel_id[sorted_id]  # 按照排序后的顺序更新体素索引
            pcd.apply_index(torch.from_numpy(sorted_id).to(device))  # 根据排序后的索引对点云数据进行重新排列

        # 去除相同voxel，id即为每个voxel内的采样点，cnt为当前采样点所在voxel的点云数量之和
        _, unique_id, cnt = np.unique(voxel_id, return_index=True, return_counts=True)  # 计算每个体素的唯一索引，并统计其中点的数量

        if self.num is not None and unique_id.shape[0] > self.num:  # 如果设置了最大采样点数量，且采样点数量超过该值
            # 大于指定数量时仅保留最密集的voxel
            cnt_topk_id = np.argpartition(cnt, kth=-self.num)[-self.num:]  # 仅保留点最多的体素
            unique_id = unique_id[cnt_topk_id]  # 更新保留的体素索引

        pcd.apply_index(torch.from_numpy(unique_id).to(device))  # 根据最终保留的体素索引过滤点云数据
        return pcd  # 返回经过下采样后的点云数据


class FarthestPointSample:  # 定义一个类 FarthestPointSample，用于执行最远点采样
    """最远点采样"""

    def __init__(self, num):  # 初始化函数，接收采样点数量
        """
        :param num: 采样点数量
        """
        if not has_t3d:  # 检查是否安装了 PyTorch3D 库
            raise NotImplementedError('Module pytorch3d not found! '
                                      '\'FarthestPointSample\' is only supported by PyTorch3D')  # 如果没有安装 PyTorch3D，则抛出异常
        self.num = num  # 将传入的采样点数量保存为类的属性

    def __call__(self, pcd: PointCloud) -> PointCloud:  # 定义 __call__ 方法，使得 FarthestPointSample 对象可以像函数一样被调用
        if pcd.nbr_point > self.num:  # 如果点云的总点数大于所需的采样点数
            points_xyz = pcd.xyz  # 获取点云的坐标数据
            idx = sample_farthest_points(points=points_xyz.unsqueeze(0), K=self.num)[1][0]  # 使用 PyTorch3D 提供的 sample_farthest_points 函数进行最远点采样，获取采样点的索引
            pcd.apply_index(idx)  # 根据采样的索引对点云进行过滤，只保留采样后的点
        return pcd  # 返回经过采样后的点云数据


class RandomSample:  # 定义一个类 RandomSample，用于执行点云数据的随机下采样
    """点云随机下采样"""

    def __init__(self, num):  # 初始化函数，接收采样点的数量
        """
        :param num: 采样点数量
        """
        self.num = num  # 将传入的采样点数量保存为类的属性

    def __call__(self, pcd: PointCloud) -> PointCloud:  # 定义 __call__ 方法，使得 RandomSample 对象可以像函数一样被调用
        if pcd.nbr_point > self.num:  # 如果点云的总点数大于所需的采样点数
            downsample_ids = torch.randperm(pcd.nbr_point, device=pcd.device)[:self.num]  # 使用 torch.randperm 生成一个随机排列的索引，取前 num 个作为采样点的索引
            pcd.apply_index(downsample_ids)  # 根据生成的随机索引对点云进行过滤，只保留采样后的点
        return pcd  # 返回经过随机下采样后的点云数据


class DistanceSample:  # 定义一个类 DistanceSample，用于执行距离采样
    """距离采样，保留给定距离内的点"""

    def __init__(self, min_dis: float, max_dis: float):  # 初始化函数，接收最小和最大距离参数
        """
        :param min_dis: 最近距离
        :param max_dis: 最远距离
        """
        self.min_dis = min_dis  # 保存最小距离
        self.max_dis = max_dis  # 保存最大距离

    def __call__(self, pcd: PointCloud) -> PointCloud:  # 定义 __call__ 方法，使得 DistanceSample 对象可以像函数一样被调用
        dis = torch.norm(pcd.xyz, p=2, dim=1)  # 计算点云中每个点到原点的欧几里得距离（L2范数）
        mask = (self.min_dis <= dis) & (dis <= self.max_dis)  # 生成一个掩码，只保留在给定距离范围内的点
        pcd.apply_index(mask)  # 使用掩码过滤点云数据，保留满足条件的点
        return pcd  # 返回经过距离采样后的点云数据


class CoordinatesNormalization:  # 定义一个类 CoordinatesNormalization，用于执行点云坐标归一化
    """点云坐标归一化"""

    def __init__(self, ratio: float):  # 初始化函数，接收归一化比率参数
        """
        :param ratio: 归一化比率
        """
        self.ratio = ratio  # 保存归一化比率

    def __call__(self, pcd: PointCloud) -> PointCloud:  # 定义 __call__ 方法，使得 CoordinatesNormalization 对象可以像函数一样被调用
        pcd.xyz /= self.ratio  # 将点云的坐标除以归一化比率，从而缩放点云的大小
        return pcd  # 返回经过归一化后的点云数据


class RandomShuffle:  # 定义一个类 RandomShuffle，用于随机打乱点云数据的顺序
    """随机打乱顺序"""

    def __init__(self, p: float = 1.0):  # 初始化函数，接收打乱概率参数
        self.p = p  # 保存打乱概率

    def __call__(self, pcd: PointCloud) -> PointCloud:  # 定义 __call__ 方法，使得 RandomShuffle 对象可以像函数一样被调用
        if random.random() > self.p:  # 如果生成的随机数大于打乱概率，则不进行打乱，直接返回原始点云
            return pcd
        shuffle_ids = torch.randperm(pcd.nbr_point, device=pcd.device)  # 生成一个随机排列的索引，用于打乱点云数据的顺序
        pcd.apply_index(shuffle_ids)  # 根据随机索引对点云数据进行重新排列
        return pcd  # 返回经过随机打乱后的点云数据


class RandomDrop:  # 定义一个类 RandomDrop，用于随机丢弃点云中的部分点
    """随机丢弃点"""

    def __init__(self, max_ratio: float, p: float = 1.0):  # 初始化函数，接收最大丢弃率和执行概率
        """
        :param max_ratio: 最大丢弃率
        :param p: 执行随机丢弃的概率
        """
        self.max_ratio = max_ratio  # 保存最大丢弃率
        self.p = p  # 保存执行丢弃操作的概率

    def __call__(self, pcd: PointCloud) -> PointCloud:  # 定义 __call__ 方法，使得 RandomDrop 对象可以像函数一样被调用
        if random.random() > self.p:  # 如果生成的随机数大于执行概率，则不进行丢弃操作，直接返回原始点云
            return pcd
        drop_ratio = random.uniform(0, self.max_ratio)  # 随机生成一个丢弃率，范围在 0 到最大丢弃率之间
        remained_ids = torch.rand(size=(pcd.nbr_point, ), device=pcd.device) >= drop_ratio  # 生成随机掩码，保留丢弃率以下的点
        pcd.apply_index(remained_ids)  # 根据掩码过滤点云数据，保留未丢弃的点
        return pcd  # 返回经过随机丢弃后的点云数据


class RandomShield:  # 定义一个类 RandomShield，用于模拟因近距离遮挡物（如车辆、墙体等）造成的放射状遮挡
    """随机遮挡，模仿因近距离车辆、墙体等大型目标造成的放射状遮挡"""

    def __init__(self, angle_range: list, dis_range: list, max_num: int, p: float = 0.1):  # 初始化函数，接收遮挡角度范围、距离范围、遮挡物数量和执行概率
        super().__init__()
        self.angle_range = angle_range  # 保存遮挡角度范围
        self.dis_range = dis_range  # 保存遮挡物与点云的距离范围
        self.max_num = max_num  # 保存遮挡物的最大数量
        self.p = p  # 保存执行遮挡操作的概率

    def __call__(self, pcd: PointCloud) -> PointCloud:  # 定义 __call__ 方法，使得 RandomShield 对象可以像函数一样被调用
        if random.random() > self.p:  # 如果生成的随机数大于执行概率，则不进行遮挡操作，直接返回原始点云
            return pcd
        xyz = pcd.xyz  # 获取点云的坐标
        device = pcd.device  # 获取点云所在的设备（CPU 或 GPU）

        # 计算所有点的方位角和距离
        azimuth_angle = (torch.atan2(other=xyz[:, 0], input=xyz[:, 1])) * 180 / torch.pi  # 计算每个点的方位角（以度为单位）
        distance = torch.norm(xyz, p=2, dim=1)  # 计算每个点到原点的欧几里得距离
        mask = torch.ones(size=(pcd.nbr_point, ), dtype=torch.bool, device=device)  # 初始化一个全为 True 的掩码，用于保留未被遮挡的点

        # 随机生成 [1, max_num] 个遮挡物
        num = random.randint(1, self.max_num)  # 随机生成遮挡物的数量
        for i in range(num):  # 对每个遮挡物进行处理

            angle, dis, direction = torch.rand(size=(3, ), device=device)  # 随机生成遮挡角度、距离和起始方位角
            angle = (angle * (self.angle_range[1] - self.angle_range[0]) + self.angle_range[0]) / (i + 1)  # 根据角度范围计算遮挡角度，并逐渐减小遮挡角度
            dis_threshold = dis * (self.dis_range[1] - self.dis_range[0]) + self.dis_range[0]  # 根据距离范围计算遮挡物的距离阈值
            direction = direction * 360 - 180  # 计算遮挡区域的起始方位角，范围在 [-180, 180] 度之间

            # 去掉被遮挡的扇形区域
            angle_start, angle_end = direction, direction + angle  # 计算遮挡区域的起始和结束方位角
            if angle_end <= 180:  # 如果结束角度在 -180 到 180 度之间
                shield_angle = (azimuth_angle >= angle_start) & (azimuth_angle <= angle_end)  # 生成遮挡区域的掩码
            else:  # 如果结束角度超出了 180 度
                shield_angle = (azimuth_angle >= angle_start) | (azimuth_angle <= angle_end - 360)  # 生成跨越 -180 和 180 度的遮挡区域掩码
            shield_dis = (distance >= dis_threshold)  # 生成遮挡物距离的掩码
            mask &= ~(shield_angle & shield_dis)  # 更新掩码，排除遮挡区域内的点

        pcd.apply_index(mask)  # 根据最终生成的掩码过滤点云数据，只保留未被遮挡的点
        return pcd  # 返回经过随机遮挡后的点云数据


class RandomRT:  # 定义一个类 RandomRT，用于对点云数据进行随机旋转和平移变换
    """随机旋转平移变换"""

    def __init__(self, r_mean: float = 0, r_std: float = 3.14, t_mean: float = 0, t_std: float = 1, p: float = 1.0, pair: bool = True):  # 初始化函数，接收旋转和平移的参数
        self.r_mean = r_mean  # 旋转角度的均值
        self.r_std = r_std  # 旋转角度的标准差
        self.t_mean = t_mean  # 平移的均值
        self.t_std = t_std  # 平移的标准差
        self.p = p  # 执行旋转平移变换的概率
        self.pair = pair  # 指定是否成对应用旋转变换，通常用于数据增强
        self.flag = True  # 标志位，用于在成对变换中区分第一次和第二次应用

    def __call__(self, pcd: PointCloud) -> PointCloud:  # 定义 __call__ 方法，使得 RandomRT 对象可以像函数一样被调用
        if random.random() > self.p:  # 如果生成的随机数大于执行概率，则不进行任何变换，直接返回原始点云
            return pcd

        xyz, R, T, device = pcd.xyz, pcd.R, pcd.T, pcd.device  # 获取点云的坐标、旋转矩阵、平移矩阵以及设备信息

        # 生成三方向的随机角度，得到各方位的旋转矩阵，最后整合为总体旋转矩阵
        if self.pair:  # 如果设置为成对应用旋转变换
            # 为使得一组两个旋转变换的差值服从均匀分布，令第一个服从U1分布，第二个服从U1+U2分布，差值即符合U2的均匀分布
            if self.flag:  # 第一次应用时生成完整的随机旋转
                x, y, z = (torch.rand(size=(3, )) - 0.5) * 2 * torch.pi  # 第一次的随机旋转角度
            else:  # 第二次应用时生成相对较小的旋转
                x, y, z = (torch.rand(size=(3, )) - 0.5) * 2 * self.r_std  # 第二次的指定旋转角度
            x, y = x / 10, y / 10  # 将俯仰角和滚转角的范围缩小到原来的十分之一

            # 生成 x、y 和 z 方向的旋转矩阵
            R_x = torch.tensor([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
            R_y = torch.tensor([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
            R_z = torch.tensor([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
            R_aug = R_x @ R_y @ R_z  # 将三个方向的旋转矩阵组合成一个最终的旋转矩阵

            if self.flag:  # 如果是第一次应用旋转
                self.random_R = R_aug  # 保存第一次的旋转矩阵
            else:  # 如果是第二次应用旋转
                R_aug = R_aug @ self.random_R  # 在第一次旋转的基础上应用第二次的随机旋转
            self.flag = not self.flag  # 切换标志位

        else:  # 如果不需要成对变换
            x, y, z = (torch.rand(size=(3, )) - 0.5) * 2 * self.r_std  # 直接生成随机旋转角度
            x, y = x / 10, y / 10  # 将俯仰角和滚转角的范围缩小到原来的十分之一
            # 生成 x、y 和 z 方向的旋转矩阵
            R_x = torch.tensor([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
            R_y = torch.tensor([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
            R_z = torch.tensor([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
            R_aug = R_x @ R_y @ R_z  # 将三个方向的旋转矩阵组合成一个最终的旋转矩阵

        R_aug = R_aug.to(pcd.device)  # 将旋转矩阵移到与点云相同的设备上

        if self.t_std > 0:  # 如果平移标准差大于0，则生成平移矩阵
            T_aug = torch.normal(size=(3, 1), mean=self.t_mean, std=self.t_std, device=device)  # 生成随机平移向量
            T_aug[2] /= 2  # 缩小z轴方向的平移幅度
        else:  # 否则平移矩阵为0
            T_aug = torch.zeros(size=(3, 1), device=device)

        pcd.xyz = (R_aug @ xyz.T + T_aug).T  # 应用旋转和平移矩阵变换点云的坐标
        if pcd.has_norm:  # 如果点云包含法向量
            pcd.norm = (R_aug @ pcd.norm.T).T  # 对法向量应用相同的旋转矩阵
        '''
        R @ pcd + T = R_new @ (R_aug @ pcd + T_aug) + T_new
                    = (R_new @ R_aug) @ pcd + R_new @ T_aug + T_new
        R = R_new @ R_aug
        T = R_new @ T_aug + T_new
        '''
        R_new = R @ R_aug.T  # 计算新的旋转矩阵 R_new
        T_new = T - R_new @ T_aug  # 计算新的平移矩阵 T_new
        calib_SE3 = torch.eye(4, dtype=torch.float32, device=device)  # 初始化 4x4 的校准矩阵
        calib_SE3[:3, :3] = R_aug  # 将旋转部分填入校准矩阵
        calib_SE3[:3, 3:] = T_aug  # 将平移部分填入校准矩阵
        pcd.calib = calib_SE3 @ pcd.calib  # 更新点云的校准矩阵

        pcd.R = R_new  # 更新点云的旋转矩阵
        pcd.T = T_new  # 更新点云的平移矩阵
        return pcd  # 返回经过随机旋转和平移变换后的点云数据


class RandomPosJitter:  # 定义一个类 RandomPosJitter，用于对点云的坐标进行随机抖动
    """点云坐标随机抖动"""

    def __init__(self, mean: float = 0, std: float = 0.05, p: float = 1.0):  # 初始化函数，接收抖动的均值、标准差和执行概率
        self.mean = mean  # 保存抖动的均值
        self.std = std  # 保存抖动的标准差
        self.p = p  # 保存执行抖动操作的概率

    def __call__(self, pcd: PointCloud) -> PointCloud:  # 定义 __call__ 方法，使得 RandomPosJitter 对象可以像函数一样被调用
        if random.random() > self.p:  # 如果生成的随机数大于执行概率，则不进行抖动操作，直接返回原始点云
            return pcd

        # 生成一个与点云大小相同的随机抖动矩阵，并将其应用到点云坐标上
        pos_jitter = torch.normal(size=(pcd.nbr_point, 3), mean=self.mean, std=self.std, device=pcd.device)\
            .clamp(min=-3 * self.std, max=3 * self.std)  # 限制抖动的范围在 [-3*std, 3*std] 之间
        pcd.xyz += pos_jitter  # 将抖动矩阵加到点云坐标上
        return pcd  # 返回经过抖动处理后的点云


class ToGPU:  # 定义一个类 ToGPU，用于将点云类的成员转移到 GPU 上以加速处理
    """将点云类成员转移至GPU以加速处理"""

    def __init__(self):  # 初始化函数，检查是否有可用的 GPU
        self.has_gpu = torch.cuda.is_available()  # 检查当前是否有可用的 GPU
        # assert torch.cuda.is_available(), '\'ToGPU\' needs CUDA gpu, but CUDA is not available'

    def __call__(self, pcd: PointCloud) -> PointCloud:  # 定义 __call__ 方法，使得 ToGPU 对象可以像函数一样被调用
        if self.has_gpu:  # 如果有可用的 GPU
            pcd.to_gpu()  # 将点云的成员转移到 GPU 上
        return pcd  # 返回转移后的点云


class ToCPU:  # 定义一个类 ToCPU，用于将点云类的成员转移到 CPU 上
    """将点云类成员转移至CPU"""

    def __init__(self):  # 初始化函数，这里没有特殊初始化操作
        pass

    def __call__(self, pcd: PointCloud) -> PointCloud:  # 定义 __call__ 方法，使得 ToCPU 对象可以像函数一样被调用
        pcd.to_cpu()  # 将点云的成员转移到 CPU 上
        return pcd  # 返回转移后的点云


class ToTensor:  # 定义一个类 ToTensor，用于将点云类转换为独立的张量
    """从点云类转换为独立的张量"""

    def __init__(self, use_norm: bool = False, use_uvd: bool = False, use_label: bool = False, padding_to: int = -1, use_image: bool = False, use_calib: bool = False):  # 初始化函数，接收各种转换选项
        self.use_norm = use_norm  # 是否使用法线信息
        self.use_uvd = use_uvd  # 是否使用与图像的对应位置数据
        self.use_label = use_label  # 是否使用标签信息
        self.use_image = use_image  # 是否返回图像信息
        self.use_calib = use_calib  # 是否返回校准矩阵
        self.padding_to = padding_to  # 是否填充到固定长度

    def __call__(self, pcd: PointCloud):
        return pcd.to_tensor(use_norm=self.use_norm, use_uvd=self.use_uvd, use_label=self.use_label, use_image=self.use_image, use_calib=self.use_calib, padding_to=self.padding_to)  # 将点云转换为张量，并返回转换后的数据


pointcloud_transforms = {  # 定义一个字典，包含了各种点云数据预处理和数据增强类。
    'GroundFilter': GroundFilter,  # 地面滤波器，用于去除点云中的地面点。
    'OutlierFilter': OutlierFilter,  # 离群点滤波器，用于去除点云中的离群点。
    'LowPassFilter': LowPassFilter,  # 低通滤波器，用于平滑点云数据，去除高频噪声。
    'VerticalCorrect': VerticalCorrect,  # 垂直校正，用于将点云的垂直方向校正为统一的参考方向。
    'VoxelSample': VoxelSample,  # 体素采样，用于将点云体素化以减少点的数量。
    'FarthestPointSample': FarthestPointSample,  # 最远点采样，用于选择点云中距离最远的点，进行均匀采样。
    'RandomSample': RandomSample,  # 随机采样，用于从点云中随机选择一定数量的点。
    'DistanceSample': DistanceSample,  # 距离采样，根据距离进行点云采样，通常用于保留较远的点。
    'CoordinatesNormalization': CoordinatesNormalization,  # 坐标归一化，用于将点云坐标归一化到某个范围内。
    'RandomShuffle': RandomShuffle,  # 随机打乱点的顺序。
    'RandomDrop': RandomDrop,  # 随机丢弃点云中的一些点，用于数据增强。
    'RandomShield': RandomShield,  # 随机遮挡点云的一部分，用于模拟部分遮挡情况。
    'RandomRT': RandomRT,  # 随机旋转和平移，用于对点云进行随机的刚性变换。
    'RandomPosJitter': RandomPosJitter,  # 随机位置抖动，用于在点的坐标上加入微小的随机噪声。
    'ToGPU': ToGPU,  # 将点云数据移动到GPU上进行计算。
    'ToCPU': ToCPU,  # 将点云数据移动到CPU上进行计算。
    'ToTensor': ToTensor  # 将点云数据转换为PyTorch的张量格式，方便后续操作。
}

def get_transforms(args_dict: dict, return_list: bool = False) -> Union[Compose, List]:
    # 定义一个函数，根据传入的配置字典生成数据变换链
    transforms_list = []  # 初始化一个空列表，用于存储生成的变换对象
    for key, value in args_dict.items():  # 遍历配置字典中的每个键值对
        if key != 'RandomChoice':  # 如果当前键不是 'RandomChoice'
            transforms_list.append(pointcloud_transforms[key](**value))  # 根据键在字典中找到对应的变换类，并用参数实例化，然后添加到列表中
        else:  # 如果当前键是 'RandomChoice'
            sub_transforms = get_transforms(value['transforms'], return_list=True)  # 递归调用 get_transforms 生成子变换列表
            p = value['p']  # 获取随机选择的概率
            transforms_list.append(RandomChoice(transforms=sub_transforms, p=p))  # 用生成的子变换列表和概率实例化 RandomChoice 对象，并添加到列表中
    if return_list:  # 如果函数的参数要求返回列表
        return transforms_list  # 返回包含所有变换对象的列表
    else:  # 否则
        return Compose(transforms=transforms_list)  # 返回组合后的变换链，使用 Compose 将列表中的变换对象依次应用



class PointCloudTransforms:
    """点云数据预处理与数据增强"""

    def __init__(self, args, mode: Literal['train', 'infer'] = 'train'):
        assert mode in ['train', 'infer']
        self.args = args
        self.transforms = get_transforms(args.transforms)
        self.mode = mode  # 保存模式
        if mode == 'train':  # 如果模式是训练
            self._call_method = self._call_train  # 设置调用方法为训练方法
        else:  # 如果模式是推理
            self._call_method = self._call_infer  # 设置调用方法为推理方法

    def __call__(self, pcd: PointCloud):  # 定义 __call__ 方法，使得 PointCloudTransforms 对象可以像函数一样被调用
        return self._call_method(pcd)  # 调用当前模式下的处理方法

    def _call_train(self, pcd: PointCloud):  # 定义训练模式下的处理方法
        return self.transforms(pcd)  # 直接应用变换链

    def _call_infer(self, pcd: PointCloud):  # 定义推理模式下的处理方法
        original_pcd = pcd.xyz.clone()  # 克隆原始点云的坐标，以便后续使用
        results = self.transforms(pcd)  # 应用变换链
        return *results, original_pcd  # 返回变换后的结果和原始点云