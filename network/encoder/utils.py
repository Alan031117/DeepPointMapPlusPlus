import time

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor as Tensor
from typing import Literal, Tuple, List
from random import randint
import colorlog as logging
import pytorch3d

# 创建并配置一个日志记录器对象，用于打印和记录日志信息
logger = logging.getLogger(__name__)  # 获取一个命名为当前模块名的日志记录器实例
logger.setLevel(logging.INFO)  # 设置日志记录器的级别为INFO，意味着会记录INFO及以上级别的日志信息

# 尝试从pytorch3d中导入相关函数和操作。如果失败（即pytorch3d未安装），则忽略异常
try:
    from pytorch3d.ops import ball_query, knn_points, knn_gather, sample_farthest_points
except:
    pass  # 如果导入失败，则继续运行，不做处理

# 定义一个标志变量，用于指示是否打印pytorch3d的警告信息
print_t3d_warning = False


class Querier:
    """邻域点搜索"""

    # 初始化函数，接受一个字符串参数method，用于指定邻域点搜索的方法
    def __init__(self, method: Literal['knn', 'hybrid', 'ball', 'knn-t3d', 'ball-t3d', 'hybrid-t3d']):
        # 创建一个字典，将不同的字符串method值映射到相应的搜索方法
        method_dict = {
            'knn': self.knn_query,  # 对应K近邻搜索方法
            'ball': self.ball_query,  # 对应球查询搜索方法
            'hybrid': self.hybrid_query,  # 对应混合搜索方法
            'knn-t3d': self.knn_query_t3d,  # 对应基于pytorch3d的K近邻搜索方法
            'ball-t3d': self.ball_query_t3d,  # 对应基于pytorch3d的球查询搜索方法
            'hybrid-t3d': self.hybrid_query_t3d,  # 对应基于pytorch3d的混合搜索方法
        }

        # 如果method以't3d'结尾，则尝试导入pytorch3d的相关操作函数
        if method.endswith('t3d'):
            try:
                from pytorch3d.ops import ball_query, knn_points, knn_gather  # 尝试导入pytorch3d库的函数
            except:  # 如果导入失败
                method = method[:-4]  # 将method的't3d'部分去掉，切换为非pytorch3d实现
                global print_t3d_warning  # 声明一个全局变量，用于标记是否已经打印过警告
                if not print_t3d_warning:
                    # 如果还没有打印过警告，则打印警告信息，提示pytorch3d模块未找到，性能可能会下降
                    logger.warning(f'Module pytorch3d not found. The implementations of python version will be used, '
                                   f'which may cause significant speed decrease.')
                    print_t3d_warning = True  # 将全局变量设为True，避免重复打印警告
        # 根据method确定使用的查询方法，并将其赋值给self.query_method
        self.query_method = method_dict[method.lower()]

    # 定义一个调用函数，使实例对象可以像函数一样被调用
    def __call__(self, *args, **kwargs):
        # 调用查询方法并传入参数，获取分组索引
        grouped_idx = self.query_method(**kwargs)

        # ***** TEST *****  # 以下代码为测试部分，可以取消注释来测试查询结果
        # 从kwargs中提取半径、K值、点集、中心点集以及点集的填充数量
        # radius, K, points, centers, points_padding = \
        #     kwargs['radius'], kwargs['K'], kwargs['points'], kwargs['centers'], kwargs['points_padding']
        # 计算每个分组的独立点数量
        # grouped_num = torch.tensor([torch.unique(idx).shape[0] for idx in grouped_idx], device=grouped_idx.device)
        # 计算每个点集的有效点数量（即没有被填充的点数）
        # points_num = points.shape[1] - kwargs['points_padding'].sum(1)
        # 计算查询比率（即分组的点数占总点数的比例）
        # query_ratio = grouped_num / points_num
        # 获取查询比率的最大值、平均值和最小值
        # ratio_max = query_ratio.max().item()
        # ratio_mean = query_ratio.mean().item()
        # ratio_min = query_ratio.min().item()
        # 获取点集的总点数和中心点集的点数
        # N = points.shape[1]
        # S = centers.shape[1]
        # 打印查询方法的名称以及相关参数和比率统计信息
        # print(f'{self.query_method.__name__}(R={radius}, K={K}): '
        #       f'[{str(N):>5s} -> {str(S):>5s}] ==> {ratio_max:.2%} | {ratio_mean:.2%} | {ratio_min:.2%}')

        # 如果点集总点数与中心点集点数不一致，则可视化查询结果中最小比率的情况
        # if N != S:
        #     from utils.visualization import show_pcd  # 导入点云可视化工具
        #     index = torch.argmin(query_ratio)  # 找到查询比率最小的索引
        #     grouped_points = points[index][grouped_idx[index].unique()]  # 获取最小比率对应的分组点集
        #     all_points = points[index][~points_padding[index]]  # 获取所有的有效点
        #     show_pcd([all_points, grouped_points], [[0, 1, 0], [1, 0, 0]], window_name=f'worst_ratio {ratio_min}')
        # 使用绿色和红色分别显示全部点和分组点

        # ***** TEST *****  # 测试代码结束

        # 返回分组索引
        return grouped_idx

        # return self.query_method(**kwargs)  # 可以选择直接返回查询方法的结果

    @staticmethod
    def knn_query(K: int, points: Tensor, centers: Tensor, points_padding: Tensor) -> Tensor:
        """
        基于采样点进行K近邻搜索（KNN）

        :param K: 近邻点的数量
        :param points: (B, N, 3+) 原始点云数据，B为批次大小，N为点的数量，3+表示每个点至少有3个坐标值（x, y, z）以及可能更多的特征
        :param centers: (B, S, 3+) 查询中心点，S为中心点的数量
        :param points_padding: (B, N) 原始点云的掩码，True表示该点为填充点，False表示该点为有效点，填充点应被忽略
        :return: (B, S, K) 每个中心点对应的K近邻点的索引，输出的Tensor形状为(B, S, K)
        """

        # 克隆原始点云数据，避免在原数据上进行修改
        points = points.clone()

        # 将填充点的位置设置为一个较远的值，以确保在距离计算中它们不会被选为近邻点
        # 填充值为原始点云数据的绝对最大值的三倍，确保距离足够大
        points[points_padding] = points.abs().max() * 3

        # 计算每个中心点与所有点的距离的平方，dist的形状为(B, S, N)
        dist = coordinate_distance(centers[..., :3], points[..., :3])

        # 对距离进行升序排序，并选取距离最近的K个点的索引，group_idx的形状为(B, S, K)
        group_idx = torch.topk(dist, k=K, dim=-1, largest=False)[1]

        # 返回选取的K近邻点的索引
        return group_idx

    @staticmethod
    def ball_query(radius: float, K: int, points: Tensor, centers: Tensor, points_padding: Tensor) -> Tensor:
        """
        基于采样点进行ball query

        :param radius: 搜索半径
        :param K: 近邻点数量
        :param points: (B, N, 3+) 原始点云
        :param centers: (B, S, 3+) 采样点
        :param points_padding: (B, N) 原始点云的掩码，True为填充点，False为其他点，忽略填充点
        :return: (B, S, K) 每个采样点grouping的点云索引
        """

        # 获取批次大小B，点云数量N，以及点云数据的维度_（通常为3或更多）
        B, N, _ = points.shape

        # 获取采样点的数量S以及采样点数据的维度_
        _, S, _ = centers.shape

        # 获取点云张量所在的设备（CPU或GPU）
        device = points.device

        # 克隆点云数据，防止修改原始数据
        points = points.clone()

        # 将填充点（根据points_padding中的True值）调整到一个很远的位置（以便它们在计算距离时不会干扰真实点）
        # 这里用的是点云数据的绝对值最大值乘以3，确保填充点足够远
        points[points_padding] = points.abs().max() * 3

        # 计算每个采样点与所有点之间的欧氏距离的平方
        dist = coordinate_distance(centers[..., :3], points[..., :3])

        # 初始化一个用于存储每个采样点对应的最近邻点索引的张量，形状为(B, S, N)
        group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])

        # 将超出搜索半径的点的索引设置为N（标记为无效）
        group_idx[dist > radius ** 2] = N

        # 对索引进行排序，并取前K个（即K个最近邻点的索引）
        group_idx = group_idx.sort(dim=-1)[0][:, :, :K]

        # 获取每组K个最近邻点中的第一个点的索引
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, K])

        # 生成掩码，标记那些被设置为无效的索引（即超出搜索半径的点）
        mask = group_idx == N

        # 将超出半径的点索引设置为最邻近的有效点的索引
        group_idx[mask] = group_first[mask]

        # 返回每个采样点所对应的最近邻点的索引，形状为(B, S, K)
        return group_idx

    @staticmethod
    def hybrid_query(radius: float, K: int, points: Tensor, centers: Tensor, points_padding: Tensor) -> Tensor:
        """
        基于采样点进行KNN与ball query的混合查询

        :param radius: 搜索半径
        :param K: 近邻点数量
        :param points: (B, N, 3+) 原始点云
        :param centers: (B, S, 3+) 采样点
        :param points_padding: (B, N) 原始点云的掩码，True为填充点，False为其他点，忽略填充点
        :return: (B, S, K) 每个采样点grouping的点云索引
        """
        # 获取批次大小B，点云数量N，以及点云数据的维度_（通常为3或更多）
        B, N, _ = points.shape

        # 获取采样点的数量S以及采样点数据的维度_
        _, S, _ = centers.shape

        # 克隆点云数据，防止修改原始数据
        points = points.clone()

        # 将填充点（根据points_padding中的True值）调整到一个很远的位置（以便它们在计算距离时不会干扰真实点）
        # 这里用的是点云数据的绝对值最大值乘以3，确保填充点足够远
        points[points_padding] = points.abs().max() * 3

        # 计算每个采样点与所有点之间的欧氏距离的平方
        dist = coordinate_distance(centers[..., :3], points[..., :3])

        # 基于距离从小到大选择最近的K个点作为采样点，并返回对应的距离和索引
        dist, group_idx = torch.topk(dist, k=K, dim=-1, largest=False)

        # 创建一个掩码，标记距离大于搜索半径的点
        mask = dist > (radius ** 2)

        # 获取每组K个最近邻点中的第一个点的索引
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, K])

        # 将超出半径的点索引替换为最近的有效点的索引
        group_idx[mask] = group_first[mask]

        # 返回每个采样点所对应的最近邻点的索引，形状为(B, S, K)
        return group_idx

    @staticmethod
    def knn_query_t3d(K: int, points: Tensor, centers: Tensor, points_padding: Tensor) -> Tensor:
        """
        基于采样点进行KNN

        :param K: 近邻点数量
        :param points: (B, N, 3+) 原始点云
        :param centers: (B, S, 3+) 查询中心
        :param points_padding: (B, N) 原始点云的掩码，True为填充点，False为其他点，忽略填充点
        :return: (B, S, K) 每个采样点grouping的点云索引
        """
        # 调用knn_points函数计算K近邻点，不返回最近邻点坐标，而只返回索引
        result = knn_points(p1=centers[..., :3], p2=points[..., :3], lengths2=(~points_padding).sum(1),
                            K=K, return_nn=False, return_sorted=False)

        # 从计算结果中提取K近邻点的索引
        idx = result.idx

        # 返回这些索引，形状为(B, S, K)
        return idx

    @staticmethod
    def ball_query_t3d(radius: float, K: int, points: Tensor, centers: Tensor, points_padding: Tensor) -> Tensor:
        """
        基于采样点进行ball query

        :param radius: 搜索半径
        :param K: 近邻点数量
        :param points: (B, N, 3+) 原始点云
        :param centers: (B, S, 3+) 采样点
        :param points_padding: (B, N) 原始点云的掩码，True为填充点，False为其他点，忽略填充点
        :return: (B, S, K) 每个采样点grouping的点云索引
        """
        # 调用ball_query函数进行球查询，获取K个最近邻点的索引
        result = ball_query(p1=centers[..., :3], p2=points[..., :3], lengths2=(~points_padding).sum(1),
                            K=K, radius=radius, return_nn=False)

        # 从查询结果中提取这些索引
        idx = result.idx

        # 创建掩码，标记索引为-1的项（表示没有找到足够的点）
        idx_mask = idx == -1

        # 如果存在任何无效索引（即-1），将它们替换为每个center采样到的第一个索引
        if idx_mask.any():
            padding_idx = idx[:, :, :1].repeat(1, 1, K)
            idx[idx_mask] = padding_idx[idx_mask]

        # 返回每个采样点所对应的最近邻点的索引，形状为(B, S, K)
        return idx

    @staticmethod
    def hybrid_query_t3d(radius: float, K: int, points: Tensor, centers: Tensor, points_padding: Tensor) -> Tensor:
        """
        基于采样点进行KNN与ball query的混合查询

        :param radius: 搜索半径
        :param K: 近邻点数量
        :param points: (B, N, 3+) 原始点云
        :param centers: (B, S, 3+) 采样点
        :param points_padding: (B, N) 原始点云的掩码，True为填充点，False为其他点，忽略填充点
        :return: (B, S, K) 每个采样点grouping的点云索引
        """

        # 计算搜索半径的平方，避免在循环或计算中重复计算
        radius_squared = radius ** 2

        # 获取有效点的数量，忽略padding
        lengths = (~points_padding).sum(1)

        # 使用 knn_points 函数执行 KNN 查询，确保 points 和 centers 是浮点型
        result = knn_points(
            p1=centers[..., :3],  # 采样点
            p2=points[..., :3],  # 原始点云
            lengths2=lengths,  # 有效点的长度
            K=K,
            return_nn=False,
            return_sorted=False
        )

        # 获取最近邻点的索引和距离
        idx = result.idx
        dists = result.dists

        # 创建掩码，标记距离大于搜索半径的点
        dists_mask = dists > radius_squared

        # 对于超出半径的点，直接将它们的索引替换为第一个有效的最近邻点索引
        # 扩展 first_neighbor_idx 的维度，使其与 dists_mask 的形状匹配
        first_neighbor_idx = idx[:, :, 0].unsqueeze(-1).expand_as(idx)

        # 将掩码中的超出半径的点索引替换为第一个最近邻点的索引
        idx[dists_mask] = first_neighbor_idx[dists_mask]

        # 返回每个采样点所对应的最近邻点的索引，形状为(B, S, K)
        return idx


class Sampler:
    """下采样"""

    def __init__(self, method: Literal['fps', 'voxel', 'fps-t3d']):
        # 定义一个字典，将method参数值映射到相应的采样方法函数
        method_dict = {
            'fps': self.fps,  # 'fps'表示最远点采样方法
            'voxel': self.voxel,  # 'voxel'表示体素网格采样方法
            'fps-t3d': self.fps_t3d,  # 'fps-t3d'表示基于PyTorch3D库的最远点采样方法
        }

        # 如果采样方法以't3d'结尾，则尝试导入PyTorch3D库中的sample_farthest_points函数
        if method.endswith('t3d'):
            try:
                from pytorch3d.ops import sample_farthest_points
            except:
                # 如果导入失败，将method的't3d'后缀去掉，并发出警告
                method = method[:-4]
                global print_t3d_warning
                if not print_t3d_warning:
                    logger.warning(
                        f'Module pytorch3d not found. The implementations of python version will be used, '
                        f'which may cause significant speed decrease.')
                    print_t3d_warning = True

        # 根据method参数的值，将sample_method设为相应的采样函数
        self.sample_method = method_dict[method.lower()]

    def __call__(self, *args, **kwargs):
        # 使Sampler对象可以像函数一样调用，直接执行选定的采样方法
        return self.sample_method(**kwargs)

    @staticmethod
    def voxel(points: Tensor, points_padding: Tensor, K: int, voxel_size: float = 0.3, sample_range: float = 1.0) \
            -> Tuple[Tensor, Tensor]:
        """
        体素下采样，均匀划分体素，每个体素保留最靠近体素中心的点

        :param points: (B, N, 3+) 原始点云
        :param points_padding: (B, N) 原始点云的掩码，True为填充点，False为其他点，忽略填充点
        :param K: 采样点数量，为None时不约束
        :param voxel_size: 体素边长
        :param sample_range: 采样范围 0 < dis < sample_range (sample_range > 0)
        :return: 采样点(B, K, 3+), 采样点padding掩码(B, K)
        """
        # 获取批次大小B和设备信息（CPU或GPU）
        B, device = points.shape[0], points.device

        # 克隆点云数据的前三个维度（通常是xyz坐标）
        pcd_xyz = points[:, :, :3].clone()

        # 将padding点调整到采样范围之外的一个位置，以避免它们被采样
        pcd_xyz[points_padding] = 2 * sample_range

        # 计算每个批次中点云的最小坐标和最大坐标
        xyz_min = torch.min(pcd_xyz, dim=1)[0]
        xyz_max = torch.max(pcd_xyz, dim=1)[0]

        # 根据点云范围和体素大小，计算每个批次的体素数量
        X, Y, Z = torch.div(xyz_max[:, 0] - xyz_min[:, 0], voxel_size, rounding_mode='trunc') + 1, \
                  torch.div(xyz_max[:, 1] - xyz_min[:, 1], voxel_size, rounding_mode='trunc') + 1, \
                  torch.div(xyz_max[:, 2] - xyz_min[:, 2], voxel_size, rounding_mode='trunc') + 1

        # 计算每个点到采样范围中心的平方距离，生成一个掩码用于标记在采样范围内的点
        dis_mask = torch.sum(pcd_xyz.pow(2), dim=-1) <= (sample_range * sample_range)

        # X和Y维度需要扩展成与点云数目相同的形状
        X, Y = X.unsqueeze(1), Y.unsqueeze(1)

        # 计算相对于最小坐标的相对坐标
        relative_xyz = pcd_xyz - xyz_min.unsqueeze(1)

        # 计算每个点所属的体素坐标，并转换为整数
        voxel_xyz = torch.div(relative_xyz, voxel_size, rounding_mode='trunc').int()

        # 计算每个点的体素索引（将体素的x, y, z索引转换为一个总索引）
        voxel_id = (voxel_xyz[:, :, 0] + voxel_xyz[:, :, 1] * X + voxel_xyz[:, :, 2] * X * Y).int()

        sampled_points = []

        '''每个体素内仅保留最接近中心点的点云，并根据体素内点云数量排序'''

        # 计算每个点与其所在体素中心的距离的平方
        dis = torch.sum((relative_xyz - voxel_xyz * voxel_size - voxel_size / 2).pow(2), dim=-1)

        # 对距离进行排序，按照距离从小到大排序后，得到排序后的索引
        dis, sorted_id = torch.sort(dis, dim=-1)

        # 根据排序后的索引调整voxel_id和points的顺序
        b_id = torch.arange(points.shape[0], device=device).unsqueeze(1)
        voxel_id = voxel_id[b_id, sorted_id]
        points = points[b_id, sorted_id]

        # 对距离掩码也应用同样的排序
        dis_mask = dis_mask[b_id, sorted_id]

        # 遍历每个批次
        for b in range(points.shape[0]):

            # 保留距离小于采样范围上限的点云
            b_voxel_id = voxel_id[b, dis_mask[b]]
            b_pcd = points[b, dis_mask[b]]

            # 使用np.unique函数去除同一个体素内的重复点，并获取每个唯一voxel的索引和点云数量
            _, unique_id, cnt = np.unique(b_voxel_id.detach().cpu(), return_index=True, return_counts=True)
            unique_id, cnt = torch.tensor(unique_id, device=device), torch.tensor(cnt, device=device)

            # 如果K指定了采样点数量，并且当前批次的体素数量多于K，则仅保留点云数量最多的K个体素
            if K is not None and unique_id.shape[0] > K:
                _, cnt_topk_id = torch.topk(cnt, k=K)
                unique_id = unique_id[cnt_topk_id]

            # 将保留的点云添加到采样点列表中
            sampled_points.append(b_pcd[unique_id])

        # 如果K指定了采样点数量，则对采样点进行padding处理，使得每个批次的点云数目都等于K
        if K is not None:
            padding_mask = torch.zeros(size=(B, K), dtype=torch.bool, device=device)
            for i, b_pcd in enumerate(sampled_points):
                if b_pcd.shape[0] < K:
                    # 如果当前批次的采样点数量少于K，则用零值填充
                    zero_padding = torch.zeros(size=(K - b_pcd.shape[0], b_pcd.shape[1]),
                                               device=b_pcd.device, dtype=b_pcd.dtype)
                    sampled_points[i] = torch.cat((b_pcd, zero_padding), dim=0)
                    # 将填充部分的掩码设置为True
                    padding_mask[i, b_pcd.shape[0]:] = True
        else:
            # 如果没有指定K，则假设只有一个批次，并生成相应的padding掩码
            assert len(sampled_points) == 1
            padding_mask = torch.zeros(size=(1, sampled_points[0].shape[0]), dtype=torch.bool, device=device)

        # 将采样点列表堆叠为张量，形状为(B, K, 3+)
        sampled_points = torch.stack(sampled_points, dim=0)

        # 返回采样点和对应的padding掩码
        return sampled_points, padding_mask

    @staticmethod
    def fps(points: Tensor, points_padding: Tensor, K: int, original_indices: Tensor,
            random_start_point: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """
        最远点采样，从一个初始点开始，循环地将与当前采样点距离最远的点当作下一个采样点，直至满足采样点的数量需求。

        :param points: (B, N, 3+) 上一次下采样后的点云
        :param points_padding: (B, N) 原始点云的掩码，True为填充点，False为其他点，忽略填充点
        :param K: 采样点数量
        :param original_indices: (B, N) 原始点云中的索引，范围为1-16384
        :param random_start_point: 初始点是否随机选取，若否则固定从索引为0的点开始
        :return: 采样点(B, K, 3+), 采样点padding掩码(B, K), 采样点的原始索引(B, K)
        """
        # 获取有效点的数量（即未被padding的点）
        lengths = (~points_padding).sum(1)

        # 提取xyz坐标，避免修改原始数据
        points_xyz = points[..., :3]

        B, P, D = points_xyz.shape  # B=batch size, P=number of points, D=dimensionality
        device = points_xyz.device

        # 确保有效点数不超过 P
        if lengths.max() > P:
            raise ValueError("Invalid lengths.")

        # 对于每个batch，确保K不超过有效点的数量
        K = min(K, lengths.max().item())

        # 初始化采样点索引列表
        sampled_indices = torch.full((B, K), -1, dtype=torch.int64, device=device)

        # 初始化最近距离，距离初始点为无穷大
        closest_dists = torch.full((B, P), float("inf"), dtype=torch.float32, device=device)

        # 初始化批次索引
        batch_indices = torch.arange(B, device=device)

        # 选择初始点：随机或固定索引为0
        if random_start_point:
            initial_idx = torch.randint(0, lengths.max(), (B,), dtype=torch.int64, device=device)
        else:
            initial_idx = torch.zeros(B, dtype=torch.int64, device=device)

        # 第一个采样点为初始点
        sampled_indices[:, 0] = initial_idx

        # 迭代选择K个最远点
        for i in range(1, K):
            # 获取上一次选择的点
            last_selected = sampled_indices[:, i - 1]

            # 计算上次采样点与所有点之间的距离平方 (B, P)
            dist_to_last_selected = (points_xyz[batch_indices, last_selected].unsqueeze(1) - points_xyz).pow(2).sum(
                dim=-1)

            # 更新到最近采样点的最小距离
            closest_dists = torch.min(closest_dists, dist_to_last_selected)

            # 选择下一个最远的点
            next_selected = torch.argmax(closest_dists, dim=1)
            sampled_indices[:, i] = next_selected

        # 从原始点索引中提取出采样点的索引
        sampled_original_indices = original_indices.gather(1, sampled_indices)

        # 提取采样点的坐标
        sampled_points = points.gather(1, sampled_indices.unsqueeze(-1).expand(-1, -1, D))

        # 生成 padding mask
        padding_mask = sampled_indices < 0

        return sampled_points, padding_mask, sampled_original_indices

    @staticmethod
    def fps_t3d(points: Tensor, points_padding: Tensor, K: int, original_indices: Tensor,
                random_start_point: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """
        最远点采样，从一个初始点开始，循环地将与当前采样点距离最远的点当作下一个采样点，直至满足采样点的数量需求。

        :param points: (B, N, 3+) 原始点云
        :param points_padding: (B, N) 原始点云的掩码，True为填充点，False为其他点，忽略填充点
        :param K: 采样点数量
        :param original_indices: (B, N) 原始点云中的索引
        :param random_start_point: 初始点是否随机选取，若否则固定从索引为0的点开始
        :return: 采样点(B, K, 3+), 采样点padding掩码(B, K), 采样点的原始索引(B, K)
        """

        if points.shape[-1] > 3:
            points_xyz = points[..., :3]
            idx = sample_farthest_points(points=points_xyz, lengths=(~points_padding).sum(1), K=K,
                                         random_start_point=random_start_point)[1]
            sampled_points = masked_gather(points, idx)
        else:
            sampled_points, idx = sample_farthest_points(points=points, lengths=(~points_padding).sum(1), K=K,
                                                         random_start_point=random_start_point)

        padding_mask = idx < 0

        idx[padding_mask] = original_indices.shape[1] - 1

        sampled_original_indices = original_indices.gather(1, idx)

        return sampled_points, padding_mask, sampled_original_indices


def coordinate_distance(src: Tensor, dst: Tensor) -> Tensor:
    """
    计算两个点集的各点间距
    !!!半精度[不要]使用化简的方法，否则会出现严重的浮点误差
    :param src: <torch.Tensor> (B, M, C) C为坐标
    :param dst: <torch.Tensor> (B, N, C) C为坐标
    :return: <torch.Tensor> (B, M, N)
    """
    # 获取输入点集的dtype（数据类型）
    dtype = src.dtype

    # 将输入的点集转换为float类型以确保计算精度
    src, dst = src.float(), dst.float()

    # 获取src和dst张量的批次大小B，以及点的数量M和N
    B, M, _ = src.shape
    _, N, _ = dst.shape

    # 计算src和dst之间的距离平方的负两倍，通过矩阵乘法实现
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))

    # 将src的坐标平方后求和，并加到距离矩阵上
    dist += torch.sum(src ** 2, -1).view(B, M, 1)

    # 将dst的坐标平方后求和，并加到距离矩阵上
    dist += torch.sum(dst ** 2, -1).view(B, 1, N)

    # 将距离
    return dist.type(dtype)

def masked_gather(points: Tensor, idx: Tensor) -> Tensor:
    """
    使用torch.gather函数收集指定索引处的点，处理索引中可能出现的-1（表示填充）情况。
    首先将这些-1的索引替换为0，然后在收集完点之后将这些填充位置的值设为0.0。

    Args:
        points: (N, P, D) float32张量，表示点云数据
        idx: (N, K) 或 (N, P, K) 的长整型张量，表示要收集的点的索引，其中有些索引可能为-1，表示填充

    Returns:
        selected_points: (N, K, D) float32张量，表示在给定索引处收集到的点
    """

    # 确保points和idx具有相同的批次维度N
    if len(idx) != len(points):
        raise ValueError("points and idx must have the same batch dimension")

    # 获取points张量的形状：批次大小N，点的数量P，以及每个点的维度D
    N, P, D = points.shape

    if idx.ndim == 3:
        # 处理KNN或Ball Query的情况，idx形状为(N, P', K)，其中P'不一定与P相同，因为可能从不同的点云中收集点。
        K = idx.shape[2]

        # 将idx扩展维度以匹配points的维度
        idx_expanded = idx[..., None].expand(-1, -1, -1, D)

        # 扩展points的维度以匹配idx的维度
        points = points[:, :, None, :].expand(-1, -1, K, -1)
    elif idx.ndim == 2:
        # 处理最远点采样的情况，idx形状为(N, K)
        idx_expanded = idx[..., None].expand(-1, -1, D)
    else:
        # 如果idx的维度不符合预期，抛出错误
        raise ValueError("idx format is not supported %s" % repr(idx.shape))

    # 创建掩码，用于标记idx中等于-1的位置
    idx_expanded_mask = idx_expanded.eq(-1)

    # 克隆idx_expanded，以防止直接修改原始idx
    idx_expanded = idx_expanded.clone()

    # 将idx中等于-1的值替换为0，以便后续使用gather函数
    idx_expanded[idx_expanded_mask] = 0

    # 使用torch.gather函数根据idx从points中收集点
    selected_points = points.gather(dim=1, index=idx_expanded)

    # 将填充位置的点的值设为0.0
    selected_points[idx_expanded_mask] = 0.0

    # 返回收集到的点
    return selected_points


def index_points(points: Tensor, idx: Tensor) -> Tensor:
    """
    根据采样点索引获取其原始点云xyz坐标等信息

    :param points: (B, N, 3+) 原始点云
    :param idx: (B, S)/(B, S, G) 采样点索引，S为采样点数量，G为每个采样点grouping的点数
    :return: (B, S, 3+)/(B, S, G, 3+) 获取了原始点云信息的采样点
    """
    # 获取批次大小B
    B = points.shape[0]

    # 获取idx的形状，将其转换为列表形式以便后续操作
    view_shape = list(idx.shape)

    # 修改view_shape列表，从第二个维度开始设为1，这样可以扩展索引的维度
    view_shape[1:] = [1] * (len(view_shape) - 1)

    # 复制idx的形状，用于后续的维度扩展
    repeat_shape = list(idx.shape)

    # 将repeat_shape的第一个维度设为1，以便后续重复批次索引
    repeat_shape[0] = 1

    # 生成批次索引，形状为(B, 1, 1,...)
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)

    # 使用批次索引和采样点索引从原始点云中获取对应的点
    new_points = points[batch_indices, idx, :]

    # 返回采样得到的点，形状为(B, S, 3+)或(B, S, G, 3+)
    return new_points


def build_mlp(in_channel: int,  # 输入特征维度
              channel_list: List[int],  # 每层MLP的输出通道数的列表
              dim: int = 2,     # 维度
              bias: bool = False,   # 是否使用偏置
              drop_last_act: bool = False,  # 是否移除最后一层的激活函数
              norm: Literal['bn', 'ln', 'in'] = 'bn',   # 归一化层类型
              act: Literal['relu', 'elu'] = 'relu') -> nn.Sequential:   # 激活函数类型
    """
    构造基于n维度的1x1卷积的MLP（多层感知器）

    :param in_channel: <int> 输入的特征维度（通道数）
    :param channel_list: <list[int]> 每层MLP的输出通道数的列表
    :param dim: <int> 维度，1维或2维
    :param bias: <bool> 卷积层是否使用偏置，一般BN层前的卷积层不使用偏置
    :param drop_last_act: <bool> 是否移除最后一层的激活函数
    :param norm: 归一化层类型，支持'bn'（BatchNorm）、'ln'（LayerNorm）和'in'（InstanceNorm）
    :param act: 激活函数类型，支持'relu'和'elu'
    :return: <torch.nn.ModuleList[torch.nn.Sequential]> 构建的MLP模型
    """

    # 定义1维归一化层的映射
    norm_1d = {'bn': nn.BatchNorm1d,  # 批量归一化，通常用于标准化每个小批量中的数据分布，有助于加快训练过程并提高模型的稳定性
               'in': nn.InstanceNorm1d,  # 实例归一化，类似于批量归一化，但在每个实例（而不是整个小批量）内执行归一化，通常用于风格迁移或生成模型中
               'ln': LayerNorm1d}  # 层归一化，在同一层内进行归一化，适用于小批量大小较小或变动较大的场景

    # 定义2维归一化层的映射
    norm_2d = {'bn': nn.BatchNorm2d,  # 批量归一化，通常用于标准化每个小批量中的数据分布，常用于卷积神经网络（CNN）中
               'in': nn.InstanceNorm2d,  # 实例归一化，执行实例级别的归一化，在每个样本内进行归一化，通常用于风格迁移和生成模型中
               'ln': LayerNorm2d}  # 层归一化，在每一层进行归一化，独立于批量大小，常用于小批量的模型

    # 定义激活函数的映射
    acts = {'relu': nn.ReLU,  # ReLU激活函数
            'elu': nn.ELU}  # ELU激活函数

    # 根据dim参数选择1维或2维的卷积层和归一化层
    if dim == 1:
        Conv = nn.Conv1d  # 1维卷积
        NORM = norm_1d.get(norm.lower(), nn.BatchNorm1d)  # 获取指定的归一化层类型，默认为BatchNorm1d
    else:
        Conv = nn.Conv2d  # 2维卷积
        NORM = norm_2d.get(norm.lower(), nn.BatchNorm2d)  # 获取指定的归一化层类型，默认为BatchNorm2d

    # 获取指定的激活函数类型，默认为ReLU
    ACT = acts.get(act.lower(), nn.ReLU)

    # 开始构建MLP模型
    mlp = []
    for channel in channel_list:
        # 对于每个指定的通道数，构建一层卷积、归一化、激活函数
        mlp.append(Conv(in_channels=in_channel, out_channels=channel, kernel_size=1, bias=bias))
        mlp.append(NORM(channel))
        mlp.append(ACT(inplace=True))

        # 更新输入通道数，以便为下一层准备
        in_channel = channel

    # 如果drop_last_act为True，则移除最后一层的激活函数
    if drop_last_act:
        mlp = mlp[:-1]

    # 将所有层打包为一个nn.Sequential模块并返回
    return nn.Sequential(*mlp)


class LayerNorm1d(nn.Module):
    """封装LayerNorm，支持(B, C, N)作为输入"""

    def __init__(self, channel):
        # 调用父类nn.Module的初始化函数
        super().__init__()
        # 创建一个nn.LayerNorm实例，用于1维数据的归一化
        self.ln = nn.LayerNorm(channel)

    def forward(self, x):
        """(B, C, N)"""
        # 对输入张量进行LayerNorm处理
        # x.transpose(1, 2)将维度从(B, C, N)转换为(B, N, C)
        out = self.ln(x.transpose(1, 2)).transpose(1, 2)
        # 然后再将维度转换回(B, C, N)
        return out



class LayerNorm2d(nn.Module):
    """封装LayerNorm，支持(B, C, H, W)作为输入"""

    def __init__(self, channel):
        # 调用父类nn.Module的初始化函数
        super().__init__()
        # 创建一个nn.LayerNorm实例，用于2维数据的归一化
        self.ln = nn.LayerNorm(channel)

    def forward(self, x):
        """(B, C, H, W)"""
        # 对输入张量进行LayerNorm处理
        # x.permute(0, 2, 3, 1)将维度从(B, C, H, W)转换为(B, H, W, C)
        out = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # 然后再将维度转换回(B, C, H, W)
        return out

