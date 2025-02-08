import torch
import torch.nn as nn
from torch import Tensor as Tensor
from typing import Literal, Tuple, List
from network.encoder.utils import Querier, Sampler, index_points, coordinate_distance, build_mlp
import time

class SetAbstraction(nn.Module):
    """
    点云特征提取模块
    包含一个单尺度的采样-分组-特征提取 (S-G-P) 过程
    """

    def __init__(self,
                 npoint: int,  # 采样点数量
                 radius: float,  # 采样半径
                 nsample: int,  # 采样点邻域内的采样点数量
                 in_channel: int,  # 输入特征维度
                 sample: dict,  # 下采样方式
                 norm: Literal['bn', 'ln', 'in'] = 'ln',  # 归一化方式，默认为 'ln' (LayerNorm)
                 bias: bool = True):  # 是否使用偏置，默认为 True

        assert 'type' in sample.keys(), f'key \'type\' must be in sample dict'  # 确认 sample 字典中存在 'type' 键
        assert sample['type'] in ['fps', 'voxel', 'fps-t3d'], f'{sample} is not a supported sampling way, ' \
                                                              f'please use \'fps\' or \'voxel\''
        # 确保采样方式是支持的类型：'fps'（最远点采样）、'voxel'（体素采样）、'fps-t3d'（最远点采样在PyTorch3D中实现）

        super().__init__()
        self.npoint = npoint  # 采样点数量
        self.radius = radius  # 采样半径
        self.nsample = nsample  # 采样点邻域内的采样点数量
        self.in_channel = in_channel  # 输入特征维度

        # 构建MLP，用于特征提取，输入维度为 in_channel + 3，输出维度为 in_channel * 2
        self.mlp = build_mlp(in_channel=in_channel + 3, channel_list=[in_channel * 2], dim=2, norm=norm, bias=bias)

        sample_type = sample['type']  # 获取采样方式类型
        if sample_type == 'voxel':
            # 如果是体素采样，需要额外设置体素大小和采样范围
            assert 'size' in sample.keys() and 'range' in sample.keys()
            self.sample_kwargs = {'voxel_size': sample['size'], 'sample_range': sample['range']}
        else:
            self.sample_kwargs = {}  # 否则，不需要额外的采样参数

        self.sample = Sampler(sample_type)  # 初始化采样器
        self.query = Querier('hybrid-t3d')  # 初始化查询器，用于查找采样点的邻域点

    def forward(self, points_coor: Tensor, points_fea: Tensor, points_padding: Tensor, points_index: Tensor) -> (
            Tuple)[Tensor, Tensor, Tensor, Tensor]:
        """
        :param points_coor: (B, 3, N) 输入点云的坐标 (批量大小, 3, 点的数量)
        :param points_fea: (B, C, N) 特征点云数据 (批量大小, 特征维度, 点的数量)
        :param points_padding: (B, N) 点云的填充掩码，True为填充点，False为正常点
        :return:
            new_coor: (B, 3, S) 下采样后的点云坐标
            new_fea: (B, D, S) 下采样后的点云特征
            new_padding: (B, S) 下采样后的点云填充掩码，True为填充点，False为正常点
        """
        points_coor, points_fea = points_coor.permute(0, 2, 1), points_fea.permute(0, 2, 1)
        # 转置张量，使其形状从 (B, 3, N) 变为 (B, N, 3)，从 (B, C, N) 变为 (B, N, C)

        bs, nbr_point_in, _ = points_coor.shape  # 获取批量大小和点的数量
        num_point_out = self.npoint  # 设置输出点的数量

        '''S 采样'''
        start_time = time.time()
        new_coor, new_mask, new_index = self.sample(points=points_coor, points_padding=points_padding,
                                                    K=num_point_out, original_indices=points_index,
                                                    **self.sample_kwargs)

        '''G 采样'''
        start_time = time.time()
        # 查询采样点邻域内的点，并获取这些点的索引
        group_idx = self.query(radius=self.radius, K=self.nsample, points=points_coor, centers=new_coor,
                               points_padding=points_padding)

        # 使用索引获取每个group内的点云坐标和特征，并进行坐标归一化处理
        grouped_points_coor = index_points(points_coor[..., :3], group_idx)  # 获取group内点云坐标 (B, S, K, 3)
        grouped_points_coor -= new_coor[..., :3].view(bs, num_point_out, 1, 3)  # 将坐标转化为与采样点的偏移量 (B, S, K, 3)
        grouped_points_coor = grouped_points_coor / self.radius  # 坐标归一化 (B, S, K, 3)

        grouped_points_fea = index_points(points_fea, group_idx)  # 获取group内点云特征 (B, S, K, C)
        grouped_points_fea = torch.cat([grouped_points_fea, grouped_points_coor], dim=-1)  # 拼接坐标偏移量 (B, S, K, C+3)

        end_time = time.time()

        '''P 特征提取'''
        start_time = time.time()

        # 通过MLP提取特征，形状从 (B, S, K, C+3) 变为 (B, C+3, K, S)，然后通过池化层获取特征 (B, D, S)
        grouped_points_fea = grouped_points_fea.permute(0, 3, 2, 1)  # 转置为 (B, C_in+3, K, S)
        grouped_points_fea = self.mlp(grouped_points_fea)  # 通过 MLP 提取特征 (B, C_out, K, S)
        new_fea = torch.max(grouped_points_fea, dim=2)[0]  # 池化操作，获取每个group内的最大特征 (B, C_out, S)

        new_coor = new_coor.permute(0, 2, 1)  # 将坐标张量转置回 (B, 3, S)

        end_time = time.time()

        return new_coor, new_fea, new_mask, new_index  # 返回下采样后的点云坐标、特征、掩码、索引


class LocalAggregation(nn.Module):
    """
    局部特征提取模块
    包含一个单尺度的分组-特征提取 (G-P) 过程，每个点都作为采样点进行分组以聚合局部特征，无下采样过程
    """

    def __init__(self,
                 radius: float,
                 nsample: int,
                 in_channel: int,
                 norm: Literal['bn', 'ln', 'in'] = 'ln',
                 bias: bool = True):
        """
        :param radius: 采样半径
        :param nsample: 每个分组内的采样点数量
        :param in_channel: 输入特征的维度
        :param norm: 归一化方式，默认为 'ln' (LayerNorm)
        :param bias: 是否使用偏置，默认为 True
        """
        super().__init__()  # 初始化父类 nn.Module
        self.radius = radius  # 设置采样半径
        self.nsample = nsample  # 设置每个分组内的采样点数量
        self.in_channel = in_channel  # 设置输入特征的维度

        # 构建MLP，用于特征提取，输入维度为 in_channel + 3，输出维度为 in_channel
        self.mlp = build_mlp(in_channel=in_channel + 3, channel_list=[in_channel], dim=2, norm=norm, bias=bias)

        # 初始化查询器，用于查找采样点的邻域点
        self.query = Querier('hybrid-t3d')

    def forward(self, points_coor: Tensor, points_fea: Tensor, points_padding: Tensor) -> Tensor:
        """
        :param points_coor: (B, 3, N) 输入点云的坐标 (批量大小, 3, 点的数量)
        :param points_fea: (B, C, N) 输入点云的特征 (批量大小, 特征维度, 点的数量)
        :param points_padding: (B, N) 点云的填充掩码，True为填充点，False为正常点
        :return:
            new_fea: (B, D, N) 局部特征聚合后的特征
        """
        # 将输入的点云坐标和特征转置，变为 (B, N, 3) 和 (B, N, C)
        points_coor, points_fea = points_coor.permute(0, 2, 1), points_fea.permute(0, 2, 1)
        bs, npoint, _ = points_coor.shape  # 获取批量大小和点的数量

        '''G 分组'''
        # 查询每个采样点的邻域点，并获取这些点的索引 (B, N, K)
        group_idx = self.query(radius=self.radius, K=self.nsample, points=points_coor, centers=points_coor,
                               points_padding=points_padding)

        # 使用索引获取每个group内的点云坐标，并计算相对于采样点的偏移量，进行归一化
        grouped_points_coor = index_points(points_coor[..., :3], group_idx)  # 获取group内点云坐标 (B, N, K, 3)
        grouped_points_coor = grouped_points_coor - points_coor[..., :3].view(bs, npoint, 1, 3)  # 计算偏移量 (B, N, K, 3)
        grouped_points_coor = grouped_points_coor / self.radius  # 对偏移量进行归一化 (B, N, K, 3)

        # 使用索引获取每个group内的点云特征，并拼接坐标偏移量
        grouped_points_fea = index_points(points_fea, group_idx)  # 获取group内点云特征 (B, N, K, C)
        grouped_points_fea = torch.cat([grouped_points_fea, grouped_points_coor], dim=-1)  # 拼接特征和坐标 (B, N, K, C+3)

        '''P 特征提取'''
        # 通过MLP提取特征，形状从 (B, N, K, C+3) 变为 (B, C+3, K, N)，再通过池化层获取局部特征 (B, D, N)
        grouped_points_fea = grouped_points_fea.permute(0, 3, 2, 1)  # 转置为 (B, C+3, K, N)
        grouped_points_fea = self.mlp(grouped_points_fea)  # 通过 MLP 提取特征 (B, D, K, N)
        new_fea = torch.max(grouped_points_fea, dim=2)[0]  # 对每个group内的特征进行池化，获取最大特征 (B, D, N)

        return new_fea  # 返回聚合后的局部特征


class InvResMLP(nn.Module):
    """
    逆瓶颈残差块
    """

    def __init__(self,
                 radius: float,
                 nsample: int,
                 in_channel: int,
                 expansion: int = 4,
                 norm: Literal['bn', 'ln', 'in'] = 'ln',
                 bias: bool = True):
        """
        :param radius: 采样半径
        :param nsample: 采样点数量
        :param in_channel: 输入特征的维度
        :param expansion: 中间层通道数的扩张倍数，默认为4
        :param norm: 归一化方式，默认为'ln' (LayerNorm)
        :param bias: 是否使用偏置，默认为True
        """
        super().__init__()  # 调用父类 nn.Module 的初始化函数
        self.la = LocalAggregation(radius=radius, nsample=nsample, in_channel=in_channel, norm=norm, bias=bias)
        # 构建逆瓶颈结构的MLP层，包含两个全连接层，中间层通道数为 in_channel 的 expansion 倍，最后恢复为 in_channel
        channel_list = [in_channel * expansion, in_channel]
        self.pw_conv = build_mlp(in_channel=in_channel, channel_list=channel_list, dim=1, drop_last_act=True,
                                 norm=norm, bias=bias)
        self.act = nn.ReLU(inplace=True)  # 激活函数为ReLU

    def forward(self, points: List[Tensor]) -> List[Tensor]:
        """
        :param points:
            包含以下3个元素的列表：
            (B, 3, N) 点云原始坐标
            (B, C, N) 点云特征
            (B, N) 点云填充掩码，True为填充点，False为正常点
        :return:
            (B, 3, N) 点云原始坐标
            (B, D, N) 点云新特征
            (B, N) 点云填充掩码，True为填充点，False为正常点
        """
        points_coor, points_fea, points_padding = points  # 解包输入的点云坐标、特征和填充掩码
        identity = points_fea  # 保存输入的点云特征作为残差
        points_fea = self.la(points_coor, points_fea, points_padding)  # 通过局部特征聚合模块提取特征
        points_fea = self.pw_conv(points_fea)  # 通过逆瓶颈结构的MLP层进一步提取特征
        points_fea = points_fea + identity  # 将提取的特征与残差相加，形成残差连接
        points_fea = self.act(points_fea)  # 经过激活函数处理
        return [points_coor, points_fea, points_padding]  # 返回更新后的点云坐标、特征和填充掩码


class Stage(nn.Module):
    """
    PointNeXt一个下采样阶段
    """

    '''参数均源于输入'''

    def __init__(self,
                 npoint: int,  # 采样点数量
                 radius_list: List[float],  # 多尺度采样半径列表
                 nsample_list: List[int],  # 多尺度采样邻点数量列表
                 in_channel: int,  # 输入特征的维度
                 sample: dict,  # 下采样方式的字典
                 expansion: int = 4,  # 中间层通道数的扩张倍数，默认为4
                 norm: Literal['bn', 'ln', 'in'] = 'ln',  # 归一化方式，默认为'ln' (LayerNorm)
                 bias: bool = True):  # 是否使用偏置，默认为True

        assert len(radius_list) == len(nsample_list)  # 确保 len(半径列表) = len(邻点数量列表)
        super().__init__()  # 调用父类 nn.Module 的初始化函数
        # 创建 SetAbstraction 模块，用于下采样和特征提取
        self.sa = SetAbstraction(npoint=npoint, radius=radius_list[0], nsample=nsample_list[0],
                                 in_channel=in_channel, sample=sample, norm=norm, bias=bias)
        irm = []  # 用于存储多个 InvResMLP 模块
        for i in range(1, len(radius_list)):
            irm.append(
                InvResMLP(radius=radius_list[i], nsample=nsample_list[i], in_channel=in_channel * 2,
                          expansion=expansion, norm=norm, bias=bias)
            )
        if len(irm) > 0:
            self.irm = nn.Sequential(*irm)  # 如果存在多个尺度，使用 nn.Sequential 进行顺序排列
        else:
            self.irm = nn.Identity()  # 如果没有更多的尺度，则使用 nn.Identity 作为占位符

    def forward(self, points_coor: Tensor, points_fea: Tensor, points_padding: Tensor, points_index:Tensor) -> Tuple[
        Tensor, Tensor, Tensor, Tensor]:
        """
        :param points_coor: (B, 3, N) 点云原始坐标
        :param points_fea: (B, D, N) 点云特征
        :param points_padding: (B, N) 点云填充掩码，True为填充点，False为正常点
        :return:
            new_coor: (B, 3, S) 下采样后的点云坐标
            new_fea: (B, D', S) 下采样后的点云特征
            new_padding: (B, S) 下采样后的点云填充掩码，True为填充点，False为正常点
        """
        new_coor, new_fea, new_padding, new_index = self.sa(points_coor, points_fea,
                                                            points_padding, points_index)  # 下采样和初步特征提取
        new_coor, new_fea, new_padding = self.irm([new_coor, new_fea, new_padding])  # 通过多个 InvResMLP 模块进行进一步特征提取
        return new_coor, new_fea, new_padding, new_index  # 返回下采样后的点云坐标、特征和填充掩码


class FeaturePropagation(nn.Module):
    """
    FP上采样模块，用于在点云中进行特征的上采样（传播）。
    """

    def __init__(self,
                 in_channel: List[int],
                 mlp: List[int],
                 norm: Literal['bn', 'ln', 'in'] = 'ln',
                 bias: bool = True):
        """
        :param in_channel: 输入的特征维度列表，包含同层和下层的特征维度
        :param mlp: MLP的通道维度列表，用于构建MLP层
        :param norm: 归一化方式，可以选择批归一化(bn)、层归一化(ln)或实例归一化(in)
        :param bias: 是否使用偏置，默认为True
        """
        super(FeaturePropagation, self).__init__()  # 调用父类 nn.Module 的初始化函数
        # 构建MLP层，用于处理拼接后的特征
        self.mlp = build_mlp(in_channel=sum(in_channel), channel_list=mlp, dim=1, norm=norm, bias=bias)

    def forward(self, points_coor1: Tensor, points_coor2: Tensor, points_fea1: Tensor, points_fea2: Tensor,
                points_padding2: Tensor) -> Tensor:
        """
        :param points_coor1: (B, 3, N) 同层点云原始坐标
        :param points_coor2: (B, 3, S) 下层点云原始坐标
        :param points_fea1: (B, D1, N) 同层点云特征
        :param points_fea2: (B, D2, S) 下层点云特征
        :param points_padding2: (B, S) 下层点云填充掩码
        :return: (B, D, N) 上采样后的点云特征
        """
        B, _, N = points_coor1.shape  # 获取同层点云的批次大小、维度数和点数量
        _, _, S = points_coor2.shape  # 获取下层点云的点数量

        if S == 1:
            # 如果下层只有一个特征点，则直接将其扩展至所有同层点
            new_fea = points_fea2.repeat(1, 1, N)
        else:
            # 将坐标和特征的维度从 (B, C, N) 转换为 (B, N, C) 以便计算
            points_coor1, points_coor2, points_fea2 = \
                points_coor1.transpose(1, 2), points_coor2.transpose(1, 2), points_fea2.transpose(1, 2)

            # 将下层点云的填充点（无效点）调整至远离原始点的位置，避免影响后续计算
            points_coor2 = points_coor2.clone()
            points_coor2[points_padding2] = points_coor2.abs().max() * 3

            # 计算每个同层点与下层点的距离，并选择距离最近的3个下层点
            dists = coordinate_distance(points_coor1[..., :3], points_coor2[..., :3])
            dists, idx = torch.topk(dists, k=3, dim=-1, largest=False)  # 基于距离选择最近的点 (B, N, 3)

            # 基于距离进行特征值加权求和，以计算新的特征值
            dist_recip = 1.0 / dists.float().clamp(min=1e-8)  # 计算距离的倒数
            norm = torch.sum(dist_recip, dim=2, keepdim=True)  # 计算归一化因子
            weight = dist_recip / norm  # 计算每个最近邻点的加权比例
            interpolated_points = torch.sum(index_points(points_fea2, idx) * weight.view(B, N, 3, 1), dim=2)
            # 将 (B, N, D2) 转换回 (B, D2, N)
            new_fea = interpolated_points.permute(0, 2, 1).type_as(points_fea2)

        # 将同层特征值与下层特征值进行拼接，并通过MLP进一步处理得到新的特征值 (B, D1+D2, N) -> (B, D, N)
        new_fea = torch.cat((points_fea1, new_fea), dim=1)
        new_fea = self.mlp(new_fea)
        return new_fea  # 返回上采样后的特征
