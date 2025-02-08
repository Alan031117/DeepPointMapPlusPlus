import torch
import torch.nn as nn
from torch import Tensor as Tensor
from typing import List
from network.encoder.pointnext import Stage, FeaturePropagation
import time

class Encoder(nn.Module):
    """
    基于 PointNeXt 和 FPN (特征金字塔网络) 的骨干网络，用于特征提取
    """
    def __init__(self, args):
        super().__init__()

        # ======================== 输入参数 ========================
        self.args = args
        self.encoder_cfg = self.args.encoder

        self.in_channel = self.encoder_cfg.in_channel
        self.out_channel = self.encoder_cfg.out_channel
        self.downsample_layers = len(self.encoder_cfg.npoint)
        self.upsample_layers = self.encoder_cfg.upsample_layers
        width = self.encoder_cfg.width
        norm = self.encoder_cfg.get('norm', 'LN').lower()
        bias = self.encoder_cfg.get('bias', True)

        # ======================== 构建网络 ========================
        self.point_mlp0 = nn.Conv1d(in_channels=self.in_channel, out_channels=width, kernel_size=1)
        # 第一个 MLP，一个 1D 卷积层，用于把每个点的特征维度改为 width

        self.downsampler = nn.ModuleList()  # 列表 downsampler：下采样层
        self.upsampler = nn.ModuleList()  # 列表 upsampler：上采样层

        for i in range(self.downsample_layers):  # 创建指定个数下采样层（下采样层个数及参数全部源于YAML）
            self.downsampler.append(
                Stage(npoint=self.encoder_cfg.npoint[i],
                      radius_list=self.encoder_cfg.radius_list[i],
                      nsample_list=self.encoder_cfg.nsample_list[i],
                      in_channel=width,
                      sample=self.encoder_cfg.sample[i],
                      expansion=self.encoder_cfg['expansion'],
                      norm=norm,
                      bias=bias))
            width *= 2  # 每经过一次下采样，特征通道数翻倍，用于捕获更复杂的特征

        upsampler_in = width  # 上采样层的输入通道数初始化为下采样的最后一层输出通道数
        for i in range(self.upsample_layers):
            upsampler_out = max(self.out_channel, width // 2)  # 确保上采样的输出通道数不低于 out_channel
            self.upsampler.append(
                FeaturePropagation(in_channel=[upsampler_in, width // 2],
                                   mlp=[upsampler_out, upsampler_out],
                                   norm=norm,
                                   bias=bias))
            width = width // 2  # 每经过一次上采样，特征通道数减半
            upsampler_in = upsampler_out  # 更新上采样层的输入通道数

    def forward(self, points: Tensor, points_padding: Tensor) -> List[Tensor]:
        """FBH
            将输入点云的坐标和特征分开处理
                l0_coor：所有点云初始坐标 (S, 3, 16384)
                l0_fea：所有初始点云的所有特征 (S, 3, 16384)
                l0_padding：所有点云的填充掩码 (S, 16384) ，True为忽略点，False为正常点
        """
        l0_coor, l0_fea, l0_padding = points[:, :3, :].clone(), points[:, :self.in_channel, :].clone(), points_padding
        l0_index = torch.arange(l0_coor.shape[-1], device=l0_coor.device).unsqueeze(0).repeat(l0_coor.shape[0], 1)
        l0_fea = self.point_mlp0(l0_fea)  # l0_fea过第一个MLP（一个1D卷积层），将l0_fea调整至[B,width,N]，width此时为16
        recorder = [[l0_coor, l0_fea, l0_padding, l0_index]]  # 记录第一层处理后的坐标、特征、掩码


        # 下采样（剔除数据）
        for layer in self.downsampler:
            new_coor, new_fea, new_padding, new_index = layer(*recorder[-1])   # 将recorder[-1]解包后输入当前下采样层
            recorder.append([new_coor, new_fea, new_padding, new_index])   # 记录新一层的坐标、特征、掩码、索引
            # recoder[[预],[下1],[下2],[下3],[下4],[下5]]

        # 上采样（内插数据）
        for i, layer in enumerate(self.upsampler):
            points_coor1, points_fea1, points_padding1, points_index1 = recorder[self.downsample_layers - i - 1]  # 获取对应的浅层特征
            points_coor2, points_fea2, points_padding2, points_index2 = recorder[-1]  # 获取当前的深层特征
            new_points_fea1 = layer(points_coor1, points_coor2, points_fea1, points_fea2, points_padding2)
            recorder.append([points_coor1.clone(), new_points_fea1, points_padding1.clone(), points_index1.clone()])
            # recoder[[预],[下1],[下2],[下3],[下4],[下5],[上1(深层下5,浅层下4)],[上2(深层上1,浅层下3)]]

        return recorder[-1]  # 返回最终上采样后的坐标、特征、掩码、索引
