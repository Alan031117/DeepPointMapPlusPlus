import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as Tensor


def CoarsePairingHead(emb_dim: int):
    # 粗匹配头,使输入特征更容易匹配
    return nn.Sequential(
        nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=1),  # 一维卷积层，保持特征维度不变
        nn.ReLU(inplace=True),  # ReLU激活函数
        nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=1)  # 再次使用一维卷积层
    )


def SimilarityHead(emb_dim: int):
    # 创建一个相似性头部网络，用于计算特征之间的相似性
    return nn.Sequential(
        nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=1),  # 一维卷积层，保持特征维度不变
        nn.ReLU(inplace=True),  # ReLU激活函数
        nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=1)  # 再次使用一维卷积层
    )


class OffsetHead(nn.Module):
    """预测匹配点间的相对偏移量"""
    def __init__(self, emb_dim: int, coor_dim: int = 3):
        super().__init__()
        # 多层感知机（MLP）用于逐步减少特征维度
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim // 2, kernel_size=1),  # 第一层一维卷积层，减少特征维度为一半
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv1d(in_channels=emb_dim // 2, out_channels=emb_dim // 4, kernel_size=1),  # 第二层一维卷积层，继续减少特征维度为原来四分之一
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv1d(in_channels=emb_dim // 4, out_channels=emb_dim // 8, kernel_size=1),  # 第三层一维卷积层，减少特征维度为原来八分之一
        )
        # 下采样操作，用于匹配MLP后的特征维度
        self.downsample = nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim // 8, kernel_size=1)
        # 预测头，用于计算最终的偏移量
        self.head = nn.Conv1d(in_channels=emb_dim // 8, out_channels=coor_dim, kernel_size=1)
        self.act = nn.ReLU(inplace=True)  # ReLU激活函数

    def forward(self, pcd_fea: Tensor) -> Tensor:
        """
        :param pcd_fea: (B, C, N) 输入的点云特征
        :return: (B, coor_dim, N) 输出点云坐标的偏移量
        """
        out = self.mlp(pcd_fea)  # 将输入特征通过MLP处理
        identity = self.downsample(pcd_fea)  # 对输入特征进行下采样，作为跳跃连接的部分
        out = self.act(out + identity)  # 将处理后的特征与下采样特征相加，并通过激活函数处理
        out = self.head(out)  # 通过预测头生成最终的偏移量
        return out  # 返回计算出的偏移量


class LoopHead(nn.Module):
    """ 回环检测头，预测两帧之间是否产生回环 """
    def __init__(self, emb_dim: int):
        super().__init__()

        # 神经网络构建
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=1)
        )
        self.projection = nn.Sequential(
            nn.Linear(in_features=2 * emb_dim, out_features=2 * emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2 * emb_dim, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, src_fea, dst_fea, *args) -> Tensor:

        # 提取：源特征、目标特征
        src_fea = self.mlp(src_fea)
        dst_fea = self.mlp(dst_fea)

        # 特征维度从 (B, C, N) 变为到 (B, C)
        src_fea = torch.mean(src_fea, dim=-1)
        dst_fea = torch.mean(dst_fea, dim=-1)

        # [源特征, 目标特征] 经卷积层后 --> 回环概率
        loop_pro = self.projection(torch.cat([src_fea, dst_fea], dim=-1)).flatten()  # (B,)

        return loop_pro  # 回环概率


class LoopHeadV2(nn.Module):
    """ 回环检测头，预测两帧之间是否产生回环 """
    def __init__(self, emb_dim: int):
        super().__init__()

        # 神经网络构建
        self.query = nn.Parameter(torch.randn(emb_dim, ), requires_grad=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=emb_dim, nhead=8, dropout=0, batch_first=True, dim_feedforward=256),
            num_layers=2,
        )
        self.projection = nn.Sequential(
            nn.Linear(in_features=2 * emb_dim, out_features=2 * emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2 * emb_dim, out_features=1),
        )

    def forward(self, src_fea, dst_fea, src_padding_mask=None, dst_padding_mask=None) -> Tensor:
        src_fea, dst_fea = src_fea.transpose(1, 2), dst_fea.transpose(1, 2)  # (B, N, C)
        if src_padding_mask is not None:
            src_padding_mask = F.pad(src_padding_mask, (1, 0), mode='constant', value=False)  # (B, 1+N)
        if dst_padding_mask is not None:
            dst_padding_mask = F.pad(dst_padding_mask, (1, 0), mode='constant', value=False)  # (B, 1+N)

        # 提取：源特征、目标特征
        src_map_fea = self.query[None, None, :].repeat(src_fea.shape[0], 1, 1)
        dst_map_fea = self.query[None, None, :].repeat(dst_fea.shape[0], 1, 1)
        src_fea = torch.cat([src_map_fea, src_fea], dim=1)  # (B, 1+N, C)
        dst_fea = torch.cat([dst_map_fea, dst_fea], dim=1)  # (B, 1+N, C)
        src_fea = self.transformer_encoder(src_fea, src_key_padding_mask=src_padding_mask)
        dst_fea = self.transformer_encoder(dst_fea, src_key_padding_mask=dst_padding_mask)
        src_map_fea = src_fea[:, 0, :]  # (B, C)
        dst_map_fea = dst_fea[:, 0, :]

        # [源特征, 目标特征] 经卷积层后 --> 回环概率
        loop_pro = self.projection(torch.cat([src_map_fea, dst_map_fea], dim=-1)).flatten()  # (B,)
        loop_pro = F.sigmoid(loop_pro)

        return loop_pro  # 回环概率

