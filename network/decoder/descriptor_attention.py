import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as Tensor
from typing import Tuple


class DescriptorAttentionLayer(nn.Module):
    """
    交互两组descriptor
    self-attn ==> cross-attn ==> mlp
    """
    def __init__(self, emb_dim: int):
        """
        :param emb_dim: 输入的维度
        """
        super().__init__()
        # 自注意力机制，用于处理每个输入自身的信息
        self.self_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=8, batch_first=True, dropout=0)
        # 交叉注意力机制，用于处理两个输入之间的相互关系
        self.cross_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=8, batch_first=True, dropout=0)
        # 多层感知机（MLP），用于对经过注意力机制处理后的特征进行进一步的非线性变换
        self.mlp = nn.Sequential(
            nn.Linear(in_features=emb_dim, out_features=emb_dim),  # 全连接层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Linear(in_features=emb_dim, out_features=emb_dim))  # 全连接层
        # LayerNorm，用于对特征进行归一化处理，减少梯度消失或爆炸的风险
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.norm3 = nn.LayerNorm(emb_dim)

    def forward(self, src_fea: Tensor, dst_fea: Tensor, src_pos_embedding: Tensor, dst_pos_embedding: Tensor,
                src_padding_mask=None, dst_padding_mask=None) -> Tuple[Tensor, Tensor]:
        # (B, C, N) -> (B, N, C) 多头注意力模块中的特征维度为最后一维
        src_fea, dst_fea = src_fea.transpose(1, 2), dst_fea.transpose(1, 2)
        src_pos_embedding, dst_pos_embedding = src_pos_embedding.transpose(1, 2), dst_pos_embedding.transpose(1, 2)

        # 自注意力机制
        src_fea = src_fea + src_pos_embedding  # 将位置编码添加到源特征上
        dst_fea = dst_fea + dst_pos_embedding  # 将位置编码添加到目标特征上
        out_fea, _out_weight = self.self_attn(query=src_fea, key=src_fea, value=src_fea, key_padding_mask=src_padding_mask)  # 源特征自注意力计算
        src_fea = self.norm1(src_fea + out_fea)  # 将自注意力输出与原始源特征相加并进行归一化
        out_fea, _out_weight = self.self_attn(query=dst_fea, key=dst_fea, value=dst_fea, key_padding_mask=dst_padding_mask)  # 目标特征自注意力计算
        dst_fea = self.norm1(dst_fea + out_fea)  # 将自注意力输出与原始目标特征相加并进行归一化

        # 交叉注意力机制
        src_fea = src_fea + src_pos_embedding  # 再次将位置编码添加到源特征上
        dst_fea = dst_fea + dst_pos_embedding  # 再次将位置编码添加到目标特征上
        src_out_fea, _src_out_weight = self.cross_attn(query=src_fea, key=dst_fea, value=dst_fea, key_padding_mask=dst_padding_mask)  # 源特征和目标特征之间的交叉注意力计算
        dst_out_fea, _dst_out_weight = self.cross_attn(query=dst_fea, key=src_fea, value=src_fea, key_padding_mask=src_padding_mask)  # 目标特征和源特征之间的交叉注意力计算
        src_fea = self.norm2(src_fea + src_out_fea)  # 将交叉注意力输出与源特征相加并进行归一化
        dst_fea = self.norm2(dst_fea + dst_out_fea)  # 将交叉注意力输出与目标特征相加并进行归一化

        # 多层感知机（MLP）操作：加法和归一化
        src_fea = self.norm3(self.mlp(src_fea) + src_fea)  # 将源特征通过MLP处理并与自身相加，然后归一化
        dst_fea = self.norm3(self.mlp(dst_fea) + dst_fea)  # 将目标特征通过MLP处理并与自身相加，然后归一化

        src_fea, dst_fea = src_fea.transpose(1, 2), dst_fea.transpose(1, 2)  # 转换回原始形状
        return src_fea, dst_fea  # 返回处理后的源特征和目标特征


class PositionEmbeddingCoordsSine(nn.Module):
    """
    多维正余弦位置编码(attention预处理)
    """
    def __init__(self, in_dim: int = 3, emb_dim: int = 256, temperature: int = 10000, scale: float = 1.0):
        """
        :param in_dim: 输入维度数
        :param emb_dim: 输出维度数
        :param temperature: 三角函数最大周期数
        :param scale: 缩放系数
        """
        super().__init__()

        self.in_channels = in_dim  # 输入维度数
        self.num_pos_feats = emb_dim // in_dim // 2 * 2  # 位置编码的特征数目，确保是偶数
        self.temperature = temperature  # 控制位置编码周期的参数
        self.padding = emb_dim - self.num_pos_feats * self.in_channels  # 计算需要填充的维度数
        self.scale = scale * math.pi  # 缩放因子，默认值为π

    def forward(self, coor: Tensor) -> Tensor:
        """
        :param coor: (B, 3, N) 输入的坐标张量
        :return: (B, emb_dim, N) 输出的编码后的张量
        """
        coor = coor.transpose(1, 2)  # 转换维度顺序 (B, N, 3)
        assert coor.shape[-1] == self.in_channels  # 确保输入的最后一个维度与预期一致

        # 计算不同向量位置的编码周期 (0~temperature)
        dim_t = torch.arange(self.num_pos_feats, dtype=coor.dtype, device=coor.device)  # 生成位置编码的频率序列
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)  # 计算每个频率的编码

        coor = coor * self.scale  # 对输入坐标进行缩放
        pos_divided = coor.unsqueeze(-1) / dim_t  # 将坐标除以编码周期，生成正弦和余弦编码
        pos_sin = pos_divided[..., 0::2].sin()  # 计算sin编码
        pos_cos = pos_divided[..., 1::2].cos()  # 计算cos编码
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*coor.shape[:-1], -1)  # 将sin和cos编码堆叠在一起，形成最终的编码

        # 用零填充未使用的维度
        pos_emb = F.pad(pos_emb, (0, self.padding))  # 在末尾添加必要的填充
        pos_emb = pos_emb.transpose(1, 2)  # 转换回原始形状 (B, emb_dim, N)
        return pos_emb  # 返回生成的位置编码

