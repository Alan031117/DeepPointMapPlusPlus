import colorlog as logging  # 导入colorlog模块，并将其命名为logging，用于彩色日志记录。
logging.basicConfig(level=logging.INFO)  # 设置日志记录的基本配置，日志级别为INFO。
logger = logging.getLogger(__name__)  # 获取一个名为当前模块名的logger对象，用于日志记录。
logger.setLevel(logging.INFO)  # 设置logger对象的日志级别为INFO。

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from network.decoder.descriptor_attention import PositionEmbeddingCoordsSine


# 标签解码器模块
class Label_Decoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2, enable=True, replace_mask=True, use_HNM=True):
        super(Label_Decoder, self).__init__()
        self.enable = enable
        self.replace_mask = replace_mask
        self.use_HNM = use_HNM
        self.num_classes = output_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(True),
        )

        self.pos_embedding_layer = PositionEmbeddingCoordsSine(3, hidden_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )

    def forward(self, accurate_labels, points_coor, points_fea, mask):
        if not self.enable:
            return 0, 0, mask

        B, _, N = points_coor.shape
        points_fea = points_fea.transpose(1, 2)  # (B, C, N) -> (B, N, C)
        pos_embedding = self.pos_embedding_layer(points_coor).transpose(1, 2)

        # 特征提取
        points_fea = self.input_proj(points_fea) + pos_embedding
        mid_fea, _ = self.self_attn(points_fea, points_fea, points_fea, key_padding_mask=mask)  # attention
        points_fea = self.norm1(points_fea + mid_fea)  # residual & norm
        points_fea = self.norm2(self.mlp(points_fea) + points_fea)  # mlp & residual & norm

        # 语义分割
        logits = self.head(points_fea)  # (B, N, 2)

        if self.use_HNM:
            # 计算交叉熵损失，只考虑有效点，使用难分样本挖掘(HNM)，对于背景点仅考虑少量困难负样本
            valid_mask = (accurate_labels >= 0) & ~mask
            valid_pred = logits[valid_mask]
            valid_gt = accurate_labels[valid_mask]
            static_pred = valid_pred[valid_gt == 0]
            dynamic_pred = valid_pred[valid_gt == 1]
            static_pred_dynamic_ratio = static_pred.softmax(-1)[:, 1]  # 对静态点预测的动态得分
            hard_negative_value, hard_negative_indices = \
                torch.topk(static_pred_dynamic_ratio, k=min(dynamic_pred.shape[0] * 3, static_pred.shape[0]))
            hard_negative_static_pred = static_pred[hard_negative_indices]
            logits_selected = torch.cat([dynamic_pred, hard_negative_static_pred], dim=0)
            labels_selected = accurate_labels.new_zeros((logits_selected.shape[0],))
            labels_selected[:dynamic_pred.shape[0]] = 1
            labels_selected = F.one_hot(labels_selected, num_classes=self.num_classes).float()
            loss = F.binary_cross_entropy_with_logits(logits_selected, labels_selected, reduction='mean')
        else:
            # 计算交叉熵损失，只考虑有效点
            valid_mask = (accurate_labels >= 0) & ~mask
            accurate_labels[~valid_mask] = 0
            label_gt = F.one_hot(accurate_labels, num_classes=self.num_classes).float()
            sample_weight = valid_mask.float() / valid_mask.sum(1, keepdim=True).clamp(min=1)
            batch_weight = valid_mask.any(1) / valid_mask.any(1).sum().clamp(min=1)
            loss = F.binary_cross_entropy_with_logits(logits, label_gt, reduction='none')
            loss = (loss * sample_weight.unsqueeze(-1)).sum(-1).sum(-1)
            loss = (loss * batch_weight).sum()

        # 计算准确率
        pred = torch.argmax(logits, dim=-1)  # (B, N)
        accuracy = (pred[valid_mask] == accurate_labels[valid_mask]).float().sum() / valid_mask.sum().clamp(min=1)

        # 更新掩码，将动态点与原始掩码结合
        dynamic_points_mask = (pred == 1)  # 假设类别1为动态点
        too_many_dynamic_mask = dynamic_points_mask.sum(1) > (0.5 * N)
        if too_many_dynamic_mask.any():
            # 限制动态点数量，不得超过总点数的一半
            logger.warning('Too many dynamic points (predicted by model), clipping...')
            num_static = int(0.5 * N)
            static_score, static_index = (
                torch.topk(logits[too_many_dynamic_mask, :, 0], k=num_static, largest=True, dim=-1))
            dynamic_points_mask[torch.where(too_many_dynamic_mask)[0].unsqueeze(-1).repeat(1, num_static), static_index] = False

        if self.replace_mask:
            updated_mask = mask | dynamic_points_mask  # 与原始掩码取并集
        else:
            updated_mask = mask

        return loss, accuracy, updated_mask

    # 标签推理函数，用于推断点的类别
    def label_inference(self, points_coor, points_fea, mask):
        if not self.enable:
            return mask

        B, _, N = points_coor.shape
        points_fea = points_fea.transpose(1, 2)  # (B, C, N) -> (B, N, C)
        pos_embedding = self.pos_embedding_layer(points_coor).transpose(1, 2)

        # 特征提取
        points_fea = self.input_proj(points_fea) + pos_embedding
        mid_fea, _ = self.self_attn(points_fea, points_fea, points_fea, key_padding_mask=mask)  # attention
        points_fea = self.norm1(points_fea + mid_fea)  # residual & norm
        points_fea = self.norm2(self.mlp(points_fea) + points_fea)  # mlp & residual & norm

        # 语义分割
        logits = self.head(points_fea)  # (B, N, 2)

        # 获取预测结果
        thr = 0.9
        pred = (F.sigmoid(logits)[..., 1] >= thr).long()

        # 更新掩码，将动态点与原始掩码结合
        dynamic_points_mask = (pred == 1)  # 类别1为动态点
        too_many_dynamic_mask = dynamic_points_mask.sum(1) > (0.5 * N)
        if too_many_dynamic_mask.any():
            # 限制动态点数量，不得超过总点数的一半
            logger.warning('Too many dynamic points (predicted by model), clipping...')
            num_static = int(0.5 * N)
            static_score, static_index = (
                torch.topk(logits[too_many_dynamic_mask, :, 0], k=num_static, largest=True, dim=-1))
            dynamic_points_mask[torch.where(too_many_dynamic_mask)[0].unsqueeze(-1).repeat(1, num_static), static_index] = False

        # 根据预测结果更新mask
        if self.replace_mask:
            mask = mask | dynamic_points_mask  # 与原始掩码取并集

        return mask


# class Label_Decoder(nn.Module):
#     def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
#         super(Label_Decoder, self).__init__()
#
#         # 改进的特征提取模块：包含卷积、批标准化、ReLU和多头注意力机制
#         self.feature_extractor = nn.Sequential(
#             nn.Conv1d(input_dim, hidden_dim, kernel_size=1, padding=1),  # 卷积层，提取特征
#             nn.BatchNorm1d(hidden_dim),  # 批标准化，稳定训练
#             nn.ReLU(),  # 激活函数
#             nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, padding=1),  # 第二个卷积层
#             nn.BatchNorm1d(hidden_dim),  # 批标准化
#             nn.ReLU(),  # 激活函数
#             nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),  # 1x1卷积，用于特征融合
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU()
#         )
#
#         # 多头注意力模块，用于捕捉全局特征
#         self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
#
#         # 分类头，包含卷积层
#         self.fc1 = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
#
#     def forward(self, accurate_labels, points_fea, mask):
#         # 特征提取
#         x = self.feature_extractor(points_fea)
#         # 多头注意力，获取全局上下文信息
#         x = x.permute(0, 2, 1)  # 转换为 [batch_size, num_points, hidden_dim]
#         x, _ = self.multihead_attn(x, x, x)
#         x = x.permute(0, 2, 1)  # 转回 [batch_size, hidden_dim, num_points]
#
#         # 分类层，得到分类结果
#         logits = self.fc1(x)
#
#         # 确保标签是Tensor并在正确的设备上
#         if isinstance(accurate_labels, np.ndarray):
#             accurate_labels = torch.tensor(accurate_labels, dtype=torch.long, device=points_fea.device)
#         else:
#             accurate_labels = accurate_labels.to(torch.long).to(points_fea.device)
#
#         # 标签映射，将准确标签映射到静态或动态对象
#         static_objects = torch.tensor([40, 44, 48, 49, 50, 51, 52, 60, 70, 71, 72, 80, 81, 99],
#                                       dtype=torch.long, device=points_fea.device)
#         dynamic_objects = torch.tensor(
#             [0, 1, 10, 11, 13, 15, 16, 18, 20, 30, 31, 32, 252, 253, 254, 255, 256, 257, 258, 259],
#             dtype=torch.long, device=points_fea.device)
#
#         # 创建标签映射数组，将静态对象映射为0，动态对象映射为1
#         max_label_value = max(max(static_objects), max(dynamic_objects))
#         mapping_array = torch.full((max_label_value + 1,), -1, dtype=torch.long, device=points_fea.device)
#         mapping_array[static_objects] = 0
#         mapping_array[dynamic_objects] = 1
#
#         # 映射准确标签
#         mapped_labels = mapping_array[accurate_labels.view(-1)].view_as(accurate_labels)
#         valid_mask = (mapped_labels >= 0)
#
#         # 计算交叉熵损失，只考虑有效点
#         logits_reshaped = logits.permute(0, 2, 1).contiguous()  # logits形状调整为 [batch_size, num_points, num_classes]
#         mapped_labels = mapped_labels.view(-1)  # 调整标签形状为 [batch_size * num_points]
#         valid_mask = valid_mask.view(-1)  # 调整valid_mask的形状为 [batch_size * num_points]
#
#         ce_loss = nn.functional.cross_entropy(logits_reshaped.view(-1, logits_reshaped.size(-1)),
#                                               mapped_labels, reduction='none')
#         ce_loss = ce_loss[valid_mask].mean()
#
#         # 计算准确率
#         pred = torch.argmax(logits, dim=1)
#         accuracy = (pred[valid_mask.view_as(pred)] == mapped_labels[valid_mask]).float().mean()
#
#         # 更新掩码，将动态点与原始掩码结合
#         dynamic_points_mask = (pred == 1)  # 假设类别1为动态点
#         updated_mask = mask | dynamic_points_mask  # 与原始掩码取并集
#
#         # 梯度裁剪，防止梯度爆炸
#         torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
#
#         return ce_loss, accuracy, updated_mask
#
#     # 标签推理函数，用于推断点的类别
#     def label_inference(self, points_fea, mask):
#         self.eval()
#         with torch.no_grad():
#             # 提取特征
#             x = self.feature_extractor(points_fea)
#             # 多头注意力，获取全局上下文信息
#             x = x.permute(0, 2, 1)  # 转换为 [batch_size, num_points, hidden_dim]
#             x, _ = self.multihead_attn(x, x, x)
#             x = x.permute(0, 2, 1)  # 转回 [batch_size, hidden_dim, num_points]
#
#             # 分类层，得到分类结果
#             logits = self.fc1(x)
#             # 获取预测结果
#             predictions_mapped = torch.argmax(logits, dim=1)
#             # 根据预测结果更新mask
#             static_mask = predictions_mapped == 0
#             mask = mask.to(points_fea.device)
#             mask[static_mask] = False
#             mask[~static_mask] = True
#
#         return mask

