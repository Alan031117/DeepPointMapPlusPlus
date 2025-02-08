import torch  # 导入PyTorch库，用于构建和训练神经网络
import torch.nn as nn  # 从torch中导入神经网络模块，用于定义神经网络的层
from torch.nn import functional as F  # 导入torch.nn.functional模块，提供常用的函数如激活函数、损失函数等
from torch import Tensor as Tensor  # 从torch中导入Tensor类型，并为其起一个别名为Tensor
from typing import Tuple  # 从typing模块中导入Tuple，用于类型注解，表示返回值的类型

import colorlog as logging  # 导入colorlog模块，并以logging为别名，用于生成带有颜色的日志输出
logger = logging.getLogger(__name__)  # 创建一个日志记录器实例，名称为当前模块的名称
logger.setLevel(logging.INFO)  # 设置日志记录器的日志级别为INFO，表示只记录INFO级别及以上的日志信息


class RegistrationLoss(nn.Module):  # 定义一个继承自nn.Module的自定义损失类RegistrationLoss，用于计算配准任务中的总损失
    """
    配准总损失
    L = lambda1 * L_p + lambda2 * L_c + lambda3 * L_o
    """

    def __init__(self, args):  # 初始化方法，接收一个包含参数的args对象
        super().__init__()  # 调用父类nn.Module的初始化方法
        self.args = args  # 保存传入的参数对象args到实例变量self.args中
        self.loss_cfg = self.args.loss  # 提取并保存args中的损失配置到self.loss_cfg中

        # 从损失配置中提取各种超参数并保存到实例变量中
        self.tau = self.loss_cfg.tau  # 配对损失的温度系数
        self.offset_value = self.loss_cfg.offset_value  # 偏移损失的计算方式（如曼哈顿距离、欧式距离或马氏距离）
        self.eps_positive = self.loss_cfg.eps_positive  # 正点与目标点的距离阈值
        self.eps_offset = self.loss_cfg.eps_offset  # 偏移量的距离阈值
        self.lambda_p = self.loss_cfg.lambda_p  # 配对损失的权重系数
        self.lambda_c = self.loss_cfg.lambda_c  # 粗粒度配对损失的权重系数
        self.lambda_o = self.loss_cfg.lambda_o  # 偏移损失的权重系数

    def forward(self, src_global_coor: Tensor, dst_global_coor: Tensor,
                src_padding_mask: Tensor, dst_padding_mask: Tensor,
                src_pairing_fea: Tensor, dst_pairing_fea: Tensor,
                src_coarse_pairing_fea: Tensor, dst_coarse_pairing_fea: Tensor,
                src_offset_res: Tensor, dst_offset_res: Tensor,
                label_loss: Tensor):
        """
        定义前向传播方法，用于计算损失和配对精度
        :param src_global_coor: (B, 3, S) 源点云全局坐标
        :param dst_global_coor: (B, 3, D) 目标点云全局坐标
        :param src_padding_mask: (B, S) 源点云填充标记，True为填充点，False为正常点
        :param dst_padding_mask: (B, D) 目标点云填充标记，True为填充点，False为正常点
        :param src_pairing_fea: (B, C, S) 源点云相似度特征
        :param dst_pairing_fea: (B, C, D) 目标点云相似度特征
        :param src_coarse_pairing_fea: (B, C', S) 源点云粗粒度相似度判别特征
        :param dst_coarse_pairing_fea: (B, C', D) 目标点云粗粒度相似度判别特征
        :param src_offset_res: (K, 3, 1) 源点云预测相对目标点云中对应点坐标偏移量与gt的差值
        :param dst_offset_res: (K, 3, 1) 目标点云预测源点云中对应点坐标偏移量与gt的差值
        :return: loss, metrics
        """
        # 取反填充标记，以便于后续操作（False变True表示非填充点）
        src_padding_mask, dst_padding_mask = ~src_padding_mask, ~dst_padding_mask
        # 将全局坐标和特征的通道维度调整到最后一个维度以便于后续计算
        src_global_coor, dst_global_coor = src_global_coor.transpose(1, 2), dst_global_coor.transpose(1, 2)
        src_pairing_fea, dst_pairing_fea = src_pairing_fea.transpose(1, 2), dst_pairing_fea.transpose(1, 2)
        src_coarse_pairing_fea, dst_coarse_pairing_fea = \
            src_coarse_pairing_fea.transpose(1, 2), dst_coarse_pairing_fea.transpose(1, 2)
        src_offset_res, dst_offset_res = src_offset_res.transpose(1, 2), dst_offset_res.transpose(1, 2)

        '''配准方向src ==> dst'''
        # 调用 make_pairs 方法，为src点云生成与dst点云的配对信息
        corr_ids_src, corr_mask_src, neutral_mask_src = \
            self.make_pairs(src_global_coor, dst_global_coor, self.eps_positive)
        # 计算 src 到dst方向的配对损失
        loss_pairing_src = self.pairing_loss(
            src_pairing_fea, dst_pairing_fea, src_padding_mask, corr_ids_src, corr_mask_src,
            neutral_mask=torch.zeros_like(neutral_mask_src, dtype=torch.bool, device=neutral_mask_src.device)
        )
        # 计算src到dst方向的粗粒度配对损失
        loss_coarse_pairing_src = self.pairing_loss(
            src_coarse_pairing_fea, dst_coarse_pairing_fea, src_padding_mask, corr_ids_src, corr_mask_src,
            neutral_mask=neutral_mask_src
        )
        # 计算src到dst方向的偏移损失
        loss_offset_src = self.offset_loss(src_offset_res)
        # 计算src到dst方向的配对准确率
        top1_pairing_acc_src = self.eval_pairing_acc(
            src_pairing_fea, dst_pairing_fea, src_padding_mask, corr_ids_src, corr_mask_src)

        '''配准方向dst ==> src'''
        # 调用make_pairs方法，为dst点云生成与src点云的配对信息
        corr_ids_dst, corr_mask_dst, neutral_mask_dst = \
            self.make_pairs(dst_global_coor, src_global_coor, self.eps_positive)
        # 计算dst到src方向的配对损失
        loss_pairing_dst = self.pairing_loss(
            dst_pairing_fea, src_pairing_fea, dst_padding_mask, corr_ids_dst, corr_mask_dst,
            neutral_mask=torch.zeros_like(neutral_mask_dst, dtype=torch.bool, device=neutral_mask_dst.device)
        )
        # 计算dst到src方向的粗粒度配对损失
        loss_coarse_pairing_dst = self.pairing_loss(
            dst_coarse_pairing_fea, src_coarse_pairing_fea, dst_padding_mask, corr_ids_dst, corr_mask_dst,
            neutral_mask=neutral_mask_dst
        )
        # 计算dst到src方向的偏移损失
        loss_offset_dst = self.offset_loss(dst_offset_res)

        # 计算dst到src方向的配对准确率
        top1_pairing_acc_dst = self.eval_pairing_acc(
            dst_pairing_fea, src_pairing_fea, dst_padding_mask, corr_ids_dst, corr_mask_dst)

        '''最终损失和指标为双向的均值'''
        # 计算配对损失、粗粒度配对损失和偏移损失的双向平均值
        loss_pairing = (loss_pairing_src + loss_pairing_dst) / 2
        loss_coarse_pairing = (loss_coarse_pairing_src + loss_coarse_pairing_dst) / 2
        loss_offset = (loss_offset_src + loss_offset_dst) / 2

        # 计算配对准确率的双向平均值
        top1_pairing_acc = (top1_pairing_acc_src + top1_pairing_acc_dst) / 2

        # 计算总损失，使用lambda权重系数
        loss = self.lambda_p * loss_pairing + self.lambda_c * loss_coarse_pairing + self.lambda_o * loss_offset + label_loss * 1

        return loss, top1_pairing_acc, loss_pairing, loss_coarse_pairing, loss_offset  # 返回总损失和各个子损失以及配对准确率

    @staticmethod  # 静态方法装饰器，表示这个方法不需要访问类实例的状态
    def make_pairs(src_global_coor: Tensor, dst_global_coor: Tensor, dis_threshold: float) -> Tuple[
        Tensor, Tensor, Tensor]:
        # 这个方法用于在特征描述符级别为点云中的每个点找到对应的配对点，包括正点、中立点和负点。
        """
        descriptor级别配对
        为源点云的每一个点，找到目标点云中的正点、中立点和负点
        正点：全局坐标下距离最近且小于距离阈值的点
        中立点：小于距离阈值的非对应点 （距离足够近但又不是最近的）
        负点：大于距离阈值的点

        :param src_global_coor: (B, S, 3) 源点云全局坐标
        :param dst_global_coor: (B, D, 3) 目标点云全局坐标
        :param dis_threshold: 区分负点的距离阈值
        :return: (B, S): 对应点索引; (B, S) 对应点掩码; (B, S, D) 中立点掩码矩阵
        """
        # 获取源点云的批量大小B、源点云的点数S，以及计算设备device
        B, S, device = src_global_coor.shape[0], src_global_coor.shape[1], src_global_coor.device

        # 计算源点云与目标点云之间的平方欧氏距离 (B, S, D)，B是批量大小，S是源点云中的点数，D是目标点云中的点数
        dis = torch.sum(torch.square(src_global_coor.unsqueeze(2) - dst_global_coor.unsqueeze(1)), dim=-1)

        # 找到每个源点与目标点云中距离最近的点，并获取这些距离和对应的索引
        min_dis, min_dis_corr_ids = torch.min(dis, dim=-1)  # (B, S)

        # 创建批量索引和行索引，用于构建中立点掩码矩阵
        batch_idx = torch.arange(0, B, device=device).unsqueeze(1).repeat(1, S)
        row_idx = torch.arange(0, S, device=device).unsqueeze(0).repeat(B, 1)
        # 创建中立点掩码矩阵，标记那些距离小于指定阈值且不是最近点的点
        neutral_mask = dis <= (dis_threshold * dis_threshold)  # (B, S, D)
        # 确保最近点不会被标记为中立点
        neutral_mask[batch_idx, row_idx, min_dis_corr_ids] = False

        # 创建对应点掩码，标记那些距离小于阈值的最近点
        corr_mask = min_dis <= (dis_threshold * dis_threshold)  # (B, S)
        corr_ids = min_dis_corr_ids  # 获取最近点的索引
        # 对于没有对应点的点，将其对应索引标记为-1
        corr_ids[~corr_mask] = -1

        # 返回对应点索引、对应点掩码和中立点掩码矩阵
        return corr_ids, corr_mask, neutral_mask

    def pairing_loss(self, src_pairing_fea: Tensor, dst_pairing_fea: Tensor, src_padding_mask: Tensor,
                     corr_ids: Tensor, corr_mask: Tensor, neutral_mask: Tensor) -> Tensor:
        # 这个方法用于计算配对损失（pairing loss），即基于相似度特征计算源点云和目标点云之间的匹配情况
        B, S, D = src_pairing_fea.shape[0], src_pairing_fea.shape[1], dst_pairing_fea.shape[1]

        # 将源和目标点云的配对特征进行归一化处理，使每个特征向量的L2范数为1
        samples_src = F.normalize(src_pairing_fea, p=2, dim=-1)
        samples_dst = F.normalize(dst_pairing_fea, p=2, dim=-1)

        # 计算归一化后的源点云和目标点云之间的余弦相似度矩阵 (B, S, D)
        logits = samples_src @ samples_dst.transpose(1, 2)

        # 将相似度矩阵logits展平为二维矩阵 (B*S, D)，并去除padding点（即无效点）
        src_padding_mask_1d = src_padding_mask.reshape(-1)
        logits = logits.reshape(B * S, D)[src_padding_mask_1d]
        labels = corr_ids.reshape(-1)[src_padding_mask_1d]
        neutral_mask = neutral_mask.reshape(B * S, D)[src_padding_mask_1d]
        corr_mask = corr_mask.reshape(-1)[src_padding_mask_1d]

        # 过滤掉中立点（即距离足够近但不是最近的点），并确保仅计算那些存在配对关系的点的损失
        logits[neutral_mask] = -1e8  # 对于中立点，将其相似度设为非常小的值，避免其对损失产生影响
        logits_pos = logits[corr_mask]  # 提取存在配对关系的点的相似度
        labels = labels[corr_mask]  # 提取对应的标签（配对索引）

        # 计算配对相似度的InfoNCE损失，若存在有效的配对点
        if logits_pos.shape[0] > 0:
            logits_pos = logits_pos / self.tau  # 使用温度系数进行缩放
            logprobs_pos = F.log_softmax(logits_pos, dim=-1)  # 计算log(softmax)以避免数值溢出
            loss_pos = -logprobs_pos.gather(dim=-1, index=labels.unsqueeze(1)).squeeze(1)  # 获取正确配对的损失
            loss_pos = loss_pos.mean()  # 计算平均损失
        else:
            loss_pos = 0  # 如果没有有效的配对点，损失设为0

        loss_pairing = loss_pos  # 最终的配对损失为正样本损失
        return loss_pairing  # 返回计算出的配对损失

    def offset_loss(self, src_offset_res: Tensor) -> Tensor:
        # 这个方法用于计算偏移量损失，即预测的点偏移量与真实偏移量之间的误差
        src_offset_res = src_offset_res.squeeze(1)  # (K, 1, 3) -> (K, 3)，去除维度为1的维度

        # 根据配置选择不同的距离度量方式来计算偏移量误差
        if self.offset_value == 'manhattan':  # 曼哈顿距离
            offset_err = torch.sum(torch.abs(src_offset_res), dim=-1)  # 计算曼哈顿距离误差
        elif self.offset_value == 'euclidean':  # 欧式距离
            offset_err = torch.norm(src_offset_res, p=2, dim=-1)  # 计算欧式距离误差
        elif self.offset_value == 'mahalanobis':  # 马氏距离
            try:
                cov_inv = torch.linalg.inv(torch.cov(src_offset_res.detach().T))  # 计算协方差矩阵的逆矩阵
            except:
                # 如果协方差矩阵不可逆，发出警告并使用欧式距离替代马氏距离
                logger.warning('The cov of current data is invertible. '
                               '\'mahalanobis\' will not be calculated. Using l2 instead')
                cov_inv = torch.eye(3, device=src_offset_res.device, dtype=src_offset_res.dtype)  # 使用单位矩阵作为替代

            # 使用einsum公式计算马氏距离
            offset_err = torch.sqrt(torch.einsum('nj,jk,nk->n', src_offset_res, cov_inv, src_offset_res))
        else:
            raise ValueError  # 如果提供了未知的距离度量方式，抛出错误

        # 计算平均偏移量误差并返回
        loss_offset = torch.sum(offset_err, dim=-1) / max(offset_err.shape[0], 1.0)
        return loss_offset  # 返回偏移量损失

    @staticmethod
    def eval_pairing_acc(src_pairing_fea: Tensor, dst_pairing_fea: Tensor,
                         src_padding_mask: Tensor, corr_ids_src: Tensor, corr_mask_src: Tensor) -> float:
        # 这个方法用于评估配对准确率，即计算预测的配对点与真实配对点之间的一致性
        # 归一化源点云和目标点云的特征向量
        samples_src = F.normalize(src_pairing_fea, p=2, dim=-1)
        samples_dst = F.normalize(dst_pairing_fea, p=2, dim=-1)
        # 计算相似度矩阵 (B, S, D)
        similarity_matrix = samples_src @ samples_dst.transpose(1, 2)
        # 获取每个源点与目标点云中相似度最高的点的索引
        _, src_corr_ids_pred = torch.max(similarity_matrix, dim=2)

        # 忽略padding点
        src_padding_mask = src_padding_mask.reshape(-1)
        src_corr_ids_pred = src_corr_ids_pred.reshape(-1)[src_padding_mask]
        src_corr_ids_gt = corr_ids_src.reshape(-1)[src_padding_mask]
        corr_mask_src = corr_mask_src.reshape(-1)[src_padding_mask]

        # 比较预测的配对点与真实配对点，计算配对准确率
        correspondence_matrix = (src_corr_ids_pred == src_corr_ids_gt)[corr_mask_src]

        # 计算Top-1准确率并返回
        top1_acc = torch.sum(correspondence_matrix) / max(correspondence_matrix.shape[0], 1.0)
        return top1_acc.item()  # 返回准确率的值

