import colorlog as logging  # 导入colorlog模块，并将其命名为logging，用于彩色日志记录。
logging.basicConfig(level=logging.INFO)  # 设置日志记录的基本配置，日志级别为INFO。
logger = logging.getLogger(__name__)  # 获取一个名为当前模块名的logger对象，用于日志记录。
logger.setLevel(logging.INFO)  # 设置logger对象的日志级别为INFO。
import time

import numpy as np
import open3d as o3d
from utils.visualization import show_pcd
import random  # 导入random模块，用于生成随机数。
import pickle  # 导入pickle模块，用于对象的序列化和反序列化。
import numpy as np  # 导入numpy模块，并将其命名为np，用于数值计算。
import numpy.linalg as linalg  # 导入numpy的线性代数模块，并命名为linalg，用于矩阵运算。
import torch  # 导入PyTorch库，用于深度学习任务。
import torch.nn as nn  # 从PyTorch中导入神经网络模块，并命名为nn，用于构建神经网络模型。
import torch.nn.functional as F  # 导入PyTorch的函数式接口模块，命名为F，用于调用各种神经网络功能。
from torch import Tensor as Tensor  # 从PyTorch中导入Tensor类型，并保持名称不变，代表张量数据结构。
from typing import Tuple, Dict  # 导入Tuple和Dict类型提示，用于函数参数和返回值的类型注释。
from utils.pose import rt_global_to_relative  # 从utils.pose模块中导入rt_global_to_relative函数，用于转换全局姿态到相对姿态。



class DeepPointModelPipeline(nn.Module):  # 定义一个名为DeepPointModelPipeline的类，继承自PyTorch的nn.Module类，用于封装DeepPoint网络模型的训练流程。
    """
    DeepPoint网络模型的训练流程封装
    """

    def __init__(self, args, encoder: nn.Module, decoder: nn.Module, criterion: nn.Module, label_decoder:nn.Module):
        super().__init__()  # 调用父类nn.Module的构造函数，初始化基础组件。
        self.args = args  # 保存传入的参数args，这通常包含训练和模型的超参数。
        self.encoder = encoder  # 保存传入的编码器模型，用于特征提取。
        self.decoder = decoder  # 保存传入的解码器模型，用于特征配对和配准。
        self.label_decoder = label_decoder
        self.criterion = criterion  # 保存传入的损失函数，用于计算训练过程中的损失。
        self._forward_method = None  # 初始化_forward_method为空，稍后会用于设置模型的前向传播方法。
        self.registration()  # 调用registration方法，可能用于设置模型的配准方法或初始化。
        self.refined_SE3_cache = dict()  # 初始化一个字典，用于缓存精化后的SE3变换矩阵。

    def forward(self, *args) -> Tuple[Tensor, dict]:
        return self._forward_method(*args)  # 调用之前设置的_forward_method方法进行前向传播，并返回结果。

    """ 点云配准部分 """
    def _train_registration(self, pcd: Tensor, R: Tensor, T: Tensor, padding_mask: Tensor, calib: Tensor, info: dict) \
            -> Tuple[Tensor, dict]:  # 定义方法_train_registration，用于点云配准训练
        """
        :param pcd: (S, 4, N)  # 点云数据 (32, 4, 16384)，S表示帧数，（ 前三个表示 xyz ，最后一个表示 label ），N表示点的数量。
        :param R: (S, 3, 3)  # 旋转矩阵 (32, 3, 3)。
        :param T: (S, 3, 1)  # 平移向量 (32, 3, 1)。
        :param calib: (S, 4, 4)  # 校准矩阵 (32, 4, 4)，表示相对于原始姿态的变换SE3。
        :param padding_mask: (S, 16384)  # 填充掩码 (S, 16384)，用于指示点云的有效区域。
        :param info: 数据信息 {dict:3}
            —— dsf_index: 帧索引 (list:32)
            —— refined_SE3_file: 真实 SE3 变换文件路径(list:16)
            —— num_map: 局部地图数量 (list:16)
        """

        coor, fea, mask, index = self.encoder(pcd, padding_mask)  # 使用 encoder 提取特征点(坐标、特征、掩码、索引)
        #   coor: (S, 3, 256)
        #   fea: (S, 128, 256)
        #   mask: (S, 256)
        #   index: (S, 256)


        # 获取准确标签 (S, 256)
        # accurate_labels = np.array([pcd[frame_idx, 3, index[frame_idx]].cpu().numpy() for frame_idx in range(pcd.shape[0])])
        accurate_labels = pcd[torch.arange(pcd.shape[0], device=pcd.device).unsqueeze(-1).repeat(1, index.shape[1]), 3, index]
        accurate_labels = accurate_labels.long()

        # 语义分割标签训练
        label_loss, accuracy, mask = (
            self.label_decoder(accurate_labels=accurate_labels, points_coor=coor, points_fea=fea, mask=mask))

        S, _, N = coor.shape    # 一共 S 帧，每帧 N 个点
        B = info['num_map']     # 获取num_map的值，B表示选取的局部地图数量
        S = S // B      # S 变为一张局部地图中的帧数
        coor = coor * self.args.slam_system.coor_scale  # 对坐标进行缩放，以匹配SLAM系统的尺度，尺度系数源于YAML
        pcd_index = np.asarray([dsf_index[2] for dsf_index in info['dsf_index']])
        refined_SE3_file = info['refined_SE3_file']  # 获取真实的SE3变换文件路径

        # 拆分成batch形式
        fea = fea.reshape(B, S, -1, N)  # 将特征重新reshape为batch形式。(32, 128, 256) --> (16, 2, 128, 256)
        coor = coor.reshape(B, S, -1, N)  # 将坐标重新reshape为batch形式。(32, 3, 256) --> (16, 2, 3, 256)
        R = R.reshape(B, S, 3, 3)  # 将旋转矩阵重新reshape为batch形式。(32, 3, 3) --> (16, 2, 3, 3)
        T = T.reshape(B, S, 3, 1)  # 将平移向量重新reshape为batch形式。(32, 3, 1) --> (16, 2, 3, 1)
        mask = mask.reshape(B, S, -1)  # 将掩码重新reshape为batch形式。(32, 256) --> (16, 2, 256)
        pcd_index = pcd_index.reshape((B, S))  # 将点云索引重新reshape为batch形式。(32) --> (16, 2)
        calib = calib.reshape(B, S, 4, 4)  # 将校准矩阵重新reshape为batch形式。(32, 4, 4) --> (16, 2, 4, 4)

        map_size_max = self.args.train.registration.map_size_max  # 从训练参数中获取最大局部地图尺寸
        if S <= map_size_max:  # 如果帧数不超过最大局部地图尺寸
            if random.random() < 0.5:  # 50%的概率拆分为[1帧，剩余帧]
                S1 = 1
            else:
                S1 = random.randint(1, S - 1)  # 50%的概率随机拆分帧数，确保地图帧数不超过最大局部地图尺寸
        else:
            S1 = random.randint(S - map_size_max, map_size_max)  # 如果 S 超过最大局部地图尺寸，确保 S1,S2 不超过
        S2 = S - S1  # 计算剩余帧数

        # 场景内所有帧被拆分在S1和S2两张局部地图中
        src_coor, dst_coor = coor[:, :S1], coor[:, S1:]  # (16, 2, 3, 256) --> (16, 1, 3, 256)
        src_fea, dst_fea = fea[:, :S1], fea[:, S1:]  # (16, 2, 128, 256) --> (16, 1, 128, 256)
        src_R, dst_R = R[:, :S1], R[:, S1:]  # (16, 2, 3, 3) --> (16, 1, 3, 3)
        src_T, dst_T = T[:, :S1], T[:, S1:]  # (16, 2, 3, 1) --> (16, 1, 3, 1)
        src_mask, dst_mask = mask[:, :S1], mask[:, S1:]  # (16, 2, 256) --> (16, 1, 256)
        src_index, dst_index = pcd_index[:, :S1], pcd_index[:, S1:]  # (16, 2) --> (16, 1)
        src_calib, dst_calib = calib[:, :S1], calib[:, S1:]  # (16, 2, 4, 4) --> (16, 1, 4, 4)

        """
            计算S1和S2的第一对 对应帧的真实旋转矩阵和真实平移矩阵
            :param R = {Tensor:(16, 3, 3)}  真实旋转矩阵
            :param T = {Tensor:(16, 3, 1)}  真实平移矩阵
        """
        R, T = self._get_accurate_RT(src_index=src_index[:, 0], dst_index=dst_index[:, 0],
                                     src_R=src_R[:, 0], src_T=src_T[:, 0], src_calib=src_calib[:, 0],
                                     dst_R=dst_R[:, 0], dst_T=dst_T[:, 0], dst_calib=dst_calib[:, 0],
                                     refined_SE3_file=refined_SE3_file)

        if S1 > 1:  # 如果S1中有不止一帧
            map1_relative_R, map1_relative_T = \
                self._get_accurate_RT(src_index=src_index[:, 1:], dst_index=src_index[:, 0],
                                      src_R=src_R[:, 1:], src_T=src_T[:, 1:], src_calib=src_calib[:, 1:],
                                      dst_R=src_R[:, 0], dst_T=src_T[:, 0], dst_calib=src_calib[:, 0],
                                      refined_SE3_file=refined_SE3_file)  # 计算S1中其他帧与第一帧的真实旋转矩阵和真实平移矩阵

            src_coor[:, 1:] = map1_relative_R @ src_coor[:, 1:] + map1_relative_T  # 通过旋转和平移，将其他帧对齐到第一帧

        if S2 > 1:  # 如果S2中有不止一帧
            map2_relative_R, map2_relative_T = \
                self._get_accurate_RT(src_index=dst_index[:, 1:], dst_index=dst_index[:, 0],
                                      src_R=dst_R[:, 1:], src_T=dst_T[:, 1:], src_calib=dst_calib[:, 1:],
                                      dst_R=dst_R[:, 0], dst_T=dst_T[:, 0], dst_calib=dst_calib[:, 0],
                                      refined_SE3_file=refined_SE3_file,
                                      bridge_index=src_index[:, 0])  # 计算S2中其他帧与第一帧的真实旋转矩阵和真实平移矩阵

            dst_coor[:, 1:] = map2_relative_R @ dst_coor[:, 1:] + map2_relative_T  # 通过旋转和平移，将其他帧对齐到第一帧

        src_coor = src_coor.transpose(1, 2).reshape(B, -1, S1 * N)  # 改变形状 (16, 1, 3, 256) --> (16, 3, 256)
        src_fea = src_fea.transpose(1, 2).reshape(B, -1, S1 * N)  # 改变形状 (16, 1, 128, 256) --> (16, 128, 256)
        src_mask = src_mask.reshape(B, -1)  # 改变形状 (16, 1, 256) --> (16, 256)
        dst_coor = dst_coor.transpose(1, 2).reshape(B, -1, S2 * N)  # 改变形状 (16, 1, 3, 256) --> (16, 3, 256)
        dst_fea = dst_fea.transpose(1, 2).reshape(B, -1, S2 * N)  # 改变形状 (16, 1, 128, 256) --> (16, 128, 256)
        dst_mask = dst_mask.reshape(B, -1)  # 改变形状 (16, 1, 256) --> (16, 256)

        src_global_coor = R @ src_coor + T  # 将S1的坐标转换到全局坐标系中。
        dst_global_coor = dst_coor.clone()  # 克隆S2的坐标，作为全局坐标的参考

        _DEBUG = False
        if _DEBUG:
            from utils.visualization import show_pcd
            for src, dst, src_global, dst_global in zip(src_coor, dst_coor, src_global_coor, dst_global_coor):
                show_pcd([src.T, dst.T], [[1, 0, 0], [0, 1, 0]],
                         window_name=f'src={src.shape[1]}, dst={dst.shape[1]} | local')
                show_pcd([src_global.T, dst_global.T], [[1, 0, 0], [0, 1, 0]],
                         window_name=f'src={src.shape[1]}, dst={dst.shape[1]} | gt')

        src_pairing_fea, dst_pairing_fea, src_coarse_pairing_fea, dst_coarse_pairing_fea, \
        src_offset_res, dst_offset_res = \
            self.decoder(
                torch.cat([src_fea, src_coor], dim=1),  # 将S1中每个点的特征与坐标拼接 (16, 128+3, 256)
                torch.cat([dst_fea, dst_coor], dim=1),  # 将S2中每个点的特征与坐标拼接 (16, 128+3, 256)
                src_padding_mask=src_mask,  # S1掩码 (16, 256)
                dst_padding_mask=dst_mask,  # S2掩码 (16, 256)
                gt_Rt=(R, T)    # 获取S1和S2的第一对 对应帧的真实旋转矩阵和真实平移矩阵 ((16, 3, 3), (16, 3, 1))
            )  # 返回 相似度特征、粗配对特征，偏移残差

        loss, top1_pairing_acc, loss_pairing, loss_coarse_pairing, loss_offset = \
            self.criterion(
                src_global_coor=src_global_coor, dst_global_coor=dst_global_coor,   # S1,S2 点云全局坐标
                src_padding_mask=src_mask, dst_padding_mask=dst_mask,   # S1,S2 的点云填充标记
                src_pairing_fea=src_pairing_fea, dst_pairing_fea=dst_pairing_fea,   # S1,S2 的相似度特征
                src_coarse_pairing_fea=src_coarse_pairing_fea, dst_coarse_pairing_fea=dst_coarse_pairing_fea,
                # S1,S2 的粗配对特征
                src_offset_res=src_offset_res, dst_offset_res=dst_offset_res,    # 双向位姿偏移残差
                label_loss=label_loss
            )  # 使用定义的损失函数计算配准过程中的损失和评价指标


        offset_err = (torch.norm(src_offset_res.detach(), p=2, dim=1).mean() +
                      torch.norm(dst_offset_res.detach(), p=2, dim=1).mean()).item() / 2  # 计算偏移残差的L2范数平均值作为误差度量

        metric_dict = {  # 评价指标字典
            'loss': loss.item(),  # 总损失
            'top1_acc': top1_pairing_acc,  # 点云配准正确率
            'loss_p': loss_pairing,  # 配对损失
            'loss_c': loss_coarse_pairing,   # 粗配对损失
            'loss_o': loss_offset,   # 偏移损失
            'offset_err': offset_err,  # 偏移残差
            'seg_acc': accuracy,  # 语义分割标签准确率
            'loss_seg': label_loss  # 语义分割标签交叉熵损失
        }
        return loss, metric_dict  # 损失值、指标字典

    """ 回环检测部分 """
    def _train_loop_detection(self, src_pcd: Tensor, src_R: Tensor, src_T: Tensor, src_mask: Tensor, src_calib: Tensor,
                              dst_pcd: Tensor, dst_R: Tensor, dst_T: Tensor, dst_mask: Tensor, dst_calib: Tensor
                              ) -> Tuple[Tensor, dict]:


        B = src_pcd.shape[0]  # 批次数 B
        stacked_pcd = torch.cat([src_pcd, dst_pcd], dim=0)  # 源点云和目标点云拼接，形成(2B, C, N)的张量
        stacked_mask = torch.cat([src_mask, dst_mask], dim=0)  # 源掩码和目标掩码拼接，形成(2B, N)的张量

        # encoder 提取
        coor, fea, mask, index = self.encoder(stacked_pcd, stacked_mask)  # encoder 提取坐标、特征、掩码

        # 语义分割
        mask = self.label_decoder.label_inference(points_coor=coor, points_fea=fea, mask=mask)

        # 输入数据处理，以匹配后续操作
        coor = coor * self.args.slam_system.coor_scale  # 缩放坐标，以匹配 SLAM 的收入
        src_coor, dst_coor = coor[:B], coor[B:]  # 坐标 --> 源坐标、目标坐标
        src_fea, dst_fea = fea[:B], fea[B:]  # 特征 --> 源特征、目标特征
        src_mask, dst_mask = mask[:B], mask[B:]  # 掩码 --> 源掩码、目标掩码
        src_descriptor = torch.cat([src_fea, src_coor], dim=1)  # 源特征、源坐标 --> 源描述符
        dst_descriptor = torch.cat([dst_fea, dst_coor], dim=1)  # 目标特征、目标坐标 --> 目标描述符

        # decoder 回环检测, 返回 回环概率
        loop_pred = self.decoder.loop_detection_forward(
            src_descriptor=src_descriptor, dst_descriptor=dst_descriptor,
            src_padding_mask=src_mask, dst_padding_mask=dst_mask,
        )

        dis = torch.norm((src_T - dst_T).squeeze(-1), p=2, dim=-1)  # 源、目标 之间的距离
        loop_gt = (dis <= self.args.train.loop_detection.distance).float()  # 距离 < 设定阈值时 认定发生回环
        loop_loss = F.binary_cross_entropy(input=loop_pred, target=loop_gt)  # 二元交叉熵损失计算回环检测损失

        # 回环预测结果二值化，按阈值 0.5 转为 True、False
        loop_pred_binary = loop_pred > 0.5
        loop_gt_binary = loop_gt.bool()

        precision = (torch.sum(loop_pred_binary == loop_gt_binary) / loop_pred_binary.shape[0]).item()  # 回环检测精度
        if loop_gt_binary.sum() > 0:
            recall = torch.sum(loop_pred_binary[loop_gt_binary]) / loop_gt_binary.sum()  # 计算召回率
            recall = recall.item()
        else:
            recall = 1.0  # 没有正例时，召回率设为1.0
        negative_gt_mask = ~loop_gt_binary  # 计算负例掩码。
        if negative_gt_mask.sum() > 0:  # 如果负例掩码中有负例
            false_positive = torch.sum(loop_pred_binary[negative_gt_mask]) / negative_gt_mask.sum()  # 计算假阳性率。
            false_positive = false_positive.item()
        else:
            false_positive = 0.0  # 如果没有负例，则假阳性率设为0.0。

        metric_dict = {  # 创建一个字典，存储损失和评价指标。
            'loss_loop': loop_loss.item(),  # 回环检测损失
            'loop_precision': precision,  # 回环检测精度。
            'loop_recall': recall,  # 回环检测召回率。
            'loop_false_positive': false_positive  # 回环检测假阳性率。
        }
        return loop_loss, metric_dict  # 返回回环检测损失和指标字典。

    def registration(self):
        self._forward_method = self._train_registration  # 设置模型的前向传播方法为_train_registration。
        # 训练配准时冻结回环检测部分
        for name, param in self.named_parameters():
            if 'loop' in name:  # 如果参数名称中包含'loop'，表示属于回环检测部分
                param.requires_grad = False  # 冻结该部分参数，即不更新其梯度。
            else:
                param.requires_grad = True  # 启用非回环检测部分的参数梯度更新。

    def loop_detection(self):
        self._forward_method = self._train_loop_detection  # 设置模型的前向传播方法为_train_loop_detection。
        # 训练回环检测时冻结其他网络
        for name, param in self.named_parameters():
            if 'loop' in name:  # 如果参数名称中包含'loop'，表示属于回环检测部分
                param.requires_grad = True  # 启用回环检测部分的参数梯度更新。
            else:
                param.requires_grad = False  # 冻结非回环检测部分的参数，即不更新其梯度。

    def _get_accurate_RT(self, src_index: np.ndarray, dst_index: np.ndarray, refined_SE3_file: str,
                         src_R: Tensor, src_T: Tensor, src_calib: Tensor,
                         dst_R: Tensor, dst_T: Tensor, dst_calib: Tensor, bridge_index=None,
                         src_pcd=None, dst_pcd=None) -> Tuple[Tensor, Tensor]:
        assert len(src_index) == len(dst_index) == len(refined_SE3_file) == len(src_R) == len(src_T) == len(src_calib) \
               == len(dst_R) == len(dst_T) == len(dst_calib)  # 确保各个输入长度一致。
        B = len(src_index)  # 一张局部地图由 B 帧点云组成
        device = src_calib.device  # 获取校准张量的设备（CPU或GPU）
        if bridge_index is None:
            bridge_index = [None] * B  # 如果未提供桥接索引，则初始化为None。
        else:
            assert len(bridge_index) == B  # 如果提供了桥接索引，确保其与其他输入一致
        use_squeeze = src_index.ndim == 1 and dst_index.ndim == 1  # 如果源、目标索引是一维数组，则需要调整维度

        src_index = src_index[:, np.newaxis] if src_index.ndim == 1 else src_index  # 如果S1索引一维，扩展维度 (16) --> (16,1)
        dst_index = dst_index[:, np.newaxis] if dst_index.ndim == 1 else dst_index  # 如果S2索引一维，扩展维度 (16) --> (16,1)
        src_R = src_R.unsqueeze(1) if src_R.ndim == 3 else src_R  # 如果S1旋转矩阵3维，扩展维度 (16, 3, 3) --> (16, 1, 3, 3)
        src_T = src_T.unsqueeze(1) if src_T.ndim == 3 else src_T  # 如果S1平移向量3维，扩展维度。(16, 3, 1) --> (16, 1, 3, 1)
        src_calib = src_calib.unsqueeze(1) if src_calib.ndim == 3 else src_calib  # 如果S1校准矩阵是3维张量，扩展维度 (16, 4, 4) --> (16, 1, 4, 4)
        dst_R = dst_R.unsqueeze(1) if dst_R.ndim == 3 else dst_R  # 如果S2旋转矩阵3维，扩展维度 (16, 3, 3) --> (16, 1, 3, 3)
        dst_T = dst_T.unsqueeze(1) if dst_T.ndim == 3 else dst_T  # 如果S2平移向量3维，扩展维度 (16, 3, 1) --> (16, 1, 3, 1)
        dst_calib = dst_calib.unsqueeze(1) if dst_calib.ndim == 3 else dst_calib  # 如果S2校准矩阵3维，扩展维度 (16, 4, 4) --> (16, 1, 4, 4)

        """取源索引和目标索引帧数中的最大值，作为统一处理的目标帧数 S"""
        S = max(src_index.shape[1], dst_index.shape[1])
        if src_index.shape[1] < S and src_index.shape[1] == 1:
            src_index = src_index.repeat(repeats=S, axis=1)
            src_R = src_R.repeat(1, S, 1, 1)
            src_T = src_T.repeat(1, S, 1, 1)
            src_calib = src_calib.repeat(1, S, 1, 1)

        if dst_index.shape[1] < S and dst_index.shape[1] == 1:
            dst_index = dst_index.repeat(repeats=S, axis=1)
            dst_R = dst_R.repeat(1, S, 1, 1)
            dst_T = dst_T.repeat(1, S, 1, 1)
            dst_calib = dst_calib.repeat(1, S, 1, 1)

        # 逐场景校准
        _DEBUG, show_index = src_pcd is not None, 0
        R_list, T_list = [], []  # 初始化旋转矩阵列表和平移向量列表
        """遍历S1，S2中所有对应点及其特征"""
        for b_src_i, b_src_R, b_src_T, b_src_c, b_dst_i, b_dst_R, b_dst_T, b_dst_c, file, bridge in \
                zip(src_index, src_R, src_T, src_calib, dst_index, dst_R, dst_T, dst_calib, refined_SE3_file,
                    bridge_index):
            SE3_dict = self._load_refined_SE3(file)  # 从文件中加载真实位姿矩阵(标准答案)
            if SE3_dict is not None:  # 如果位姿矩阵不为空
                b_SE3 = []
                # 遍历 S1 , S2 中每一帧的索引校准矩阵
                for i, (s, d, s_calib, d_calib) in enumerate(zip(b_src_i, b_dst_i, b_src_c, b_dst_c)):
                    try:
                        icp_SE3 = torch.from_numpy(get_SE3_from_dict(SE3_dict, s, d, bridge)).float().to(
                            device)  # 从 SE3_dict 字典中获取索引为(S1某帧,S2某帧)的真实位姿矩阵(标准答案)
                        current_SE3 = d_calib @ icp_SE3 @ s_calib.inverse()  # 计算真实位姿变换矩阵(标准答案)
                    except:
                        r, t = rt_global_to_relative(center_R=b_dst_R[i], center_T=b_dst_T[i],
                                                     other_R=b_src_R[i],
                                                     other_T=b_src_T[i])
                        current_SE3 = torch.eye(4, dtype=torch.float32, device=device)
                        current_SE3[:3, :3] = r
                        current_SE3[:3, 3:] = t
                        import os
                        src_SE3 = torch.eye(4, dtype=torch.float32, device=device)
                        dst_SE3 = torch.eye(4, dtype=torch.float32, device=device)
                        src_SE3[:3, :3] = b_src_R[i]
                        src_SE3[:3, 3:] = b_src_T[i]
                        dst_SE3[:3, :3] = b_dst_R[i]
                        dst_SE3[:3, 3:] = b_dst_T[i]
                        gt_ori_relative_SE3 = d_calib.inverse() @ dst_SE3.inverse() @ src_SE3 @ s_calib
                        dist = torch.norm(gt_ori_relative_SE3[:3, -1]).item()
                        logger.warning(f'Found a pair without icp_SE3, in {os.path.dirname(file)}:({s}, {d}), '
                                       f'{dist=:.2f}, use gt instead')
                    b_SE3.append(current_SE3)  # 将所有真实位姿变换矩阵组成一个列表
                b_SE3 = torch.stack(b_SE3, dim=0)  # 列表 --> 张量
                R_list.append(b_SE3[:, :3, :3])  # 从真实位姿变换列表中提取 真实旋转向量列表
                T_list.append(b_SE3[:, :3, 3:])  # 从真实位姿变换列表中提取 真实平移向量列表
            else:
                R, T = rt_global_to_relative(center_R=b_dst_R, center_T=b_dst_T, other_R=b_src_R,
                                             other_T=b_src_T)
                R_list.append(R)
                T_list.append(T)

            if _DEBUG:
                gt_Rs, gt_Ts = rt_global_to_relative(center_R=dst_R[show_index], center_T=dst_T[show_index],
                                                     other_R=src_R[show_index],
                                                     other_T=src_T[show_index])
                from utils.visualization import show_pcd
                src_xyz, dst_xyz = src_pcd[show_index], dst_pcd[show_index]
                src_xyz = src_xyz[None].repeat(S, 1, 1) if src_xyz.ndim == 2 else src_xyz
                dst_xyz = dst_xyz[None].repeat(S, 1, 1) if dst_xyz.ndim == 2 else dst_xyz
                for src, dst, gt_R, gt_T, icp_R, icp_T in zip(src_xyz, dst_xyz, gt_Rs, gt_Ts, R_list[-1],
                                                              T_list[-1]):
                    src_gt = (gt_R @ src + gt_T).T
                    src_icp = (icp_R @ src + icp_T).T
                    src, dst = src.T, dst.T
                    show_pcd([src, dst], [[1, 0, 0], [0, 1, 0]],
                             window_name=f'local | {S=} | {file}')
                    show_pcd([src_gt, dst], [[1, 0, 0], [0, 1, 0]],
                             window_name=f'gt | {S=} | {file}')
                    show_pcd([src_icp, dst], [[1, 0, 0], [0, 1, 0]],
                             window_name=f'icp | {S=} | {file}')
                if S > 1:
                    src_xyz_gt = (gt_Rs @ src_xyz + gt_Ts).transpose(1, 2).reshape(-1, 3)
                    src_xyz_icp = (R_list[-1] @ src_xyz + T_list[-1]).transpose(1, 2).reshape(-1,
                                                                                              3)
                    dst_xyz_map = dst_xyz.transpose(1, 2).reshape(-1, 3)
                    show_pcd([src_xyz_gt, dst_xyz_map], [[1, 0, 0], [0, 1, 0]],
                             window_name=f'gt map | {S=} | {file}')
                    show_pcd([src_xyz_icp, dst_xyz_map], [[1, 0, 0], [0, 1, 0]],
                             window_name=f'icp map | {S=} | {file}')
                show_index += 1

        R, T = torch.stack(R_list, dim=0), torch.stack(T_list, dim=0)  # 旋转矩阵列表，平移向量列表 --> 张量
        if use_squeeze:  # 如果需要调整维度
            R, T = R.squeeze(1), T.squeeze(1)  # 压缩旋转矩阵和平移向量的维度。
            """
                R：(16, 1, 3, 3) --> (16, 3, 3)
                T：(16, 1, 3, 1) --> (16, 3, 1)
            """
        return R, T  # 返回真实旋转矩阵和真实平移向量。

    def _load_refined_SE3(self, file):
        if file not in self.refined_SE3_cache.keys():  # 如果文件未被缓存
            if file != '':
                with open(file, 'rb') as f:
                    refined_SE3: Dict[Tuple[int, int], np.ndarray] = pickle.load(f)  # 从文件中加载精化后的SE3变换矩阵。
            else:
                refined_SE3 = None  # 如果文件为空，设置refined_SE3为None。
            self.refined_SE3_cache[file] = refined_SE3  # 将加载的SE3变换矩阵缓存到字典中。
        return self.refined_SE3_cache[file]  # 返回缓存的SE3变换矩阵。


def get_SE3_from_dict(SE3_dict: Dict[Tuple[int, int], np.ndarray], s: int, d: int, bridge=None) -> np.ndarray:
    if s == d:  # 如果源点和目标点相同，则变换矩阵为单位矩阵。
        SE3 = np.eye(4)
    elif s < d:  # 如果源点的索引小于目标点的索引
        SE3 = SE3_dict.get((s, d), None)  # 从字典中获取s到d的SE3变换矩阵。
        if SE3 is not None:
            SE3 = linalg.inv(SE3)  # 如果找到变换矩阵，则计算其逆矩阵。
    else:
        SE3 = SE3_dict.get((d, s), None)  # 如果源点的索引大于目标点的索引，则直接获取d到s的SE3变换矩阵。

    if SE3 is None:  # 如果SE3矩阵为None，表示字典中没有找到直接的变换矩阵。
        SE3_s2b = get_SE3_from_dict(SE3_dict, s, bridge, None)  # 递归调用自己，从源点到桥接点的SE3变换矩阵。
        SE3_b2d = get_SE3_from_dict(SE3_dict, bridge, d, None)  # 递归调用自己，从桥接点到目标点的SE3变换矩阵。
        SE3 = SE3_b2d @ SE3_s2b  # 计算最终的变换矩阵为桥接点变换矩阵的组合。

    return SE3  # 返回计算得到的SE3变换矩阵。

