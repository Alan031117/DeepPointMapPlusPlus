import torch  # 导入 PyTorch 库，用于深度学习模型的实现和计算
import torch.nn as nn  # 导入 PyTorch 的神经网络模块，提供各种神经网络层和相关功能
import torch.nn.functional as F  # 导入 PyTorch 的函数式接口，提供激活函数、损失函数等
from torch import Tensor as Tensor  # 从 PyTorch 导入 Tensor 类型，用于定义张量
from typing import List, Union, Tuple  # 导入类型提示工具，用于定义函数参数和返回值的类型
from network.decoder.descriptor_attention import DescriptorAttentionLayer, PositionEmbeddingCoordsSine  # 从自定义模块中导入描述符注意力层和位置嵌入层
from network.decoder.heads import CoarsePairingHead, SimilarityHead, OffsetHead, LoopHead, LoopHeadV2  # 从自定义模块中导入不同的网络头部，用于处理配对、相似性、偏移和回环检测等任务


class Decoder(nn.Module):  # 定义一个名为 Decoder 的类，继承自 nn.Module，用于解码器网络
    """
    解码器，计算两组descriptor之间的关系
    """

    def __init__(self, args):  # 初始化函数，接收一个参数 args，用于配置解码器
        super().__init__()  # 调用父类 nn.Module 的初始化方法
        self.args = args  # 保存传入的参数配置
        self.decoder_cfg = self.args.decoder  # 获取解码器相关的配置参数

        self.in_channel = self.decoder_cfg.in_channel  # 获取输入通道数
        self.model_channel = self.decoder_cfg.model_channel  # 获取模型的通道数
        self.attention_layers = self.decoder_cfg.attention_layers  # 获取注意力层的数量

        self.tau = self.args.loss.tau  # 获取损失函数中的参数 tau，用于控制损失的缩放
        descriptor_pairing_method = getattr(self.args.slam_system, 'descriptor_pairing_method', 'topk')  # 获取描述符配对方法，默认为 'topk'
        if descriptor_pairing_method == 'topk':  # 如果配对方法是 'topk'
            self.descriptor_pairing_method = self._descriptor_pairing  # 使用 _descriptor_pairing 方法进行配对
        elif descriptor_pairing_method == 'one2many':  # 如果配对方法是 'one2many'
            self.descriptor_pairing_method = self._descriptor_pairing_one2many  # 使用 _descriptor_pairing_one2many 方法进行配对

        # ======================== Build Network ========================
        self.projection = nn.Conv1d(in_channels=self.in_channel, out_channels=self.model_channel, kernel_size=1)  # 定义一个 1D 卷积层用于通道转换
        self.pos_embedding_layer = PositionEmbeddingCoordsSine(in_dim=3, emb_dim=self.model_channel)  # 定义位置嵌入层，用于对坐标进行嵌入
        self.descriptor_attention = nn.ModuleList()  # 定义一个模块列表，用于存储多个描述符注意力层
        for _ in range(self.attention_layers):  # 根据配置的注意力层数量，循环添加描述符注意力层
            self.descriptor_attention.append(DescriptorAttentionLayer(emb_dim=self.model_channel))  # 将描述符注意力层添加到模块列表中
        self.similarity_head = SimilarityHead(emb_dim=self.model_channel)  # 定义相似性头部，用于计算描述符之间的相似性
        self.offset_head = OffsetHead(emb_dim=self.model_channel * 2)  # 定义偏移头部，用于计算描述符之间的偏移量
        # self.loop_head = LoopHead(emb_dim=self.model_channel)  # 定义回环检测头部，用于回环检测任务
        self.loop_head = LoopHeadV2(emb_dim=self.model_channel)  # 定义回环检测头部，用于回环检测任务
        self.coarse_pairing_head = CoarsePairingHead(emb_dim=self.in_channel)  # 定义粗配对头部，用于初步配对描述符

    def forward(self, src_descriptor: Tensor, dst_descriptor: Tensor,
                src_padding_mask: Tensor = None, dst_padding_mask: Tensor = None,
                gt_Rt: Tuple[Tensor, Tensor] = None) -> List[Tensor]:  # 定义前向传播函数，用于计算描述符之间的关系
        """
        :param src_descriptor: (B, F+3, M) 一组descriptor，"F"表示特征维，"3"表示坐标维
        :param dst_descriptor: (B, F+3, N) 另一组descriptor，"F"表示特征维，"3"表示坐标维
        :param src_padding_mask: (B, M) padding掩码，True为填充点，False为正常点
        :param dst_padding_mask: (B, N) padding掩码，True为填充点，False为正常点
        :param gt_Rt: ((B, 3, 3), (B, 3, 1)) 位姿的真实值
        """
        assert self.training, 'forward is not available during inference!'  # 只在训练模式下可用
        return self._train_forward(src_descriptor, dst_descriptor, src_padding_mask, dst_padding_mask, gt_Rt)  # 前向传播

    def _train_forward(self, src_descriptor: Tensor, dst_descriptor: Tensor,
                       src_padding_mask: Tensor, dst_padding_mask: Tensor,
                       gt_Rt: Tuple[Tensor, Tensor]) -> List[Tensor]:  # 定义训练模式下的前向传播函数
        assert gt_Rt is not None, 'gt_Rt must be provided during training'  # 确保位姿存在
        self.gt_R, self.gt_T = gt_Rt  # 将真实位姿拆成 旋转矩阵+平移矩阵

        '''统一描述符 -> 粗配对特征'''
        src_fea, dst_fea = src_descriptor[:, :-3, :], dst_descriptor[:, :-3, :]  # 从S1，S2描述符中提取特征（去掉坐标）(B, F, M)
        src_coarse_pairing_fea = self.coarse_pairing_head(src_fea)  # 使用粗配对头，计算S1粗配对特征
        dst_coarse_pairing_fea = self.coarse_pairing_head(dst_fea)  # 使用粗配对头，计算S2粗配对特征

        '''统一描述符 -> 相关描述符'''
        # 通过注意力机制，计算相关描述符，并提取特征和坐标
        src_corr_descriptor, dst_corr_descriptor = \
            self._descriptor_attention_forward(src_descriptor, dst_descriptor, src_padding_mask, dst_padding_mask)  # 调用描述符注意力机制
        src_fea, src_xyz = src_corr_descriptor[:, :-3, :], src_corr_descriptor[:, -3:, :]
        # S1相关描述符 = 特征(16, 256, 256) + 坐标(16, 3, 256)
        dst_fea, dst_xyz = dst_corr_descriptor[:, :-3, :], dst_corr_descriptor[:, -3:, :]
        # S2相关描述符 = 特征(16, 256, 256) + 坐标(16, 3, 256)

        '''相关描述符[特征] -> 相似性特征'''
        # 使用相似性头，计算S1和S2描述符云之间的对应关系
        src_pairing_fea = self.similarity_head(src_fea)  # 计算S1描述符的相似性特征(16, 256, 256)
        dst_pairing_fea = self.similarity_head(dst_fea)  # 计算S2描述符的相似性特征(16, 256, 256)

        '''真实位姿 -> 描述符偏移量'''
        src_gt_xyz = self.gt_R @ src_xyz + self.gt_T  # 使用真实位姿与S1，预测S2位置
        src_gt_xyz, dst_gt_xyz = src_gt_xyz.transpose(1, 2), dst_xyz.transpose(1, 2)
        # 转置两个坐标矩阵S2预测坐标与S2实际坐标(16, 3, 256) --> (16, 256, 3)
        dist = torch.sum(torch.square(src_gt_xyz.unsqueeze(2) - dst_gt_xyz.unsqueeze(1)), dim=-1)
        # 计算S2预测位置与实际位置的差，并生成距离矩阵 (B, M, N)
        dist_mask = dist <= (self.args.loss.eps_offset ** 2)
        # 生成距离掩码矩阵dist_mask，选出不符合要求的帧 (B, M, N)

        # 从 dist_mask 剔除 S1,S2 中的无效点
        dist_mask &= ~src_padding_mask.unsqueeze(2)
        dist_mask &= ~dst_padding_mask.unsqueeze(1)

        # 根据距离掩码，提取所有成对的描述符属性
        pairs_index = torch.nonzero(dist_mask)  # 找到掩码为 True 的索引
        batch_index = pairs_index[:, 0]  # 提取指定批次索引
        src_index = pairs_index[:, 1]  # 提取指定 S1 描述符索引
        dst_index = pairs_index[:, 2]  # 提取指定 S2 描述符索引
        src_pair_gt_xyz = src_gt_xyz[batch_index, src_index, :]  # 提取指定 S1 描述符的位姿
        dst_pair_gt_xyz = dst_gt_xyz[batch_index, dst_index, :]  # 提取指定 S2 描述符的位姿
        src_pair_fea = src_fea.transpose(1, 2)[batch_index, src_index, :]  # 提取指定 S1 描述符的特征
        dst_pair_fea = dst_fea.transpose(1, 2)[batch_index, dst_index, :]  # 提取指定 S2 描述符的特征

        # 计算所有成对描述符的 ground truth 偏移量
        src2dst_R = torch.cat([self.gt_R[i].unsqueeze(0).repeat(num, 1, 1)
                               for i, num in enumerate(torch.sum(dist_mask, dim=[1, 2]))], dim=0)
        # 对旋转矩阵 self.gt_R 进行扩展，使其匹配每对描述符

        src_offset_gt = src2dst_R.transpose(1, 2) @ (dst_pair_gt_xyz - src_pair_gt_xyz).unsqueeze(2)  # 计算 S1 -> S2 的真实位姿偏移量
        dst_offset_gt = (src_pair_gt_xyz - dst_pair_gt_xyz).unsqueeze(2)  # 计算 S2 -> S1 的真实位姿偏移量

        src_offset_fea = torch.cat([src_pair_fea, dst_pair_fea], dim=1).unsqueeze(2)
        # 将源和目标描述符的特征拼接，并添加一个额外维度
        dst_offset_fea = torch.cat([dst_pair_fea, src_pair_fea], dim=1).unsqueeze(2)
        # 将目标和源描述符的特征拼接，并添加一个额外维度
        src_offset = self.offset_head(src_offset_fea)  # 使用偏移头，预测匹配点间 S1 -> S2 的位姿偏移量
        dst_offset = self.offset_head(dst_offset_fea)  # 使用偏移头，预测匹配点间 S2 -> S1 的位姿偏移量

        src_offset_res, dst_offset_res = src_offset - src_offset_gt, dst_offset - dst_offset_gt
        # 计算 预测的位姿偏移量 与 真实位姿偏移量 之间的残差

        return [src_pairing_fea, dst_pairing_fea,
                src_coarse_pairing_fea, dst_coarse_pairing_fea,
                src_offset_res, dst_offset_res]
        # 返回 相似性特征、粗配对特征、位姿偏移量残差

    def registration_forward(self, src_descriptor: Tensor, dst_descriptor: Tensor,
                             src_padding_mask: Tensor = None, dst_padding_mask: Tensor = None,
                             num_sample: Union[int, float] = 0.5) \
            -> Tuple[Tensor, Tensor, Tensor, Union[List[float], float]]:
        """
        :param src_descriptor: ((B), F+3, M) 一组descriptor，"F"表示特征维，"3"表示坐标维
        :param dst_descriptor: ((B), F+3, N) 另一组descriptor，"F"表示特征维，"3"表示坐标维
        :param src_padding_mask: ((B), M) padding掩码，True为填充点，False为正常点
        :param dst_padding_mask: ((B), N) padding掩码，True为填充点，False为正常点
        :param num_sample: 采样点数量，int型表示数量，float型表示比例
        """
        # 推理阶段不进行批量化处理，可以不提供batch维度
        if src_descriptor.ndim == 2 and dst_descriptor.ndim == 2:  # 如果输入的描述符只有两个维度（没有批量维度）
            batch = False  # 标记为非批量模式
            src_descriptor = src_descriptor.unsqueeze(0)  # 为源描述符增加一个批量维度
            dst_descriptor = dst_descriptor.unsqueeze(0)  # 为目标描述符增加一个批量维度
        else:
            batch = True  # 否则标记为批量模式

        '''unified descriptor -> correlated descriptor'''
        # 通过描述符注意力机制计算相关描述符
        src_corr_descriptor, dst_corr_descriptor = \
            self._descriptor_attention_forward(src_descriptor, dst_descriptor, src_padding_mask, dst_padding_mask)

        '''pairing correlated descriptor -> merge offset -> pairing coordinate'''
        # 进行描述符配对并获取配对的坐标和配对置信度
        src_pairing_descriptor, dst_pairing_descriptor, pairing_conf = \
            self.descriptor_pairing_method(src_corr_descriptor, dst_corr_descriptor, num_sample)
        src_pairing_coor, dst_pairing_coor, pairing_conf = \
            self._get_corres_sets(src_pairing_descriptor, dst_pairing_descriptor, pairing_conf)

        from utils.global_dict import global_dict  # 引入全局字典，用于存储调试信息
        if False:  # 如果条件为 True，则会执行该代码块（这里为 False，因此不会执行）
            src_SE3 = global_dict['src_SE3']  # 从全局字典中获取源SE3矩阵
            dst_SE3 = global_dict['dst_SE3']  # 从全局字典中获取目标SE3矩阵
            offset_s2d = global_dict['offset'][0]  # 获取源到目标的偏移量
            src_pairing_id = global_dict['src_pairing_id'][0]  # 获取源配对的索引
            dst_pairing_id = global_dict['dst_pairing_id'][0]  # 获取目标配对的索引
            src_pcd = global_dict['src_pcd']  # 获取源点云数据
            dst_pcd = global_dict['dst_pcd']  # 获取目标点云数据

            src_R, src_T = src_SE3[:3, :3], src_SE3[:3, 3:]  # 获取源的旋转矩阵和平移向量
            dst_R, dst_T = dst_SE3[:3, :3], dst_SE3[:3, 3:]  # 获取目标的旋转矩阵和平移向量
            src_xyz = src_descriptor[0, -3:, :]  # 获取源的坐标
            dst_xyz = dst_descriptor[0, -3:, :]  # 获取目标的坐标

            offset_s2d_all = torch.zeros_like(src_xyz)  # 初始化一个与源坐标形状相同的零张量
            offset_s2d_all[:, src_pairing_id] = offset_s2d  # 根据配对索引设置偏移量
            src_s2d_xyz = src_xyz + offset_s2d_all  # 应用偏移量后得到的源坐标

            src_xyz = src_R @ src_xyz + src_T  # 将源坐标变换到目标坐标系下
            src_s2d_xyz = src_R @ src_s2d_xyz + src_T  # 将应用偏移量后的源坐标变换到目标坐标系下
            dst_xyz = dst_R @ dst_xyz + dst_T  # 将目标坐标变换到目标坐标系下
            src_pcd = src_R @ src_pcd + src_T  # 变换源点云数据到目标坐标系下
            dst_pcd = dst_R @ dst_pcd + dst_T  # 变换目标点云数据到目标坐标系下
            pcd = torch.cat([src_pcd, dst_pcd], dim=1)  # 合并源和目标的点云数据

            from utils.visualization import show_pcd  # 引入用于可视化的模块
            # show_pcd([src_xyz.T, dst_xyz.T], [[1, 0, 0], [0, 0, 1]])  # 可视化源和目标的点云数据（未应用偏移量）
            # show_pcd([src_s2d_xyz.T, dst_xyz.T], [[1, 0, 0], [0, 0, 1]])  # 可视化源和目标的点云数据（应用了偏移量）

            pairing_id = torch.zeros(size=(src_xyz.shape[-1],), dtype=torch.int)  # 初始化一个与源点云数量相同的零张量
            pairing_id[:] = -1  # 将所有值设置为 -1，表示未配对
            pairing_id[src_pairing_id] = dst_pairing_id.int()  # 根据配对索引设置配对信息
            conf = torch.zeros(size=(src_xyz.shape[-1],), dtype=torch.float)  # 初始化一个与源点云数量相同的零张量
            conf[src_pairing_id] = pairing_conf[0, :src_pairing_id.shape[0]]  # 根据配对置信度设置置信度信息

            with open(r'key_points.txt', 'w') as f:  # 打开一个文件用于写入关键点信息
                for i in range(src_xyz.shape[-1]):  # 遍历每个源点云
                    src_xyz_i = src_xyz.T[i].tolist()  # 获取源点云的坐标并转为列表
                    src_s2d_xyz_i = src_s2d_xyz.T[i].tolist()  # 获取应用了偏移量的源点云坐标并转为列表
                    dst_xyz_i = dst_xyz.T[i].tolist()  # 获取目标点云的坐标并转为列表
                    pairing_id_i = [pairing_id[i].item()]  # 获取配对ID
                    conf_i = [conf[i].item()]  # 获取置信度值
                    for data in src_xyz_i + src_s2d_xyz_i + dst_xyz_i + pairing_id_i + conf_i:  # 将所有数据拼接成一行
                        f.write(f'{data} ')  # 写入文件
                    f.write('\n')  # 换行

            with open(r'pcd.txt', 'w') as f:  # 打开一个文件用于写入点云数据
                for i in range(pcd.shape[-1]):  # 遍历每个点云
                    pcd_i = pcd.T[i].tolist()  # 获取点云的坐标并转为列表
                    for data in pcd_i:  # 遍历每个坐标分量
                        f.write(f'{data} ')  # 写入文件
                    f.write('\n')  # 换行

            import matplotlib.pyplot as plt  # 引入用于绘图的模块
            plt.figure(figsize=(20, 20), dpi=300)  # 创建一个20x20英寸，分辨率为300 dpi的图形
            src_xyz = src_xyz.T.numpy()  # 将源坐标转为NumPy数组
            src_s2d_xyz = src_s2d_xyz.T.numpy()  # 将应用了偏移量的源坐标转为NumPy数组
            dst_xyz = dst_xyz.T.numpy()  # 将目标坐标转为NumPy数组
            pcd = pcd.T.numpy()  # 将点云数据转为NumPy数组

            plt.subplot(2, 2, 1)  # 创建一个2x2网格的子图，选择第一个子图
            plt.title('w/o offset')  # 设置子图标题
            plt.axis('equal')  # 设置子图坐标轴比例相等
            plt.scatter(x=pcd[:, 0], y=pcd[:, 1], c=pcd[:, 2], s=1, alpha=0.1)  # 绘制点云散点图
            plt.scatter(x=src_xyz[:, 0], y=src_xyz[:, 1], c='red', s=10, alpha=0.5)  # 绘制源点云散点图
            plt.scatter(x=dst_xyz[:, 0], y=dst_xyz[:, 1], c='blue', s=10, alpha=0.5)  # 绘制目标点云散点图
            plt.plot([src_xyz[src_pairing_id, 0], dst_xyz[dst_pairing_id, 0]],
                     [src_xyz[src_pairing_id, 1], dst_xyz[dst_pairing_id, 1]],
                     c='#FF8C00', linewidth=3)  # 绘制源和目标点之间的连接线

            plt.subplot(2, 2, 2)  # 创建第二个子图
            plt.title('w offset')  # 设置子图标题
            plt.axis('equal')  # 设置子图坐标轴比例相等
            plt.scatter(x=pcd[:, 0], y=pcd[:, 1], c=pcd[:, 2], s=1, alpha=0.1)  # 绘制点云散点图
            plt.scatter(x=src_s2d_xyz[:, 0], y=src_s2d_xyz[:, 1], c='red', s=10, alpha=0.5)  # 绘制应用了偏移量的源点云散点图
            plt.scatter(x=dst_xyz[:, 0], y=dst_xyz[:, 1], c='blue', s=10, alpha=0.5)  # 绘制目标点云散点图
            plt.plot([src_s2d_xyz[src_pairing_id, 0], dst_xyz[dst_pairing_id, 0]],
                     [src_s2d_xyz[src_pairing_id, 1], dst_xyz[dst_pairing_id, 1]],
                     c='#FF8C00', linewidth=3)  # 绘制应用了偏移量的源和目标点之间的连接线

            import numpy as np  # 引入NumPy模块
            r = [0, 20, 0, 20]  # 设置区域范围 (x_min, x_max, y_min, y_max)

            # 先扣出pcd与src的范围
            region_mask = (pcd[:, 0] > r[0]) & (pcd[:, 0] < r[1]) & (pcd[:, 1] > r[2]) & (pcd[:, 1] < r[3])  # 创建区域掩码
            region_pcd = pcd[region_mask]  # 筛选出区域内的点云数据
            region_mask = (src_xyz[:, 0] > r[0]) & (src_xyz[:, 0] < r[1]) & (src_xyz[:, 1] > r[2]) & (
                        src_xyz[:, 1] < r[3])
            region_src_xyz = src_xyz[region_mask]  # 筛选出区域内的源点云
            region_src_s2d_xyz = src_s2d_xyz[region_mask]  # 筛选出区域内的应用了偏移量的源点云

            # 扣出src范围内的配对情况
            pairing_ids = np.stack([src_pairing_id, dst_pairing_id], axis=1)  # 将配对索引堆叠成一个二维数组
            region_mask_id = np.nonzero(region_mask)[0]  # 获取区域内源点云的索引
            region_mask = np.intersect1d(region_mask_id, src_pairing_id, return_indices=True)[2]  # 获取区域内配对点的索引
            region_pairing_ids = pairing_ids[region_mask]  # 筛选出区域内的配对索引

            # 扣出dst范围
            region_mask = (dst_xyz[:, 0] > r[0]) & (dst_xyz[:, 0] < r[1]) & (dst_xyz[:, 1] > r[2]) & (
                        dst_xyz[:, 1] < r[3])
            region_dst_xyz = dst_xyz[region_mask]  # 筛选出区域内的目标点云

            plt.subplot(2, 2, 3)  # 创建第三个子图
            plt.axis('equal')  # 设置子图坐标轴比例相等
            plt.scatter(x=region_pcd[:, 0], y=region_pcd[:, 1], c=region_pcd[:, 2], s=1, alpha=0.2)  # 绘制区域内点云散点图
            plt.scatter(x=region_src_xyz[:, 0], y=region_src_xyz[:, 1], c='red', s=100, alpha=0.5)  # 绘制区域内源点云散点图
            plt.scatter(x=region_dst_xyz[:, 0], y=region_dst_xyz[:, 1], c='blue', s=100, alpha=0.5)  # 绘制区域内目标点云散点图
            plt.plot([src_xyz[region_pairing_ids[:, 0], 0], dst_xyz[region_pairing_ids[:, 1], 0]],
                     [src_xyz[region_pairing_ids[:, 0], 1], dst_xyz[region_pairing_ids[:, 1], 1]],
                     c='#FF8C00', linewidth=5)  # 绘制区域内配对点之间的连接线

            plt.subplot(2, 2, 4)  # 创建第四个子图
            plt.axis('equal')  # 设置子图坐标轴比例相等
            plt.scatter(x=region_pcd[:, 0], y=region_pcd[:, 1], c=region_pcd[:, 2], s=1, alpha=0.2)  # 绘制区域内点云散点图
            plt.scatter(x=region_src_s2d_xyz[:, 0], y=region_src_s2d_xyz[:, 1], c='red', s=100,
                        alpha=0.5)  # 绘制区域内应用了偏移量的源点云散点图
            plt.scatter(x=region_dst_xyz[:, 0], y=region_dst_xyz[:, 1], c='blue', s=100, alpha=0.5)  # 绘制区域内目标点云散点图
            plt.plot([src_s2d_xyz[region_pairing_ids[:, 0], 0], dst_xyz[region_pairing_ids[:, 1], 0]],
                     [src_s2d_xyz[region_pairing_ids[:, 0], 1], dst_xyz[region_pairing_ids[:, 1], 1]],
                     c='#FF8C00', linewidth=5)  # 绘制区域内应用了偏移量的配对点之间的连接线

            plt.show()  # 显示绘制的图形

        '''solve transformation with SVD'''
        # 使用奇异值分解（SVD）求解变换矩阵
        Rs, Ts, inlier_mask_list, inlier_rmse_list = \
            self._solve_transformation_SVD(pairing_conf, src_pairing_coor, dst_pairing_coor)

        if not batch:  # 如果不是批量模式
            Rs = Rs[0]  # 获取第一个（唯一一个）旋转矩阵 (3, 3)
            Ts = Ts[0]  # 获取第一个（唯一一个）平移向量 (3, 1)
            pairing_conf = pairing_conf[0, inlier_mask_list[0]]  # 获取内点的配对置信度 (K,)
            inlier_rmse_list = inlier_rmse_list[0]  # 获取第一个（唯一一个）内点的均方根误差 (float)
        else:  # 如果是批量模式
            Rs = Rs  # 保留所有批次的旋转矩阵 (1, 3, 3)
            Ts = Ts  # 保留所有批次的平移向量 (1, 3, 1)
            pairing_conf = pairing_conf[0, inlier_mask_list[0]].unsqueeze(0)  # 获取所有批次内点的配对置信度 (1, K)
            inlier_rmse_list = inlier_rmse_list  # 保留所有批次内点的均方根误差 (List[float])

        return Rs, Ts, pairing_conf, inlier_rmse_list  # 返回旋转矩阵、平移向量、配对置信度和均方根误差

    def loop_detection_forward(self, src_descriptor: Tensor, dst_descriptor: Tensor,
                               src_padding_mask: Tensor = None, dst_padding_mask: Tensor = None):
        """
        :param src_descriptor: 源 统一描述符
        :param dst_descriptor: 目标 统一描述符
        """

        # 补全批次维度
        if src_descriptor.ndim == 2 and dst_descriptor.ndim == 2:
            src_descriptor = src_descriptor.unsqueeze(0)
            dst_descriptor = dst_descriptor.unsqueeze(0)

        # 统一描述符 --> 相关描述符
        src_corr_descriptor, dst_corr_descriptor = \
            self._descriptor_attention_forward(src_descriptor, dst_descriptor, src_padding_mask, dst_padding_mask)

        # 输入相关描述符、进行回环检测
        src_fea, dst_fea = src_corr_descriptor[:, :-3, :], dst_corr_descriptor[:, :-3, :]
        loop_pro = self.loop_head(src_fea, dst_fea, src_padding_mask, dst_padding_mask)  # 得到回环概率

        return loop_pro  # 返回 回环概率

    def _descriptor_attention_forward(self, src_descriptor: Tensor, dst_descriptor: Tensor,
                                      src_padding_mask: Tensor = None, dst_padding_mask: Tensor = None) \
            -> Tuple[Tensor, Tensor]:
        """ 提取相关描述符 """
        src_fea, src_xyz = src_descriptor[:, :-3, :], src_descriptor[:, -3:, :]  # 拆分S1描述符 = 特征 + 坐标
        dst_fea, dst_xyz = dst_descriptor[:, :-3, :], dst_descriptor[:, -3:, :]  # 拆分S2描述符 = 特征 + 坐标
        src_pos_embedding = self.pos_embedding_layer(src_xyz)  # 为源坐标计算位置嵌入(attention预处理)
        dst_pos_embedding = self.pos_embedding_layer(dst_xyz)  # 为目标坐标计算位置嵌入(attention预处理)
        src_fea = self.projection(src_fea)  # 提取特征 (16, 128, 256) --> (16, 256, 256)
        dst_fea = self.projection(dst_fea)  # 提取特征 (16, 128, 256) --> (16, 256, 256)

        for layer in self.descriptor_attention:  # 通过多个注意力层
            src_fea, dst_fea = layer(src_fea=src_fea, dst_fea=dst_fea,
                                     src_pos_embedding=src_pos_embedding, dst_pos_embedding=dst_pos_embedding,
                                     src_padding_mask=src_padding_mask, dst_padding_mask=dst_padding_mask)
        src_corr_descriptor = torch.cat([src_fea, src_xyz], dim=1)  # 将处理后的S1特征和坐标连接起来 (16, 256+3, 256)
        dst_corr_descriptor = torch.cat([dst_fea, dst_xyz], dim=1)  # 将处理后的S2特征和坐标连接起来 (16, 256+3, 256)
        return src_corr_descriptor, dst_corr_descriptor  # 返回S1，S2相关描述符

    def _descriptor_pairing(self, src_corr_descriptor: Tensor, dst_corr_descriptor: Tensor,
                            num_sample: Union[int, float] = 0.5) -> Tuple[Tensor, Tensor, Tensor]:
        """descriptor匹配"""
        B, _, M = src_corr_descriptor.shape  # 获取源描述符的形状信息，B为批次大小，M为源特征数量
        N, device = dst_corr_descriptor.shape[-1], src_corr_descriptor.device  # 获取目标描述符的特征数量和设备信息
        assert B == 1, 'batch size in inference must be 1'  # 在推理阶段，批次大小必须为1
        if isinstance(num_sample, int):  # 如果采样数是整数
            num_sample = num_sample
        elif isinstance(num_sample, float) and num_sample > 1:  # 如果采样数是大于1的浮点数
            num_sample = int(num_sample)
        elif isinstance(num_sample, float) and 0 < num_sample <= 1:  # 如果采样数是小于等于1的浮点数，按比例确定采样数
            num_sample = int(num_sample * (M + N))
        else:  # 如果采样数的值不合法
            raise ValueError(f'Argument `num_sample` with value {num_sample} is not supported')
        num_sample = num_sample // 2  # 每个配对可以构造两组对应关系

        ''' 从相关描述符中提取相似度特征 '''

        src_pairing_fea = self.similarity_head(src_corr_descriptor[:, :-3, :])  # 计算源描述符的相似度特征
        dst_pairing_fea = self.similarity_head(dst_corr_descriptor[:, :-3, :])  # 计算目标描述符的相似度特征

        '''从特征相似度矩阵(B, M, N)中按照相似度挑选top-k个'''
        # 计算相似度矩阵，并从中挑选出前k个相似度最高的配对
        similarity_matrix = F.normalize(src_pairing_fea.transpose(1, 2), p=2, dim=2) @ F.normalize(dst_pairing_fea, p=2,
                                                                                                   dim=1)
        row_softmax = F.softmax(similarity_matrix / self.tau, dim=2)  # 对相似度矩阵按行进行softmax归一化
        col_softmax = F.softmax(similarity_matrix / self.tau, dim=1)  # 对相似度矩阵按列进行softmax归一化
        similarity_matrix = row_softmax * col_softmax  # 计算最终的相似度矩阵
        similarity_matrix = similarity_matrix.reshape(B, M * N)  # 将相似度矩阵展平成一维
        k_pair_value, k_self_index = torch.topk(similarity_matrix, k=num_sample, dim=1)  # 挑选出相似度最高的前k个配对
        src_pairing_index = k_self_index // N  # 计算源配对的索引
        dst_pairing_index = k_self_index % N  # 计算目标配对的索引

        from utils.global_dict import global_dict  # 从全局字典中导入模块，用于存储配对信息
        global_dict['src_pairing_id'] = src_pairing_index  # 将源配对索引保存到全局字典
        global_dict['dst_pairing_id'] = dst_pairing_index  # 将目标配对索引保存到全局字典

        '''将top-k个配对按对应关系排列后返回'''
        # 将前k个配对的特征按对应关系排列并返回
        src_pairing_index, dst_pairing_index = src_pairing_index.squeeze(0), dst_pairing_index.squeeze(0)  # 去掉批次维度
        src_pairing_descriptor = src_corr_descriptor[:, :, src_pairing_index]  # 获取源配对的描述符 (1, C, K)
        dst_pairing_descriptor = dst_corr_descriptor[:, :, dst_pairing_index]  # 获取目标配对的描述符 (1, C, K)
        pairing_conf = k_pair_value  # 返回配对的置信度 (1, K)

        return src_pairing_descriptor, dst_pairing_descriptor, pairing_conf  # 返回源描述符、目标描述符和配对置信度

    def _get_corres_sets(self, src_pairing_descriptor: Tensor, dst_pairing_descriptor: Tensor, pairing_conf: Tensor) \
            -> Tuple[Tensor, Tensor, Tensor]:
        """获取坐标维度的对应点集"""

        # 计算双向配对的偏移量 (1, 2C-6, K) -> (1, 3, K)
        pairing_fea_s2d = torch.cat([src_pairing_descriptor[:, :-3, :], dst_pairing_descriptor[:, :-3, :]],
                                    dim=1)  # 源配对特征和目标配对特征的连接
        pairing_fea_d2s = torch.cat([dst_pairing_descriptor[:, :-3, :], src_pairing_descriptor[:, :-3, :]],
                                    dim=1)  # 目标配对特征和源配对特征的连接
        pairing_offset_s2d = self.offset_head(pairing_fea_s2d)  # 计算源到目标的偏移量
        pairing_offset_d2s = self.offset_head(pairing_fea_d2s)  # 计算目标到源的偏移量

        # 通过修正的坐标构造双向匹配关系 (1, 3, 2K)
        src_align_coor = src_pairing_descriptor[:, -3:, :] + pairing_offset_s2d  # 对源坐标进行偏移修正
        dst_align_coor = dst_pairing_descriptor[:, -3:, :] + pairing_offset_d2s  # 对目标坐标进行偏移修正
        src_pairing_coor = torch.cat([src_align_coor, src_pairing_descriptor[:, -3:, :]], dim=-1)  # 连接修正后的源坐标和原始源坐标
        dst_pairing_coor = torch.cat([dst_pairing_descriptor[:, -3:, :], dst_align_coor], dim=-1)  # 连接目标坐标和修正后的目标坐标
        pairing_conf = pairing_conf.repeat(1, 2)  # 将配对置信度扩展至双向配对 (1, 2K)

        # 去除较大的offset对应的配对
        outlier_max = self.args.loss.eps_offset ** 2  # 偏移量的最大阈值
        offset_outlier_mask_s2d = torch.sum(torch.square(pairing_offset_s2d), dim=1) <= outlier_max  # 源到目标的偏移量掩码
        offset_outlier_mask_d2s = torch.sum(torch.square(pairing_offset_d2s), dim=1) <= outlier_max  # 目标到源的偏移量掩码
        offset_outlier_mask = torch.cat([offset_outlier_mask_s2d.squeeze(0), offset_outlier_mask_d2s.squeeze(0)],
                                        dim=0)  # 连接两个掩码
        src_pairing_coor = src_pairing_coor[..., offset_outlier_mask]  # 过滤源配对坐标中偏移量超出阈值的点
        dst_pairing_coor = dst_pairing_coor[..., offset_outlier_mask]  # 过滤目标配对坐标中偏移量超出阈值的点
        pairing_conf = pairing_conf[..., offset_outlier_mask]  # 过滤配对置信度

        return src_pairing_coor, dst_pairing_coor, pairing_conf  # 返回修正后的源和目标配对坐标以及置信度

    @staticmethod
    def _solve_transformation_SVD(pairing_conf: Tensor, src_pairing_coor: Tensor, dst_pairing_coor: Tensor,
                                  num_iter: int = 3, std_ratio: float = 3.0) -> Tuple[Tensor, Tensor, List, List]:
        """去中心化点集协方差矩阵SVD分解法计算刚体变换的解析解"""

        R_list, T_list, inlier_mask_list, inlier_rmse_list = [], [], [], []  # 初始化用于保存结果的列表
        for weight, src, dst in zip(pairing_conf.float(), src_pairing_coor.float(), dst_pairing_coor.float()):
            iter_cnt = 0  # 迭代计数器
            inlier_mask = weight > 0.5  # 初始内点掩码，选择权重较大的匹配
            _, ids = torch.topk(weight, k=min(64, len(weight)), dim=0)  # 选择前64个或更少的权重较大的点作为初始内点
            inlier_mask[ids] = True  # 标记这些点为内点

            while True:
                # 去中心化
                src_inner, dst_inner, weight_inner = src[:, inlier_mask], dst[:, inlier_mask], weight[
                    inlier_mask]  # 提取内点的源和目标坐标以及对应的权重
                src_xyz_inner_centroid = (src_inner * weight_inner).sum(dim=1,
                                                                        keepdim=True) / weight_inner.sum()  # 计算源内点的质心
                dst_xyz_inner_centroid = (dst_inner * weight_inner).sum(dim=1,
                                                                        keepdim=True) / weight_inner.sum()  # 计算目标内点的质心
                S = (src_inner - src_xyz_inner_centroid) @ torch.diag(weight_inner) @ (
                            dst_inner - dst_xyz_inner_centroid).T  # 计算协方差矩阵

                # SVD分解
                u, s, v = torch.svd(S.double())  # 对协方差矩阵进行SVD分解
                R = v @ u.T  # 计算旋转矩阵
                T = dst_xyz_inner_centroid.double() - R @ src_xyz_inner_centroid.double()  # 计算平移矩阵
                R, T = R.float(), T.float()  # 转换矩阵为浮点数

                # 抑制异常配对
                err = torch.norm(R @ src + T - dst, p=2, dim=0)  # 计算残差
                inlier_mean, inlier_std = err[inlier_mask].mean(), err[inlier_mask].std()  # 计算内点的均值和标准差
                new_inlier = (err <= (inlier_mean + std_ratio * inlier_std))  # 更新内点掩码

                # 迭代次数达到上限/迭代结果不再改变/配对点数量过少时结束计算
                iter_cnt += 1  # 增加迭代计数
                if iter_cnt >= num_iter or (inlier_mask == new_inlier).all() or new_inlier.sum() < 30:
                    inlier_mask = new_inlier  # 更新内点掩码
                    break
                else:
                    inlier_mask = new_inlier  # 继续迭代

            R_list.append(R)  # 保存旋转矩阵
            T_list.append(T)  # 保存平移矩阵
            inlier_mask_list.append(inlier_mask)  # 保存内点掩码
            inlier_rmse = (R @ src[:, inlier_mask] + T - dst[:, inlier_mask]).pow(2).sum(
                0).mean().sqrt().item()  # 计算内点的RMSE
            inlier_rmse_list.append(inlier_rmse)  # 保存内点的RMSE

        Rs = torch.stack(R_list, dim=0)  # 堆叠所有旋转矩阵
        Ts = torch.stack(T_list, dim=0)  # 堆叠所有平移矩阵
        return Rs, Ts, inlier_mask_list, inlier_rmse_list  # 返回旋转矩阵、平移矩阵、内点掩码和RMSE列表

    def _descriptor_pairing_one2many(self, src_corr_descriptor: Tensor, dst_corr_descriptor: Tensor,
                                     num_sample: Union[int, float] = 0.5) -> Tuple[Tensor, Tensor, Tensor]:
        """descriptor匹配"""

        B, _, M = src_corr_descriptor.shape  # 获取源描述符的形状信息，B为批次大小，M为源特征数量
        N, device = dst_corr_descriptor.shape[-1], src_corr_descriptor.device  # 获取目标描述符的特征数量和设备信息
        assert B == 1, 'batch size in inference must be 1'  # 在推理阶段，批次大小必须为1
        if isinstance(num_sample, int):  # 如果采样数是整数
            num_sample = num_sample
        elif isinstance(num_sample, float) and num_sample > 1:  # 如果采样数是大于1的浮点数
            num_sample = int(num_sample)
        elif isinstance(num_sample, float) and 0 < num_sample <= 1:  # 如果采样数是小于等于1的浮点数，按比例确定采样数
            num_sample = int(num_sample * (M + N))
        else:  # 如果采样数的值不合法
            raise ValueError(f'Argument `num_sample` with value {num_sample} is not supported')
        num_sample = num_sample // 2  # 每个配对可以构造两组对应关系

        '''correlated descriptors -> similarity feature'''
        src_pairing_fea = self.similarity_head(src_corr_descriptor[:, :-3, :])  # 计算源描述符的相似度特征
        dst_pairing_fea = self.similarity_head(dst_corr_descriptor[:, :-3, :])  # 计算目标描述符的相似度特征

        '''从特征相似度矩阵(B, M, N)中按照相似度挑选top-k个'''
        similarity_matrix = F.normalize(src_pairing_fea.transpose(1, 2), p=2, dim=2) @ F.normalize(dst_pairing_fea, p=2,
                                                                                                   dim=1)  # 计算相似度矩阵
        row_softmax = F.softmax(similarity_matrix / self.tau, dim=2)  # 对相似度矩阵按行进行softmax归一化
        col_softmax = F.softmax(similarity_matrix / self.tau, dim=1)  # 对相似度矩阵按列进行softmax归一化
        similarity_matrix = row_softmax * col_softmax  # 计算最终的相似度矩阵
        k_pair_value, k_self_index = torch.topk(similarity_matrix.reshape(B, M * N), k=num_sample,
                                                dim=1)  # 挑选出相似度最高的前k个配对
        src_pairing_index = k_self_index // N  # 计算源配对的索引
        dst_pairing_index = k_self_index % N  # 计算目标配对的索引

        '''S2M和M2M时，一个descriptor周围可能存在多个配准点，使用one-to-many策略拓展配对'''
        judge_matrix = torch.zeros_like(similarity_matrix, dtype=torch.bool, device=device)  # 初始化判断矩阵
        judge_matrix[0, src_pairing_index[0], dst_pairing_index[0]] = True  # 将最初的配对标记为有效配对

        import math
        if N > 256:  # 如果目标特征数量大于256，进行扩展配对
            expand = max(round(math.sqrt(N / 256)), 2)  # 计算扩展数量
            candidates_conf, candidates_ids = torch.topk(similarity_matrix[0, src_pairing_index[0]], k=expand,
                                                         dim=-1)  # 获取相似度最高的候选点
            src_expand_ids = src_pairing_index[0].unsqueeze(1).repeat(1, expand)  # 扩展源配对索引
            conf_mask = candidates_conf > 0.01  # 设置最低置信度阈值
            src_expand_ids = src_expand_ids[conf_mask]  # 过滤掉低置信度的候选点
            candidates_ids = candidates_ids[conf_mask]  # 过滤掉低置信度的候选点
            judge_matrix[0, src_expand_ids, candidates_ids] = True  # 将扩展配对标记为有效配对

        if M > 256:  # 如果源特征数量大于256，进行扩展配对
            expand = max(round(math.sqrt(M / 256)), 2)  # 计算扩展数量
            candidates_conf, candidates_ids = torch.topk(similarity_matrix[0, :, dst_pairing_index[0]].T, k=expand,
                                                         dim=-1)  # 获取相似度最高的候选点
            dst_expand_ids = dst_pairing_index[0].unsqueeze(1).repeat(1, expand)  # 扩展目标配对索引
            conf_mask = candidates_conf > 0.01  # 设置最低置信度阈值
            dst_expand_ids = dst_expand_ids[conf_mask]  # 过滤掉低置信度的候选点
            candidates_ids = candidates_ids[conf_mask]  # 过滤掉低置信度的候选点
            judge_matrix[0, candidates_ids, dst_expand_ids] = True  # 将扩展配对标记为有效配对

        '''将top-k个配对按对应关系排列后返回'''
        src_pairing_index, dst_pairing_index = torch.nonzero(judge_matrix[0]).T  # 获取有效配对的源和目标索引
        pairing_conf, ids = similarity_matrix[0][judge_matrix[0]].sort(descending=True)  # 按相似度排序配对置信度
        pairing_conf = pairing_conf.unsqueeze(0)  # 添加批次维度
        src_pairing_index, dst_pairing_index = src_pairing_index[ids], dst_pairing_index[ids]  # 根据排序结果重新排列配对索引
        src_pairing_descriptor = src_corr_descriptor[:, :, src_pairing_index]  # 获取最终的源配对描述符 (1, C, K)
        dst_pairing_descriptor = dst_corr_descriptor[:, :, dst_pairing_index]  # 获取最终的目标配对描述符 (1, C, K)

        return src_pairing_descriptor, dst_pairing_descriptor, pairing_conf  # 返回源描述符、目标描述符和配对置信度

