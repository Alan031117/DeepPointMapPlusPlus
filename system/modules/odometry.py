import colorlog as logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Union
import time

from system.modules.utils import PoseTool, calculate_information_matrix_from_pcd

from system.modules.pose_graph import PoseGraph, ScanPack, PoseGraph_Edge
from system.modules.utils import simvec_to_num
from network.encoder.label_decoder import Label_Decoder


class ExtractionThread:
    def __init__(self, args, system_info, posegraph_map: PoseGraph, dpm_encoder: nn.Module, label_decoder: nn.Module, device='cpu') -> None:
        """
        初始化 ExtractionThread 类，负责管理和执行DPM编码器的线程，用于SLAM系统中的特征提取任务。

        Args:
            args (EasyDict): 包含各种超参数和配置选项的字典结构（通常用于存储模型、推理、训练等相关参数）。
            system_info (EasyDict): 包含系统信息的字典结构，通常用于记录系统状态或标识符等。
            posegraph_map (PoseGraph): SLAM系统的位姿图，存储点云扫描、位姿和里程计边缘的相关数据。
            dpm_encoder (nn.Module): 深度点云模型 (DPM) 的编码器模型，用于对输入的点云数据进行特征提取。
            device (str, optional): 推理使用的设备，默认为 'cpu'，可以选择 'cuda' 以使用GPU加速推理。
        """
        self.args = args
        self.system_info = system_info
        self.device = device

        # 初始化并加载编码器模型到指定设备上，并设置为推理模式（eval），因为这是用于推理任务
        self.encoder_model = dpm_encoder.to(self.device).eval()
        self.label_decoder = label_decoder.to(self.device).eval()

        # 保存 SLAM 系统的位姿图（posegraph_map），用于后续的点云处理和特征提取
        self.posegraph_map = posegraph_map

    def process(self, point_cloud: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # 开始计时，用于记录执行时间
        start_t = time.perf_counter()

        # 使用 torch.no_grad() 关闭梯度计算，这通常在推理阶段（inference）使用以节省内存和提高速度
        with torch.no_grad():
            # 调用 encoder_model 方法对输入的点云数据进行编码
            descriptors_xyz, descriptors_fea, mask, index = self.encoder_model(point_cloud.to(self.device),
                                                                               padding_mask.to(self.device))

            # 调用 label_inference 更新 mask
            mask = self.label_decoder.label_inference(points_coor=descriptors_xyz, points_fea=descriptors_fea, mask=mask)

            retain_mask = ~mask  # True 表示保留，False 表示滤除

            descriptors_xyz = descriptors_xyz[:, :, retain_mask.squeeze()]
            descriptors_fea = descriptors_fea[:, :, retain_mask.squeeze()]

        net_t = time.perf_counter()

        descriptors_fea = descriptors_fea
        descriptors_xyz = descriptors_xyz * self.args.coor_scale    # 坐标缩放

        # 将特征描述符和缩放后的坐标描述符沿第二维度 (特征维度) 拼接，形成最终的描述符，维度变为 [B, 128 + 3, N]
        descriptors = torch.cat([descriptors_fea, descriptors_xyz], dim=1)  # [B, 128 + 3, N]

        end_t = time.perf_counter()

        logger.info(
            f'Agent{self.system_info.agent_id}: Extract log: net = {net_t - start_t:.4f}s, cat = {end_t - net_t:.4f}s')
        logger.info(
            f'Agent{self.system_info.agent_id}: Extract done, point = {point_cloud.shape[-1]} descriptor = {descriptors.shape[-1]}')

        # 返回最终的描述符，维度为 [B, 128+3, N]
        return descriptors  # B, 128+3, N


class OdometryThread():
    def __init__(self, args, system_info, posegraph_map: PoseGraph, dpm_decoder: nn.Module, device='cpu') -> None:
        """

        Args:
            args (EasyDict):
                odometer_candidates_num (int): odometer candidates number 
                registration_sample_odometer (float | int): Registration sample number / ratio
            system_info (EasyDict):
            posegraph_map (PoseGraph): PoseGraph of SLAM system
            dpm_decoder (nn.Module): DPM Decoder Model
            device (str, optional): Inference Device. Defaults to 'cpu'.
        """
        self.args = args
        self.system_info = system_info
        self.posegraph_map = posegraph_map
        self.device = device
        self.decoder_model = dpm_decoder.to(self.device).eval()

    def search_candidates(self, new_scan: ScanPack) -> List[ScanPack]:
        # 如果位姿图中没有任何扫描，或者当前扫描的agent_id不在位姿图的已有扫描中，
        # 或者位姿图中没有已知的关键帧或任意帧，则返回空列表，表示这是位姿图中的第一个扫描
        if (len(self.posegraph_map.get_all_scans()) == 0 or new_scan.agent_id not in [s.agent_id for s in
                                                                                      self.posegraph_map.get_all_scans()] or self.posegraph_map.last_known_keyframe is None
                or self.posegraph_map.last_known_anyframe is None):
            return []  # 位姿图中的第一个扫描

        # 获取最后一个已知的关键帧
        last_scan = self.posegraph_map.get_scanpack(self.posegraph_map.last_known_keyframe)

        # 获取最后一个已知帧的预测位姿 (SE3)，用于后续与候选帧的位姿比较
        last_SE3 = self.posegraph_map.get_scanpack(self.posegraph_map.last_known_anyframe).SE3_pred

        # 确保最后已知的位姿不为 None
        assert last_SE3 is not None
        # 确保当前扫描与最后扫描不是同一个，如果是同一个扫描，则提示应先调用 `_add_odometry()` 函数处理
        assert last_scan is not new_scan, f"last_scan is new_scan, call _add_odometry() BEFORE self.posegraph_map.add_scan()"

        # 从位姿图中搜索与最后一个已知关键帧（last_scan）邻近的候选关键帧。
        # 过滤条件：候选帧必须是关键帧（即不是 'non-keyframe'），且 agent_id 必须与当前扫描一致
        key_frames = list(
            filter(lambda s: (s.type != 'non-keyframe' and s.agent_id == new_scan.agent_id),
                   # 搜索邻近的帧，neighbor_level=5 指定邻居深度，edge_type 限定搜索的边为里程计边和闭环边
                   self.posegraph_map.graph_search(token=last_scan.token, neighbor_level=5, coor_sys=last_scan.coor_sys,
                                                   edge_type=['odom', 'loop'])))

        # 确保所有候选帧都具有有效的预测位姿 (SE3_pred)
        assert set([s.SE3_pred is not None for s in key_frames]) == {True}

        # 计算每个候选关键帧与最后已知帧的距离。具体方法是：取每个关键帧的位姿向量（位置部分 SE3[:3, 3:]），
        # 并与最后已知帧的位姿进行 L2 范数的距离计算
        key_frame_distances = torch.norm(
            torch.stack([s.SE3_pred[:3, 3:] for s in key_frames], dim=0) - last_SE3[:3, 3:], p=2, dim=1)  # (N, 1)

        # 选择距离最近的前 k 个关键帧，k 的值为 odometer_candidates_num 和现有关键帧数的最小值
        topk_key_frame_dist, topk_keyframe_index = torch.topk(key_frame_distances.float(), dim=0,
                                                              k=min(len(key_frames), self.args.odometer_candidates_num),
                                                              largest=False)

        # 将索引转换为列表形式，获取最接近的关键帧
        topk_keyframe_index: List[int] = topk_keyframe_index.flatten().tolist()
        key_frames = [key_frames[i] for i in topk_keyframe_index]

        # 如果最近的关键帧距离超过 20 米，记录警告日志，提示关键帧可能距离过远
        if (key_frame_distances.min() > 20):
            logger.warning(f'The nearest key-frame seems too far ({key_frame_distances.min():.3f}m)')

        # 返回选取的候选关键帧列表
        return key_frames

    def odometry(self, new_scan: ScanPack, candidates: List[ScanPack]) -> List[PoseGraph_Edge]:
        # 初始化一个空列表，用于存储生成的位姿图边缘（PoseGraph_Edge）
        edges: List[PoseGraph_Edge] = []

        # 遍历每一个候选扫描（nearest_scan），对新扫描与每个候选扫描执行里程计计算
        for nearest_scan in candidates:
            # 注册阶段：将候选扫描和新扫描的SE3位姿、点云数据写入全局字典中，用于后续的对齐和位姿估计
            from utils.global_dict import global_dict
            global_dict['src_SE3'] = nearest_scan.SE3_gt  # 候选扫描的真实位姿
            global_dict['dst_SE3'] = new_scan.SE3_gt  # 新扫描的真实位姿
            global_dict['src_pcd'] = nearest_scan.full_pcd  # 候选扫描的完整点云数据
            global_dict['dst_pcd'] = new_scan.full_pcd  # 新扫描的完整点云数据

            # 关闭梯度计算（torch.no_grad()），因为这是推理阶段，不需要计算梯度
            with torch.no_grad():
                # 调用解码器模型的 `registration_forward` 函数，基于两个扫描的关键点进行位姿估计
                # 输出旋转矩阵 (rot_pred)、平移矩阵 (trans_pred)、相似度向量 (sim_topks) 和均方根误差 (rmse)
                rot_pred, trans_pred, sim_topks, rmse = self.decoder_model.registration_forward(
                    nearest_scan.key_points.to(self.device),  # 候选扫描的关键点
                    new_scan.key_points.to(self.device),  # 新扫描的关键点
                    num_sample=self.args.registration_sample_odometer  # 使用的采样点数目
                )

            # 将预测的旋转和平移矩阵组合为 SE3（即位姿矩阵）
            SE3 = PoseTool.SE3(rot_pred, trans_pred)

            # 计算信息矩阵（Information Matrix），用于衡量两个扫描之间的位姿变换的置信度
            # 使用完整点云数据来计算信息矩阵，通过对齐两个点云数据并结合位姿估计（SE3）
            information_mat = calculate_information_matrix_from_pcd(
                nearest_scan.full_pcd[:3, :],  # 候选扫描的点云数据（前3列是位置信息）
                new_scan.full_pcd[:3, :],  # 新扫描的点云数据
                SE3,  # 预测的位姿矩阵
                device=self.device  # 计算设备（如GPU）
            )


            # 创建一个新的位姿图边缘（PoseGraph_Edge），表示新扫描与候选扫描之间的位姿变换关系
            edge = PoseGraph_Edge(
                src_scan_token=nearest_scan.token,  # 候选扫描的标识符
                dst_scan_token=new_scan.token,  # 新扫描的标识符
                SE3=SE3.inverse(),  # 预测的位姿矩阵（逆矩阵，用于从新扫描到候选扫描的变换）
                information_mat=information_mat,  # 信息矩阵，表示位姿估计的置信度
                type='odom',  # 边缘类型（'odom' 表示里程计边缘）
                confidence=simvec_to_num(sim_topks),  # 使用相似度向量计算出的置信度
                rmse=rmse  # 里程计估计中的均方根误差
            )

            # 将生成的边缘添加到 `edges` 列表中
            edges.append(edge)

        # 返回包含所有边缘的列表
        return edges

    def process(self, new_scan: ScanPack) -> List[PoseGraph_Edge]:

        start_t = time.perf_counter()

        # 搜索候选帧，找到可能与当前帧匹配的先前扫描帧
        # `search_candidates` 函数用于在已有的位姿图中寻找与新扫描相关联的候选帧
        candidates = self.search_candidates(new_scan=new_scan)

        mid_t = time.perf_counter()

        # 里程计处理：基于新扫描和候选帧，计算相对位姿关系，生成里程计边缘
        # `odometry` 函数用于根据当前帧和候选帧之间的几何关系计算相对位姿，返回边缘列表
        edges = self.odometry(new_scan=new_scan, candidates=candidates)

        end_t = time.perf_counter()

        logger.info(
            f'Agent{self.system_info.agent_id}: Odometry log: search_candidates = {(mid_t - start_t) * 1000:.4f}ms, odometry = {(end_t - mid_t) * 1000:.4f}ms')

        # 返回计算得到的边缘列表，代表当前扫描与候选帧之间的位姿关系
        return edges
