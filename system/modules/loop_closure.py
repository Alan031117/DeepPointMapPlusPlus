import colorlog as logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from typing import Any, List, Dict, Literal, Tuple, Union

from system.modules.utils import PoseTool, calculate_information_matrix_from_pcd
from system.modules.pose_graph import PoseGraph, ScanPack, PoseGraph_Edge
from system.modules.utils import simvec_to_num


class LoopThread():
    TRANS_STD = 0.4
    ROT_STD = 0.5

    def __init__(self, args, system_info, posegraph_map: PoseGraph, dpm_decoder: nn.Module, device='cpu') -> None:
        """_summary_

        Args:
            args (EasyDict):
                enable_loop_closure (bool): Enable loop-closure (contains detection and optim)
                loop_detection_gap (int): The minium gap (#keyframe) between 2 loop-closure trials
                loop_detection_trust_range (int): The range (on graph) of trusted region. All scan within will not be considered as long-term loop-closure, and more strict loop-detection are applied
                loop_detection_gnss_distance (float): The maximum searching range based on GNSS position
                loop_detection_pred_distance (float): The maximum searching range based on Predicted position
                loop_detection_rotation_min (float): The minium rotation angle between two frame. Frame in trust-region and delta-angle < this threshold will not considered as loop-closure
                loop_detection_prob_acpt_threshold (float): Accept the loop-closure proposal if the loop-confidence is > than this threshold
                loop_detection_candidates_num (int): How many loop-edge may found during one loop-check
                registration_sample_loop (float | int): Registration sample number / ratio
                loop_detection_confidence_acpt_threshold (float): Edge confidence threshold, loop-edges with confidence < than this value will be dropped

                enable_global_optimization (bool): Enable backend optimization (only affect optim)
                global_optimization_gap: The minium gap (#keyframe) between 2 optimization
            system_info (EasyDict):
                agent_id (int): agent id
            posegraph_map (PoseGraph): PoseGraph of SLAM system
            dpm_decoder (nn.Module): DPM Decoder Model
            device (str, optional): Inference Device. Defaults to 'cpu'.
        """
        self.args = args
        self.system_info = system_info
        self.posegraph_map = posegraph_map
        self.device = device
        self.decoder_model = dpm_decoder.to(self.device).eval()
        self.last_loop_pose_num = -self.args.loop_detection_gap - 1
        self.last_optim_pose_num = -self.args.global_optimization_gap - 1
        self.last_loop_token = -1
        self.required_optim = False

    def process(self, new_scan: ScanPack, targets: Literal['self', 'others', 'all'] = 'all'):
        assert self.posegraph_map.has_scan(new_scan.token)
        if not self.args.enable_loop_closure:
            return []
        elif self.posegraph_map.key_frame_num - self.last_loop_pose_num <= self.args.loop_detection_gap:
            return []
        else:
            start_t = time.perf_counter()
            candidates_scans = self.loop_closure_detection(new_scan=new_scan, targets=targets)
            candidates_edges = self.loop_closure_registration(new_scan=new_scan, scan_list=candidates_scans)
            validated_edges = self.loop_closure_verification(edge_list=candidates_edges)
            if len(validated_edges) > 0:
                self.required_optim = True
                for edge in validated_edges:
                    self.posegraph_map.add_edge(edge)
                self.last_loop_pose_num = self.posegraph_map.key_frame_num
                self.last_loop_token = new_scan.token
                optim_result = self.global_optimization(forced=False)
                end_t = time.perf_counter()

                if (targets == 'all' or targets == 'others'):
                    self.posegraph_map.repair_coor_sys()

                # logging
                logger.info(f'Agent{self.system_info.agent_id}: Loop closure log: {end_t - start_t:.4f}s')
                logger.info(f'Agent{self.system_info.agent_id}: |- found {len(candidates_scans)} candidate scans')
                logger.info(f'Agent{self.system_info.agent_id}: |- regised {len(candidates_edges)} candidate scans')
                logger.info(f'Agent{self.system_info.agent_id}: |- verified {len(validated_edges)} candidate scans')
                if not optim_result:
                    logger.info(f'Agent{self.system_info.agent_id}: |- optimization not applied')
                else:
                    n, m, diff = optim_result
                    logger.info(f'Agent{self.system_info.agent_id}: |- optimization G = <{n}, {m}>, diff = {diff:.3f}')
                logger.info(f'Agent{self.system_info.agent_id}: Loop closure done, found {len(validated_edges)} edges')

            return validated_edges

    def loop_closure_detection(self, new_scan: ScanPack, targets: Literal['self', 'others', 'all'] = 'all') -> List[
        ScanPack]:

        """
        检测回环闭合，用于SLAM系统中的位置更新。

        参数:
            new_scan (ScanPack): 当前扫描的数据包。
            targets (str): 检测目标，'self' 表示仅检测当前agent，'others' 表示其他agents，'all' 表示所有agents。

        返回:
            List[ScanPack]: 返回可能的回环候选扫描列表。
        """

        # 1. 搜索范围：首先筛选出所有含有关键点的完整扫描作为候选扫描
        scan_candidates = filter(lambda s: s.key_points is not None and s.type == 'full',
                                 self.posegraph_map.get_all_scans())

        # 2. 根据 targets 参数进一步筛选候选扫描
        if targets == 'all':
            scan_candidates = scan_candidates  # 如果是 'all'，则使用所有候选扫描
        elif targets == 'self':
            scan_candidates = filter(lambda s: s.agent_id == new_scan.agent_id, scan_candidates)  # 仅筛选相同agent
        elif targets == 'others':
            scan_candidates = filter(lambda s: s.agent_id != new_scan.agent_id, scan_candidates)  # 仅筛选不同agent
        else:
            # 若 targets 参数不在预期范围，抛出错误
            raise RuntimeError(f'add_loop_closure received an unknown arg value: targets = {targets}')

        # 将过滤后的候选扫描转换为列表
        scan_candidates = list(scan_candidates)

        # 如果没有找到合适的候选扫描，返回空列表
        if len(scan_candidates) == 0:
            return []

        # 3. 计算信任区域，避免距离过近的回环：
        # trust_zone_range1: 临近扫描的信任区域1 (较小范围)
        trust_zone_range1 = [
            s.token
            for s in
            self.posegraph_map.graph_search(new_scan.token, neighbor_level=self.args.loop_detection_trust_range - 1,
                                            coor_sys=new_scan.coor_sys, edge_type=['odom', 'loop'], max_k=None)
        ]

        # trust_zone_range2: 临近扫描的信任区域2 (较大范围)
        trust_zone_range2 = [
            s.token
            for s in self.posegraph_map.graph_search(new_scan.token,
                                                     neighbor_level=int(self.args.loop_detection_trust_range * 10),
                                                     coor_sys=new_scan.coor_sys, edge_type=['odom', 'loop'], max_k=None)
        ]

        # 4. 使用GPS和里程计预测进行距离过滤：
        mask = torch.ones(len(scan_candidates), dtype=torch.bool)  # 初始化为全True的有效掩码

        # 如果启用了基于GNSS（GPS）的距离过滤，计算每个候选扫描与新扫描的欧氏距离，并过滤掉超出设定距离的扫描
        if self.args.loop_detection_gnss_distance > 0:
            offsets = torch.stack([s.gps_position - new_scan.gps_position for s in scan_candidates], dim=0)[:, :2,
                      :]  # 计算偏移量 (N, 2, 1)
            distance = torch.norm(offsets, p=2, dim=1).squeeze(-1)  # 计算欧氏距离 (N, )
            mask = mask & (distance <= self.args.loop_detection_gnss_distance)  # 仅保留距离符合要求的扫描 (N, )

        # 如果启用了基于预测的距离过滤，计算里程计预测的位移并进行过滤
        if self.args.loop_detection_pred_distance > 0:
            offsets = torch.stack([(s.SE3_pred - new_scan.SE3_pred)[:2, 3:] for s in scan_candidates],
                                  dim=0)  # 计算位移 (N, 2, 1)
            distance = torch.norm(offsets, p=2, dim=1).squeeze(-1)  # 计算欧氏距离 (N, )
            not_same_coorsys = torch.tensor(
                [(s.coor_sys != new_scan.coor_sys) for s in scan_candidates])  # 判断坐标系统是否相同 (N, )
            mask = mask & ((
                                   distance <= self.args.loop_detection_pred_distance) | not_same_coorsys)  # 保留距离符合要求或坐标系统不同的扫描 (N, )

        # 过滤掉掩码为 False 的候选扫描
        scan_candidates = [i for i, m in zip(scan_candidates, mask) if m]
        if len(scan_candidates) == 0:
            return []

        # 5. 对每个候选扫描进行进一步验证
        valid_scans = [True] * len(scan_candidates)
        for i, prev_scan in enumerate(scan_candidates):

            # 5.1. 如果候选扫描在信任区域1内或与新扫描相同，则标记为无效
            if (prev_scan.token in trust_zone_range1) or (prev_scan is new_scan):
                valid_scans[i] = False
                continue

            # 5.2. 如果候选扫描与新扫描属于同一个agent且在信任区域2内，计算旋转角度和位移
            if (prev_scan.agent_id == new_scan.agent_id) and (prev_scan.token in trust_zone_range2):
                delta_R, delta_T = PoseTool.Rt(
                    torch.linalg.inv(prev_scan.SE3_pred) @ new_scan.SE3_pred)  # 计算旋转矩阵和位移矩阵差异
                delta_R_rad = PoseTool.rotation_angle(delta_R)  # 计算旋转角度

                # 如果旋转角度和欧式距离变化不满足设定阈值，标记该候选扫描为无效
                if (delta_R_rad * 180 / torch.pi < self.args.loop_detection_rotation_min) \
                        or (torch.norm(delta_T) < self.args.loop_detection_translation_min):
                    valid_scans[i] = False
                    continue

                # 检查距离上一个回环点的欧氏距离是否满足要求
                if self.last_loop_token != -1:
                    last_loop_scan_SE3 = self.posegraph_map.get_scanpack(self.last_loop_token).SE3_pred
                    _, gap_T = PoseTool.Rt(torch.linalg.inv(last_loop_scan_SE3) @ new_scan.SE3_pred)
                    gap_T_euc = torch.norm(gap_T)
                    if gap_T_euc < self.args.loop_detection_transaction_gap:
                        valid_scans[i] = False
                        continue

        # 过滤掉无效的扫描
        scan_candidates = [i for i, m in zip(scan_candidates, valid_scans) if m]
        if len(scan_candidates) == 0:
            return []

        # 6. 对每个候选扫描单独计算回环置信度
        src_keypoints_list = [s.key_points for s in scan_candidates]  # [(fea+xyz, N)]
        M, C = len(src_keypoints_list), src_keypoints_list[0].shape[0]
        num_keypoints_batch = torch.tensor([i.shape[-1] for i in src_keypoints_list])  # (M,)
        num_keypoints_max = num_keypoints_batch.max()

        # 构造批量化输入，因为src的desc数量不一致，需padding
        src_keypoints_batch = src_keypoints_list[0].new_zeros(size=(M, C, num_keypoints_max))
        for i, src_keypoints in enumerate(src_keypoints_list):
            src_keypoints_batch[i, :, :src_keypoints.shape[-1]] = src_keypoints
        src_padding_mask = (torch.arange(num_keypoints_max)[None, :] >= num_keypoints_batch[:, None])
        dst_keypoints_batch = new_scan.key_points.unsqueeze(0).repeat(len(scan_candidates), 1, 1)  # M, fea+xyz, N

        with torch.no_grad():
            loop_confidences = self.decoder_model.loop_detection_forward(
                src_keypoints_batch.to(self.device),
                dst_keypoints_batch.to(self.device),
                src_padding_mask=src_padding_mask.to(self.device)
            )

        # 7. 选取置信度最高的 top-k 候选扫描
        topk_confidences, indexes = torch.topk(
            loop_confidences,
            k=min(self.args.loop_detection_candidates_num, len(loop_confidences))
        )
        topk_candidates = [scan_candidates[i] for i in indexes]

        # 8. 过滤掉置信度较低的扫描
        valid_mask = topk_confidences > self.args.loop_detection_prob_acpt_threshold
        topk_candidates = [i for i, m in zip(topk_candidates, valid_mask) if m]

        # 返回筛选后的回环候选扫描
        return topk_candidates

    def loop_closure_registration(self, new_scan: ScanPack, scan_list: List[ScanPack]) -> List[PoseGraph_Edge]:
        # Make Registration
        edges: List[PoseGraph_Edge] = []
        for prev_scan in scan_list:
            # Map query
            prev_scan_map, prev_scan_token = self.posegraph_map.global_map_query_graph(token=prev_scan.token,
                                                                                       neighbor_level=5,
                                                                                       coor_sys=prev_scan.coor_sys,
                                                                                       full_pcd=False,
                                                                                       centering_SE3=prev_scan.SE3_pred,
                                                                                       max_dist=20)
            new_scan_map, new_scan_token = self.posegraph_map.global_map_query_graph(token=new_scan.token,
                                                                                     neighbor_level=5,
                                                                                     coor_sys=new_scan.coor_sys,
                                                                                     full_pcd=False,
                                                                                     centering_SE3=new_scan.SE3_pred,
                                                                                     max_dist=20)

            # Filter overlapped key-points
            src_t = PoseTool.Rt(prev_scan.SE3_pred)[1]
            dst_t = PoseTool.Rt(new_scan.SE3_pred)[1]
            prev_scan_neighbor_token = prev_scan_token.unique().tolist()
            new_scan_neighbor_token = new_scan_token.unique().tolist()
            overlap_scan_token = list(set(prev_scan_neighbor_token) & set(new_scan_neighbor_token))
            if len(overlap_scan_token) > 0:
                overlap_scan_t = torch.cat([PoseTool.Rt(self.posegraph_map.get_scanpack(_scan_token).SE3_pred)[1]
                                            for _scan_token in overlap_scan_token], dim=1)  # (3, N)
                dist_to_src = torch.norm(overlap_scan_t - src_t, p=2, dim=0)  # (N,)
                dist_to_dst = torch.norm(overlap_scan_t - dst_t, p=2, dim=0)  # (N,)
                overlap2prev_scan_mask = dist_to_src < dist_to_dst  # 分配给prev的scan
                overlap2new_scan_mask = ~overlap2prev_scan_mask  # 分配给new的scan
                overlap2prev_scan_token = \
                    [token for token, is_reserve in zip(overlap_scan_token, overlap2prev_scan_mask) if is_reserve]
                overlap2new_scan_token = \
                    [token for token, is_reserve in zip(overlap_scan_token, overlap2new_scan_mask) if is_reserve]
                prev_scan_neighbor_token = list(set(prev_scan_neighbor_token) - set(overlap2new_scan_token))
                new_scan_neighbor_token = list(set(new_scan_neighbor_token) - set(overlap2prev_scan_token))

                # final map
                prev_scan_neighbor_token = torch.tensor(prev_scan_neighbor_token, device=prev_scan_token.device)
                new_scan_neighbor_token = torch.tensor(new_scan_neighbor_token, device=new_scan_token.device)
                prev_scan_token_mask = (prev_scan_token.unsqueeze(0).repeat(prev_scan_neighbor_token.shape[0], 1)
                                        == prev_scan_neighbor_token.unsqueeze(1)).sum(0).bool()
                new_scan_token_mask = (new_scan_token.unsqueeze(0).repeat(new_scan_neighbor_token.shape[0], 1)
                                       == new_scan_neighbor_token.unsqueeze(1)).sum(0).bool()
                prev_scan_map = prev_scan_map[:, prev_scan_token_mask]
                prev_scan_token = prev_scan_token[prev_scan_token_mask]
                new_scan_map = new_scan_map[:, new_scan_token_mask]
                new_scan_token = new_scan_token[new_scan_token_mask]

            assert (len(set(prev_scan_token.unique().tolist()) & set(new_scan_token.unique().tolist())) == 0)
            assert prev_scan_map.shape[-1] > 0 and new_scan_map.shape[-1] > 0
            # Registration
            with torch.no_grad():
                rot_pred, trans_pred, sim_topks, rmse = \
                    self.decoder_model.registration_forward(prev_scan_map.to(self.device),
                                                            new_scan_map.to(self.device),
                                                            num_sample=self.args.registration_sample_loop)
            SE3 = PoseTool.SE3(rot_pred, trans_pred)

            # Calculate InfoMat
            # __DEBUG__ infomat using fullpcd
            information_mat = calculate_information_matrix_from_pcd(prev_scan.full_pcd[:3, :], new_scan.full_pcd[:3, :],
                                                                    SE3, device=self.device)
            # information_mat = calculate_information_matrix_from_confidence(confidence=simvec_to_num(sim_topks), rmse=rmse)
            # Make edges
            edge = PoseGraph_Edge(src_scan_token=prev_scan.token,
                                  dst_scan_token=new_scan.token,
                                  SE3=SE3.inverse(),
                                  information_mat=information_mat,
                                  type='loop',
                                  confidence=simvec_to_num(sim_topks),
                                  rmse=rmse)
            edges.append(edge)
        return edges

    def loop_closure_verification(self, edge_list: List[PoseGraph_Edge]):
        vaild_mask = [True] * len(edge_list)
        for i, edge in enumerate(edge_list):
            # Loop edge confidence (map to map confidence)
            if edge.confidence < self.args.loop_detection_confidence_acpt_threshold:
                vaild_mask[i] = False
                continue

            # distance on graph, if > 5000 step, no further check
            dist_graph = self.posegraph_map.shortest_path_length(edge.src_scan_token, edge.dst_scan_token,
                                                                 edge_type=['odom', 'loop'], infinity_length=5000)
            if (dist_graph >= 5000):
                continue

            # distance on graph, if < 5000 step, check the pose difference between loop-edge and step-by-step edge
            src_scan = self.posegraph_map.get_scanpack(edge.src_scan_token)
            dst_scan = self.posegraph_map.get_scanpack(edge.dst_scan_token)
            src_loop_pose = src_scan.SE3_pred @ edge.SE3  # src经过回环边变换后的位姿
            delta_pose = torch.linalg.inv(src_loop_pose) @ dst_scan.SE3_pred
            delta_R, delta_T = PoseTool.Rt(delta_pose)

            # calculate delta_R / STD_R and delta_T / STD_T
            factor_T = torch.norm(delta_T).item() / (self.TRANS_STD * sqrt(dist_graph))
            if factor_T > 3 and dist_graph < 100:
                vaild_mask[i] = False
                continue

            factor_R = PoseTool.rotation_angle(delta_R) * 180 / torch.pi / (self.ROT_STD * sqrt(dist_graph))
            if factor_R > 3:
                vaild_mask[i] = False
                continue

        validated_edges = [i for i, m in zip(edge_list, vaild_mask) if m]
        return validated_edges

    def global_optimization(self, forced=False):
        if (self.args.enable_loop_closure == False):
            return False
        elif (not forced and self.args.enable_global_optimization == False):
            return False
        elif (
                not forced and self.posegraph_map.key_frame_num - self.last_optim_pose_num < self.args.global_optimization_gap):
            return False
        elif (not forced and self.required_optim == False):
            return False
        else:
            n, m, diff = self.posegraph_map.optim()
            self.last_optim_pose_num = self.posegraph_map.key_frame_num
            self.required_optim = False
            return n, m, diff
