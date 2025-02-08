import colorlog as logging
import numpy as np
from copy import deepcopy

# 设置日志系统，基础配置设定为显示 INFO 级别的日志信息
logging.basicConfig(level=logging.INFO)

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 将日志记录器的级别设置为 INFO
logger.setLevel(logging.INFO)

import math
import torch
import open3d as o3d  # 用于点云处理的第三方库
from torch import Tensor
from typing import Dict, List, Tuple  # 类型注解
from system.modules.utils import PoseTool  # 自定义模块，用于刚体变换工具


# 记录器类，用于记录和处理训练过程中的指标
class Recorder:
    def __init__(self):
        # 初始化记录字典，用于存储各类指标的记录
        self.record_dict: Dict[str, List] = {}

        # 定义了一个字典，将各种聚合函数映射到对应的函数
        self.reduction_func = {
            'min': self.min,  # 最小值聚合
            'max': self.max,  # 最大值聚合
            'mean': self.mean,  # 平均值聚合
            'best': self.best,  # 最优值聚合
            'none': lambda x: x  # 不做聚合，直接返回
        }

    # 添加一个字典的指标记录
    def add_dict(self, metric_dict: dict):
        for key, value in metric_dict.items():
            if key not in self.record_dict.keys():
                # 如果字典中没有该键，则初始化为一个空列表
                self.record_dict[key] = []
            # 向记录中追加值
            self.record_dict.get(key).append(value)

    # 单独添加一个项的记录
    def add_item(self, key: str, value):
        if key not in self.record_dict.keys():
            # 如果键不存在，则初始化为一个空列表
            self.record_dict[key] = []
        # 追加记录
        self.record_dict.get(key).append(value)

    # 计算所有记录的平均值
    def mean(self) -> dict:
        return_dict = {}
        for key, value in self.record_dict.items():
            if len(value) > 0:
                # 对每个键计算平均值
                return_dict[key] = sum(value) / len(value)
        return return_dict

    # 获取每个记录中的最大值
    def max(self) -> dict:
        return_dict = {}
        for key, value in self.record_dict.items():
            if len(value) > 0:
                # 对每个键取最大值
                return_dict[key] = max(value)
        return return_dict

    # 获取每个记录中的最小值
    def min(self) -> dict:
        return_dict = {}
        for key, value in self.record_dict.items():
            if len(value) > 0:
                # 对每个键取最小值
                return_dict[key] = min(value)
        return return_dict

    # 获取每个记录中最优的值
    def best(self) -> dict:
        return_dict = {}
        for key, value in self.record_dict.items():
            if len(value) > 0:
                # 判断第一个值与最后一个值的关系，选择最优的记录
                if value[0] > value[-1]:
                    return_dict[key] = min(value)
                else:
                    return_dict[key] = max(value)
        return return_dict

    # 将记录的内容转换为字符串，支持不同的聚合方法
    def tostring(self, reduction='best') -> str:
        assert reduction in ['min', 'max', 'mean', 'best', 'none']
        reduction_dic = self.reduction_func.get(reduction)()  # 根据传入的参数，选择聚合方法
        string = ''
        if len(reduction_dic) > 0:
            for key, value in reduction_dic.items():
                # 如果是数值则格式化输出
                if isinstance(value, list):
                    value_str = value
                else:
                    value_str = f'{value:4.5f}'  # 保留五位小数
                string += f'\t{key:<20s}: ({value_str})\n'
            string = '\n' + string
        return string

    # 清空记录
    def clear(self):
        self.record_dict.clear()


# 优化器类，用于根据输入参数选择不同的优化器
class Optimizer:
    def __init__(self, args):
        # 根据输入的参数选择优化器类型
        self.name = args.type.lower()
        self.kwargs = args.kwargs  # 额外的参数
        if self.name == 'adamw':
            self.optimizer = torch.optim.AdamW
        elif self.name == 'adam':
            self.optimizer = torch.optim.Adam
        elif self.name == 'sgd':
            self.optimizer = torch.optim.SGD
        else:
            raise NotImplementedError  # 如果不支持该类型则抛出异常

    # 调用该类时，会返回初始化好的优化器实例
    def __call__(self, parameters):
        return self.optimizer(params=parameters, **self.kwargs)


# 定义一个无操作的调度器
class IdentityScheduler(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    # 定义一个空的 step 方法
    def step(self):
        pass


# 调度器类，用于根据输入参数选择不同的调度器
class Scheduler:
    def __init__(self, args, epoch_length):
        # 根据输入的参数选择调度器类型
        self.name = args.type.lower()
        self.kwargs = deepcopy(args.kwargs)  # 额外的参数
        if 'T_max' in self.kwargs.keys():
            self.kwargs['T_max'] *= epoch_length
        if self.name == 'identity':
            self.scheduler = IdentityScheduler  # 无调度
        elif self.name == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR  # 余弦退火学习率
        elif self.name == 'cosine_restart':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts  # 余弦退火重启
        else:
            raise NotImplementedError  # 如果不支持该类型则抛出异常

    # 调用该类时，会返回初始化好的调度器实例
    def __call__(self, optimizer):
        return self.scheduler(optimizer=optimizer, **self.kwargs)


# 一个空的上下文管理器，作为占位符
class fakecast:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# 加载模型权重状态字典的函数
def try_load_state_dict(model, state_dict, name='model', log=True):
    model_keys = model.state_dict().keys()  # 获取模型中的权重键
    file_keys = state_dict.keys()  # 获取文件中的权重键
    if model_keys == file_keys:
        # 如果模型的键与文件的键完全一致，直接加载
        try:
            model.load_state_dict(state_dict)
            if log:
                logger.info(f"{name} loaded successfully.")
            return
        except:
            if log:
                logger.warning(f"{name} loaded failed.")
            return
    else:
        # 如果不一致，给出警告信息
        missing = model_keys - file_keys  # 模型中存在但文件中缺失的键
        warnings_str = f'{name} loaded with {len(model_keys)} in model, {len(file_keys)} in file.\n'
        if len(missing) != 0:
            warnings_str += f"{len(missing)} missing parameters (in model):\n" + ", ".join(missing) + '\n'
        unexpected = file_keys - model_keys  # 文件中存在但模型中缺失的键
        if len(unexpected) != 0:
            warnings_str += f"{len(unexpected)} unexpected parameters (in file):\n" + ", ".join(unexpected) + '\n'
        try:
            # 加载时不严格匹配
            model.load_state_dict(state_dict, strict=False)
            if log:
                logger.warning(warnings_str)
            return
        except:
            if log:
                logger.warning(f"{name} loaded failed.")
            return


# 使用 ICP (Iterative Closest Point) 算法进行刚体变换矩阵的修正
def icp_refinement(src: Tensor, dst: Tensor, init_R: Tensor, init_T: Tensor) -> Tuple[Tensor, Tensor]:
    """
    icp修正刚体变换矩阵

    :param src: (B, 3, N) 源点云
    :param dst: (B, 3, N) 目标点云
    :param init_R: (B, 3, 3) 初始旋转矩阵
    :param init_T: (B, 3, 1) 初始平移向量
    :return: (B, 3, 3), (B, 3, 1) 修正后的旋转矩阵和平移向量
    """
    B, device = init_R.shape[0], init_R.device  # 批次大小和设备
    src_pcd_o3d = o3d.geometry.PointCloud()  # 创建源点云对象
    dst_pcd_o3d = o3d.geometry.PointCloud()  # 创建目标点云对象

    src_list = src.detach().cpu().numpy()  # 将源点云转为 numpy 格式
    dst_list = dst.detach().cpu().numpy()  # 将目标点云转为 numpy 格式
    init_SE3_list = np.repeat(np.eye(4)[np.newaxis, :, :], axis=0, repeats=B)  # 生成批次初始 SE3 变换矩阵
    init_SE3_list[:, :3, :3] = init_R.detach().cpu().numpy()  # 设置初始旋转矩阵
    init_SE3_list[:, :3, 3:] = init_T.detach().cpu().numpy()  # 设置初始平移向量

    SE3_refinement = []  # 用于存储修正后的 SE3 矩阵
    for src_pcd, dst_pcd, init_SE3 in zip(src_list, dst_list, init_SE3_list):
        # 设置点云对象的坐标点
        src_pcd_o3d.points = o3d.utility.Vector3dVector(src_pcd.T)
        dst_pcd_o3d.points = o3d.utility.Vector3dVector(dst_pcd.T)

        # 进行 ICP 配准
        icp = o3d.pipelines.registration.registration_icp(
            source=src_pcd_o3d, target=dst_pcd_o3d, max_correspondence_distance=1.0, init=init_SE3,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))  # ICP 收敛标准

        # 获取 ICP 的变换矩阵
        icp_SE3 = icp.transformation
        delta_pose = np.linalg.inv(icp_SE3) @ init_SE3  # 计算变换矩阵的差异
        delta_R, delta_T = PoseTool.Rt(delta_pose)  # 提取旋转和平移差异
        delta_angle = np.arccos((np.trace(delta_R) - 1) / 2).item() * 180 / math.pi  # 计算角度差
        delta_translation = np.linalg.norm(delta_T).item()  # 计算平移差

        _DEBUG = False
        if _DEBUG:
            from utils.visualization import show_pcd  # 调用外部工具可视化
            src_init = init_SE3[:3, :3] @ src_pcd + init_SE3[:3, 3:]
            src_icp = icp_SE3[:3, :3] @ src_pcd + icp_SE3[:3, 3:]
            show_pcd([src_pcd.T, dst_pcd.T], [[1, 0, 0], [0, 1, 0]], window_name='origin')
            show_pcd([src_init.T, dst_pcd.T], [[1, 0, 0], [0, 1, 0]], window_name='gt')
            show_pcd([src_icp.T, dst_pcd.T], [[1, 0, 0], [0, 1, 0]],
                     window_name=f'icp: delta_angle={delta_angle:.2f}, delta_translation={delta_translation:.2f}')

        # 如果角度差和位移差过大，认为 ICP 失败，保留初始变换
        if delta_angle > 5 or delta_translation > 2:
            SE3_refinement.append(init_SE3)
            logger.warning('A suspected failed icp refinement has been discarded')
        else:
            # 否则使用 ICP 的结果
            SE3_refinement.append(icp_SE3)

    # 将 SE3 变换矩阵转换为 Tensor
    SE3_refinement = np.stack(SE3_refinement, axis=0)
    SE3_refinement = torch.from_numpy(SE3_refinement).float().to(device)
    R, T = SE3_refinement[:, :3, :3], SE3_refinement[:, :3, 3:]
    return R, T  # 返回修正后的旋转矩阵和平移矩阵

