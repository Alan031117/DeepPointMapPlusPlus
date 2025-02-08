import colorlog as logging  # 导入colorlog模块，并将其命名为logging，用于带有颜色的日志记录

logging.basicConfig(level=logging.INFO)  # 配置日志记录器，设置日志级别为INFO，以便记录信息级别及以上的日志
logger = logging.getLogger(__name__)  # 创建一个日志记录器对象，名称为当前模块名
logger.setLevel(logging.INFO)  # 设置日志记录器的日志级别为INFO

import os  # 导入os模块，用于操作系统相关功能，如文件路径操作
import random  # 导入random模块，用于生成随机数
import numpy as np  # 导入numpy模块，并将其命名为np，用于处理数值计算和数组操作
import torch  # 导入PyTorch框架，用于深度学习模型的构建和训练
import torch.nn as nn  # 从PyTorch中导入神经网络模块，提供各种神经网络层和功能
from torch.utils.data import Dataset  # 从PyTorch中导入Dataset类，用于构建自定义数据集
from glob import glob  # 导入glob模块，用于文件路径模式匹配操作，类似于shell的通配符功能
from tqdm import tqdm  # 导入tqdm模块，用于显示循环的进度条
from typing import Dict, List, Tuple, Callable, Union  # 从typing模块中导入类型提示工具，帮助指定变量的类型

from dataloader.heads.basic import PointCloudReader  # 从dataloader.heads.basic模块中导入PointCloudReader类，作为基础的点云数据读取器
from dataloader.heads.npz import NPZReader  # 从dataloader.heads.npz模块中导入NPZReader类，用于读取npz格式的点云数据
from dataloader.heads.npz_ss import NPZReader_W_SS
from dataloader.heads.npy import NPYReader  # 从dataloader.heads.npy模块中导入NPYReader类，用于读取npy格式的点云数据
from dataloader.heads.bin import BinReader  # 从dataloader.heads.bin模块中导入BinReader类，用于读取bin格式的点云数据
from dataloader.heads.pcd import PcdReader  # 从dataloader.heads.pcd模块中导入PcdReader类，用于读取pcd格式的点云数据
from dataloader.heads.kitti import KittiReader  # 从dataloader.heads.kitti模块中导入KittiReader类，用于读取KITTI数据集的点云数据
from dataloader.heads.kitti360 import \
    Kitti360Reader  # 从dataloader.heads.kitti360模块中导入Kitti360Reader类，用于读取KITTI-360数据集的点云数据
from scripts.icp_refinement import \
    get_refined_SE3  # 从scripts.icp_refinement模块中导入get_refined_SE3函数，用于进行ICP（迭代最近点）优化，得到优化后的SE3变换矩阵

# 创建一个字典READER，用于将字符串类型映射到相应的点云读取器类
READER: Dict[str, PointCloudReader] = {
    'auto': PointCloudReader,  # 'auto'对应的读取器类为基础的PointCloudReader类
    'npz': NPZReader,  # 'npz'对应的读取器类为NPZReader类，用于读取npz格式的点云数据
    'npz_ss': NPZReader_W_SS,  # 'npz'对应的读取器类为NPZReader类，用于读取npz格式的点云数据
    'npy': NPYReader,  # 'npy'对应的读取器类为NPYReader类，用于读取npy格式的点云数据
    'bin': BinReader,  # 'bin'对应的读取器类为BinReader类，用于读取bin格式的点云数据
    'pcd': PcdReader,  # 'pcd'对应的读取器类为PcdReader类，用于读取pcd格式的点云数据
    'kitti': KittiReader,  # 'kitti'对应的读取器类为KittiReader类，用于读取KITTI数据集的点云数据
    'kitti360': Kitti360Reader,  # 'kitti360'对应的读取器类为Kitti360Reader类，用于读取KITTI-360数据集的点云数据
}


def get_length_range(l):  # 定义一个名为 `get_length_range` 的函数，接受一个列表 `l` 作为参数
    length_range = [0]  # 初始化一个列表 `length_range`，其中包含一个初始值 `0`。这个列表用于存储累积的长度范围。
    for i in l:  # 遍历列表 `l` 中的每一个元素 `i`
        length_range.append(len(i) + length_range[
            -1])  # 计算当前元素 `i` 的长度 `len(i)`，并将其加上 `length_range` 列表中的最后一个值（即当前累积的长度）。将这个结果追加到 `length_range` 列表中。
    return length_range  # 返回计算得到的 `length_range` 列表，该列表表示累积的长度范围


class SlamDatasets(Dataset):  # 定义SlamDatasets类，继承自PyTorch的Dataset类，用于SLAM任务的数据集数据读取
    """
    SLAM任务数据集数据读取
    """

    def __init__(self,
                 args,
                 data_transforms: Callable = nn.Identity()):
        """
        :param args: 数据集参数配置字典
        :param data_transforms: 数据预处理与数据增强等变换
        """
        Dataset.__init__(self)  # 继承 Dataset

        # 输入参数 args
        self.args = args
        self.dataset_cfg = self.args.dataset
        self.registration_cfg = self.args.train.registration
        self.loop_detection_cfg = self.args.train.loop_detection

        self.data_transforms = data_transforms  # 数据变换方法

        self.dataset_list = self.load_dataset()  # 读取数据并存放在 dataset_list

        # 计算数据集数量（int32）
        self.pcd_range = get_length_range(self.dataset_list)
        self.pcd_range = torch.tensor(self.pcd_range, dtype=torch.int32)

        # 获取每条序列内部的帧间距离
        self.frame_distance = get_frame_dis(self.dataset_list)  # 获取每条数据集序列内部的帧间距离，并将其保存

        self._getitem_method = self._getitem_registration  # 默认设置getitem方法为注册任务的数据获取方法
        self.collate_fn = self.map_collate_fn  # 设置数据集的collate函数，用于在批处理中处理样本

    def __getitem__(self, item):  # 定义 __getitem__ 方法，使得 SlamDatasets 类的实例可以像列表一样使用索引访问数据
        return self._getitem_method(item)  # 返回通过调用 _getitem_method 获取的数据，传入的参数是 item（即索引值）

    def _getitem_loop_detection(self, item):  # 定义一个私有方法，用于获取指定索引 `item` 处的回环检测数据
        # 映射到具体数据集内的偏移量
        dataset_id = torch.sum(self.pcd_range <= item) - 1  # 通过累计的点云范围计算该索引位于第几个数据集，得到数据集的ID
        offset = item - self.pcd_range[dataset_id]  # 计算该索引在当前数据集中的偏移量
        curren_dataset = self.dataset_list[dataset_id]  # 获取当前数据集的引用

        # (pcd, lidar_seg), ego_rotation, ego_translation, images
        # Load first frame
        frame1 = curren_dataset[offset]  # 根据偏移量加载第一帧数据

        scene_id, frame_offset = curren_dataset.get_frame_order(offset)  # 获取当前帧的场景ID和该帧在场景内的帧偏移量
        frame_dis = self.frame_distance[dataset_id][scene_id][frame_offset]  # 获取当前帧与场景内其他帧之间的相对距离

        '''
        训练回环检测时以不等概率从各个距离内采样
        0    .50      .75     1
          <d     d~2d     >2d
        '''
        s = random.random()  # 生成一个0到1之间的随机数，用于不等概率采样
        d = self.loop_detection_cfg.distance  # 从配置中获取用于采样的距离阈值 `d`
        margin = getattr(self.loop_detection_cfg, 'margin', 0)
        if s < 0.5:  # 如果随机数小于0.5，选择距离小于等于 `d` 的帧
            dis_mask = frame_dis <= (d - margin)
        elif s < 0.75:  # 如果随机数介于0.5和0.75之间，选择距离介于 `d` 和 `2d` 之间的帧
            dis_mask = (frame_dis > (d + margin)) & (frame_dis <= 2 * d)
        else:  # 如果随机数大于等于0.75，选择距离大于 `2d` 的帧
            dis_mask = frame_dis > 2 * d

        optional_pair_offset = torch.nonzero(dis_mask).squeeze(1) - frame_offset  # 获取满足距离条件的帧相对于当前帧的偏移量
        optional_pair_offset = optional_pair_offset.tolist()  # 将张量转换为列表
        if len(optional_pair_offset) > 0:  # 如果存在满足条件的帧
            pair_offset = random.choice(optional_pair_offset)  # 随机选择一个偏移量
        else:  # 如果没有满足条件的帧
            pair_offset = 0  # 设置偏移量为0，即选择当前帧作为第二帧
        frame2 = curren_dataset[offset + pair_offset]  # 根据计算得到的偏移量加载第二帧数据
        frame1 = self.data_transforms(frame1)  # 对第一帧数据应用预处理和增强变换
        frame2 = self.data_transforms(frame2)  # 对第二帧数据应用预处理和增强变换
        return *frame1, *frame2  # 返回展开后的两个帧的数据，通常包括点云、位姿信息等

    def _getitem_registration(self, index: int) -> Tuple[List, dict]:
        """
        地图级别采样

        :param index: 采样帧的索引
        :return: 至多由K个frame组成的列表，以及包含额外信息的字典
        """
        S = random.randint(2, self.registration_cfg.K)  # S = 随机整数[2，K]
        if random.random() < 0.34:  # S = 2 的概率为 0.34
            S = 2
        if self.registration_cfg.fill:  # 若配置文件中存在 fill
            num_map = self.registration_cfg.K_max // S  # 从数据集中再随机采样若干个map，使得总帧数接近K_max
        else:
            num_map = 1
        info = dict(dsf_index=[], refined_SE3_file=[], num_map=num_map)
        """
            字典 info
                键1  dsf_index(采样信息,数据帧索引)
                键2  refined_SE3_file(优化后的SE3文件)
                键3  num_map(地图数量)
        """

        frame_list = []  # 帧列表
        kitti360_only_flag = (self.pcd_range < index).sum() == 2  # 魔法代码，由于kitti360不含有分割标签，所以我们要保证输入数据中至少包含一组非kitti360的点云
        for i in range(num_map):  # 遍历所有地图
            if i == 0:  # 对于第一个地图
                frame_list += self._map_query(index, K=S, info=info)  # 以 index 为索引采样共S帧，并将结果添加到frame_list
            else:  # 对于其它地图
                rand_index = random.randint(0, self.__len__() - 1)  # 在整个数据集范围内随机选择一个索引
                kitti360_only_flag &= ((self.pcd_range < rand_index).sum() == 2)
                while kitti360_only_flag and i == (num_map - 1):
                    # 准备采样最后一组点云，且之前从未采样过kitti360之外的数据集，需要重新采样
                    rand_index = random.randint(0, self.__len__() - 1)  # 在整个数据集范围内随机选择一个索引
                    kitti360_only_flag &= ((self.pcd_range < rand_index).sum() == 2)

                frame_list += self._map_query(rand_index, K=S,
                                              info=info)  # 调用_map_query方法，以index为索引采样S帧，并将结果添加到frame_list中

        return frame_list, info  # 返回采样得到的帧列表和包含采样信息的字典

    def _map_query(self, index: int, K: int, info: dict) -> List:
        """
        查询一片地图，返回index所指帧附近的K帧(包括index自己)
        若指定offset内不足K帧，则返回最近的K帧，否则从offset范围内随机返回K帧

        :param index: 帧索引，指定查询的帧位置
        :param K: 查询的帧数量，返回的帧数
        :param info: 记录查询地图的索引、refined SE3等信息，用于保存查询过程中的额外信息
        :return: 返回由K帧组成的列表
        """
        # 映射到具体数据集内
        dataset_id = (torch.sum(self.pcd_range <= index) - 1).item()  # 获取 index 所在数据集ID
        # 计算当前帧在该数据集中的偏移量
        offset = index - self.pcd_range[dataset_id]
        # 当前数据集是index所在数据集
        curren_dataset = self.dataset_list[dataset_id]
        # 获取该数据集中的场景ID和帧偏移量
        scene_id, frame_offset = curren_dataset.get_frame_order(offset)
        # 获取index帧和其他帧的距离信息
        frame_dis = self.frame_distance[dataset_id][scene_id][frame_offset]

        # 根据距离筛选满足条件的帧，选取距离小于注册配置中指定值减去0.25的帧
        dis_mask = frame_dis <= self.registration_cfg.distance - 0.25
        if dis_mask.sum() <= K:  # 如果满足条件的帧数 ≤ K
            # 获取符合条件的帧的偏移量
            optional_frame_offsets = torch.nonzero(dis_mask).squeeze(1) - frame_offset
            # 将偏移量转化为列表形式，并去除0偏移量（即去除自己本身）
            optional_frame_offsets = optional_frame_offsets.tolist()
            optional_frame_offsets.remove(0)

            # 若没有符合条件的帧（即自己是唯一符合条件的帧），则添加自己
            if len(optional_frame_offsets) == 0:
                optional_frame_offsets.append(0)

            # 重复optional_frame_offsets，直到其长度足够，以便能够随机选择K-1个帧
            optional_frame_offsets = optional_frame_offsets * (K // len(optional_frame_offsets) + 1)
            # 随机从可选帧中选择K-1帧
            map_frame_offsets = random.sample(optional_frame_offsets, k=K - 1)
            # 在随机帧的前面插入自己的帧（即0偏移量）
            map_frame_offsets.insert(0, 0)
        else:
            # 如果符合条件的帧数大于K，则随机选择K-1帧
            optional_frame_offsets = torch.nonzero(dis_mask).squeeze(1) - frame_offset
            optional_frame_offsets = optional_frame_offsets.tolist()
            optional_frame_offsets.remove(0)
            # 随机选择K-1个帧的偏移量
            map_frame_offsets = random.sample(optional_frame_offsets, k=K - 1)
            # 在随机选择的帧中插入自己的帧（即0偏移量）
            map_frame_offsets.insert(0, 0)

        # 将采样帧的索引信息添加到info字典中，记录每个帧的来源数据集、场景ID和帧偏移量
        info['dsf_index'] += [(dataset_id, scene_id, frame_offset + off) for off in map_frame_offsets]

        # 判断当前数据集是否为CARLA数据集，并根据此决定refined_SE3文件路径
        if 'carla' in curren_dataset.name.lower():
            refined_SE3_file = ''  # 如果是CARLA数据集，refined_SE3文件路径为空
        else:
            # 否则根据场景ID在数据集中找到refined_SE3.pkl文件
            refined_SE3_file = os.path.join(curren_dataset.scene_list[scene_id].root, 'refined_SE3.pkl')
            if not os.path.exists(refined_SE3_file):
                refined_SE3_file = ''
        # 将refined_SE3文件路径添加到info字典中
        info['refined_SE3_file'].append(refined_SE3_file)

        # 加载查询到的K帧并返回
        frame_list = []  # 用于存储加载后的K帧
        for map_frame_offset in map_frame_offsets:
            # 加载对应偏移量的帧
            frame = curren_dataset[offset + map_frame_offset]
            # 对帧数据应用预处理和增强变换
            frame = self.data_transforms(frame)
            # 将处理后的帧添加到帧列表中
            frame_list.append(frame)

        # 返回最终的K帧数据列表
        return frame_list

    @staticmethod  # 声明该方法为静态方法，与类实例无关，可以直接通过类名调用
    def map_collate_fn(batch):  # 定义一个静态方法，用于将一个批次的数据合并成一个张量
        frame_list, info = batch[0]  # 从批次中提取第一个元素
        batch_data_list = []  # 初始化一个空列表，用于存储合并后的数据
        for data in zip(*frame_list):  # 使用zip函数将frame_list中的元素逐个配对，形成一个元组的迭代器
            batch_data_list.append(
                torch.stack(data, dim=0))  # 将每个配对后的元组（对应一个时间步的数据）沿新的维度（dim=0）堆叠成一个张量，并添加到batch_data_list中
        return *batch_data_list, info  # 返回展开的batch_data_list元素，以及info字典

    def __len__(self):  # 定义 __len__ 方法，使得实例对象可以返回数据集的长度
        return self.pcd_range[-1]  # 返回 pcd_range 列表的最后一个元素，这个值表示整个数据集的累积点云数量的总长度

    def load_dataset(self):  # 定义一个加载数据集的方法
        dataset_list: List[BasicDataset] = []  # 空列表dataset_list，用于存储单个数据集
        for dataset_dict in self.dataset_cfg:  # 遍历数据集
            name = dataset_dict.name  # 数据集名称
            root = dataset_dict.root  # 数据集位置
            scenes = dataset_dict.scenes  # 数据集场景列表
            reader_cfg = dataset_dict.reader  # 数据集读取头
            reader = READER[reader_cfg.type](**reader_cfg.get('kwargs', {}))  # 根据配置中的类型创建一个数据读取器对象，并传入可选的参数
            basic_dataset = BasicDataset(root=root, reader=reader, scenes=scenes, name=name.lower(), args=self.args)
            # 创建一个BasicDataset对象
            dataset_list.append(basic_dataset)  # 将对象添加到 dataset_list 中
            logger.info(f'Load {name} successfully: \'{basic_dataset.root}\'')
        return dataset_list  # 返回dataset_list

    def get_seq_range(self):  # 定义一个方法，用于获取每个点云序列的起始范围
        """每个点云序列的起始范围"""
        real_range = [0]  # 初始化一个列表，首元素为0，用于存储每个序列的累积起始位置
        for dataset in self.dataset_list:  # 遍历每个数据集
            for scene in dataset.scene_list:  # 遍历每个数据集中的场景
                for agent in scene.agent_list:  # 遍历每个场景中的agent（例如，来自不同传感器的数据）
                    real_range.append(len(agent) + real_range[-1])  # 计算当前agent的长度，并将其累加到real_range列表的最后一个元素上
        return torch.tensor(real_range, dtype=torch.int32)  # 返回一个张量，表示每个序列的起始范围，数据类型为int32

    def get_datasets(self):  # 定义一个方法，用于获取数据集列表
        return self.dataset_list  # 返回包含所有数据集的列表

    @property  # 使用@property装饰器，使得方法可以像属性一样被访问
    def seq_begin_list(self):  # 定义一个属性方法，用于获取每个点云序列的起始范围
        return self.get_seq_range()  # 调用get_seq_range方法，返回每个点云序列的起始范围

    def get_data_source(self, item):  # 定义一个方法，用于获取给定索引的帧所在的数据集
        dataset_id = torch.sum(self.pcd_range <= item) - 1  # 通过计算索引在哪个数据集的范围内，确定数据集的ID
        return self.dataset_list[dataset_id]  # 返回对应的数据集

    def registration(self):  # 定义一个方法，用于将当前的数据获取模式设置为注册任务
        self._getitem_method = self._getitem_registration  # 将数据获取方法设置为_getitem_registration，用于点云配准
        self.collate_fn = self.map_collate_fn  # 将collate函数设置为map_collate_fn，用于将批次数据合并成张量

    def loop_detection(self):  # 定义一个方法，用于将当前的数据获取模式设置为回环检测任务
        self._getitem_method = self._getitem_loop_detection  # 将数据获取方法设置为_getitem_loop_detection，用于回环检测任务
        self.collate_fn = None  # 将collate函数设置为None，因为在回环检测中可能不需要特殊的collate处理

    def __repr__(self):  # 定义__repr__方法，用于返回一个对象的正式字符串表示，通常用于调试
        print('=' * 50)  # 打印分隔线
        print(f'SlamDatasets: num_datasets={len(self.dataset_list)}\n'  # 打印数据集的总数量
              f'    |')
        for dataset in self.dataset_list:  # 遍历每个数据集
            print(f'    |——{dataset.name}\n'  # 打印数据集名称
                  f'    |   |——train: num_scenes={len(dataset.scene_list)} | num_frames={dataset.pcd_range[-1]}\n'  # 打印数据集中场景的数量和帧的数量
                  f'    |')
        print('=' * 50)  # 打印分隔线

    def __str__(self):  # 定义__str__方法，用于返回对象的字符串表示，通常用于打印用户友好的信息
        my_str = ''  # 初始化一个空字符串
        my_str += ('=' * 50 + '\n')  # 添加分隔线
        my_str += (f'SlamDatasets: num_datasets={len(self.dataset_list)}\n'  # 添加数据集的总数量信息
                   f'    |\n')
        for dataset in self.dataset_list:  # 遍历每个数据集
            my_str += (f'    |——{dataset.name}\n'  # 添加数据集名称信息
                       f'    |   |——train: num_scenes={len(dataset.scene_list)} | num_frames={dataset.pcd_range[-1]}\n'  # 添加数据集中场景的数量和帧的数量信息
                       f'    |\n')
        my_str += ('=' * 50)  # 添加分隔线
        return my_str  # 返回生成的字符串


class BasicDataset:  # 定义 BasicDataset 类，用于读取单个数据集
    """
    单个数据集读取，文件结构如下
    dataset
        |--scenes
             |--00
             |--01
             |--02
                 |--agent 0
                 |--agent 1
                 |--agent 2
                        |--0.npz
                        |--1.npz
                        |--2.npz
    """

    def __init__(self,
                 args,  # 初始化函数，接收各种参数配置
                 root: str,  # 数据集的根目录
                 reader: NPZReader,  # 数据读取器，用于读取 .npz 文件
                 scenes: list,  # 场景列表，指定要读取的数据集中的哪些场景
                 name: str,  # 数据集的名称
                 ):
        """
        :param root: 数据集根目录
        :param reader: 数据读取头
        :param scenes: 选中的场景列表
        :param name: 数据集名称
        """

        self.args = args  # 输入参数
        self.root = root  # 数据集位置
        self.scenes = scenes  # 场景列表
        self.name = name  # 数据集的名称

        if not isinstance(self.root, str) or not os.path.isdir(self.root):  # 检查 root 是否为有效的目录路径
            raise NotADirectoryError(f'\'{self.root}\' is not a directory')  # 如果不是有效的目录，则抛出异常

        # 加载场景  [Scene1, Scene2, ...]
        self.scene_list: List[BasicScene] = []  # 列表BasicScene，用于存储单帧信息
        for scene_name in self.scenes:  # 遍历场景列表中的每个场景名
            scene_root = os.path.join(self.root, scene_name)  # 组合得到该场景的完整路径
            if not os.path.isdir(scene_root):  # 检查该场景路径是否为有效的目录
                raise NotADirectoryError(f'\'{scene_root}\' is not a directory')  # 如果不是有效目录，则抛出异常
            self.scene_list.append(BasicScene(root=scene_root, reader=reader, parent=self,
                                              args=self.args))  # 创建 BasicScene 对象，并将其添加到 scene_list 列表中

        self.pcd_range = get_length_range(self.scene_list)  # 计算每个场景的累积长度范围
        self.pcd_range = torch.tensor(self.pcd_range, dtype=torch.int32)  # 将累积长度范围转换为 PyTorch 的张量，数据类型为 int32

    def __getitem__(self, item):  # 定义 __getitem__ 方法，使得 BasicDataset 类的实例可以像列表一样使用索引访问数据
        scene_id = torch.sum(self.pcd_range <= item) - 1  # 查找 item 所在场景id
        offset = item - self.pcd_range[scene_id]  # 计算 item 在场景中的具体位置
        return self.scene_list[scene_id][offset]  # 返回场景id、帧id

    def __len__(self):  # 定义 __len__ 方法，使得实例对象可以返回数据集的长度
        return self.pcd_range[-1]  # 返回 pcd_range 列表的最后一个元素，这个值表示整个数据集的累积点云数量的总长度

    def get_scenes(self):  # 定义一个方法，用于获取数据集中的场景列表
        return self.scene_list  # 返回 scene_list 列表，其中包含所有的 BasicScene 对象

    def get_frame_order(self, item):  # 定义一个方法，用于获取给定索引的帧所在的场景及其在场景中的偏移量
        scene_id = torch.sum(self.pcd_range <= item) - 1  # 通过计算索引在 pcd_range 中的位置，确定该索引位于第几个场景
        offset = item - self.pcd_range[scene_id]  # 计算该索引在当前场景中的偏移量
        return scene_id.item(), offset.item()  # 返回场景 ID 和帧的偏移量，以元组的形式返回


class BasicScene:  # 定义 BasicScene 类，用于表示单个场景
    """
    单个场景
    """

    def __init__(self,
                 args,  # 初始化函数，接收各种参数配置
                 root: str,  # 场景的根目录
                 reader: NPZReader,  # 数据读取器，用于读取 .npz 文件
                 parent: BasicDataset,  # 所属的数据集
                 ):
        """
        :param root: 场景根目录
        :param reader: 数据读取头
        :param parent: 所属数据集
        """
        self.root = root  # 保存场景的根目录
        self.args = args  # 保存传入的参数配置
        self.parent = parent  # 保存所属的数据集

        # 加载agent  [Agent1, Agent2, ...]
        self.agent_list: List[BasicAgent] = []  # 初始化一个空列表，用于存储加载的agent
        for agent_name in sorted(os.listdir(self.root)):  # 遍历场景根目录中的所有子目录（即每个agent）
            agent_root = os.path.join(self.root, agent_name)  # 组合得到该agent的完整路径
            if os.path.isdir(agent_root):  # 检查该路径是否为有效的目录
                if self.args.multi_agent:  # 如果启用了多agent模式
                    # 将该agent的数据分割成3个部分，分别创建3个 BasicAgent 对象并添加到 agent_list 列表中
                    self.agent_list.append(
                        BasicAgent(root=agent_root, reader=reader, parent=self, split_num=3,
                                   split_index=0))  # 第一个分割部分
                    self.agent_list.append(
                        BasicAgent(root=agent_root, reader=reader, parent=self, split_num=3,
                                   split_index=1))  # 第二个分割部分
                    self.agent_list.append(
                        BasicAgent(root=agent_root, reader=reader, parent=self, split_num=3,
                                   split_index=2))  # 第三个分割部分
                else:  # 如果未启用多agent模式
                    self.agent_list.append(
                        BasicAgent(root=agent_root, reader=reader,
                                   parent=self))  # 直接创建一个 BasicAgent 对象并添加到 agent_list 列表中

        # 按顺序编排Agent采集点云的范围
        self.pcd_range = get_length_range(self.agent_list)  # agent数量
        self.pcd_range = torch.tensor(self.pcd_range, dtype=torch.int32)  # 将累积长度范围转换为 PyTorch 的张量，数据类型为 int32

    def __getitem__(self, item):  # 定义 __getitem__ 方法，使得 BasicScene 类的实例可以像列表一样使用索引访问数据
        agent_id = torch.sum(self.pcd_range <= item) - 1  # 确定 item 所在 agent ID
        offset = item - self.pcd_range[agent_id]  # 计算 item 在当前 agent 中的位置
        return self.agent_list[agent_id][offset]  # 返回 agentID、agent内偏移量

    def __len__(self):  # 定义 __len__ 方法，使得实例对象可以返回场景的长度
        return self.pcd_range[-1]  # 返回 pcd_range 列表的最后一个元素，这个值表示整个场景的累积点云数量的总长度

    def get_multi_agent(self):  # 定义一个方法，用于获取场景中的所有 agent
        return self.agent_list  # 返回 agent_list 列表，其中包含所有的 BasicAgent 对象


class BasicAgent(Dataset):  # 定义 BasicAgent 类，继承自 PyTorch 的 Dataset 类，用于表示单个采集点云的智能体
    """
    单个采集点云的智能体
    """

    def __init__(self,
                 root: str,  # 初始化函数，接收根目录路径
                 reader: Union[PointCloudReader, str],  # 数据读取器，可能是 PointCloudReader 对象或字符串
                 parent: BasicScene = None,  # 所属的场景对象，默认为 None
                 split_num: int = 1,  # 分割数量，用于将数据集分割成多个部分
                 split_index: int = 0  # 当前智能体在分割后的部分中的索引
                 ):
        """
        :param root: 场景根目录
        :param reader: 数据读取头
        :param parent: 所属场景
        """
        Dataset.__init__(self)  # 调用父类 Dataset 的初始化方法
        self.root = root  # 保存根目录路径
        self.reader = reader  # 保存数据读取器
        self.parent = parent  # 保存所属的场景对象
        self.data_transforms = None  # 初始化数据变换方法为 None
        file_name_list = glob(os.path.join(self.root, '*.*'))  # 获取根目录下所有文件的完整路径列表
        file_type = set([os.path.splitext(i)[1] for i in file_name_list])  # 提取每个文件的扩展名，并去重
        assert len(file_type) <= 1, 'The root can only contain files of the SAME type'  # 确保目录中所有文件类型一致，否则抛出异常
        file_type = file_type.pop()[1:]  # 获取唯一的文件类型（去掉扩展名前的点）
        if self.reader == 'auto':  # 如果读取器设置为 'auto'
            self.reader = READER[file_type]()  # 根据文件类型自动选择合适的读取器
        file_name_list = sorted(file_name_list,
                                key=lambda s: int(os.path.basename(s).split('.')[0]))  # 按照文件名中的数字部分排序文件列表

        if split_num > 1:  # 如果数据集被分割成多个部分
            total_len = len(file_name_list)  # 获取文件总数量
            agent_ratio = 1 / split_num  # 计算每个智能体占据的数据比例
            overlap_ratio = 1 / 20  # 设置重叠比例，默认为 5%
            start_ratio = max(agent_ratio * split_index - overlap_ratio, 0.0)  # 计算当前智能体数据的起始位置比例
            end_ratio = min(agent_ratio * (split_index + 1) + overlap_ratio, 1.0)  # 计算当前智能体数据的结束位置比例
            self.file_list = file_name_list[
                             int(total_len * start_ratio):int(total_len * end_ratio)]  # 根据起始和结束比例截取当前智能体的数据文件列表
        else:  # 如果不需要分割数据集
            self.file_list = file_name_list  # 直接使用整个文件列表

    def __getitem__(self, item):  # 定义 __getitem__ 方法，使得 BasicAgent 类的实例可以像列表一样使用索引访问数据
        data = self.reader(self.file_list[item])  # 读取帧所在npz文件，并返回数据（点云、位姿等）
        if self.data_transforms is not None:  # 如果设置了数据变换方法
            data = self.data_transforms(data)  # 对数据应用变换
        return data  # 返回处理后的数据

    def __len__(self):  # 定义 __len__ 方法，使得实例对象可以返回数据文件的数量
        return len(self.file_list)  # 返回文件列表的长度

    def set_independent(self, data_transforms: Callable):  # 定义一个方法，用于设置数据预处理变换方式
        """设置数据预处理变换方式，从而作为独立地数据提取模块，用于多智能体模式"""
        self.data_transforms = data_transforms  # 保存传入的数据变换方法，使其可以在获取数据时应用


def get_frame_dis(dataset_list: List[BasicDataset]) -> List[List[torch.Tensor]]:  # 定义一个函数，用于获取所有场景内各帧的相对距离
    """
    获取所有场景内各帧的相对距离

    :param dataset_list: 各数据集
    :return: dataset[scene[frame]]
    """
    frame_distance = []  # 初始化一个空列表，用于存储每个数据集中每个场景的帧距离矩阵
    for i, dataset in enumerate(dataset_list):  # 遍历每个数据集
        dataset_frame_dis = []  # 初始化一个空列表，用于存储当前数据集的所有场景的帧距离矩阵
        for j, scene in enumerate(dataset.scene_list):  # 遍历数据集中的每个场景
            frame_files = []  # 初始化一个空列表，用于存储当前场景中所有agent的帧文件路径
            for agent in scene.agent_list:  # 遍历场景中的每个agent
                frame_files += agent.file_list  # 将agent的帧文件列表追加到frame_files中

            # 检查场景目录下有无frame_dis文件，若有则读取并检查，否则计算后写入
            frame_dis_file = os.path.join(scene.root, 'frame_dis.npy')  # 构建frame_dis.npy文件的路径
            cache_right = False  # 初始化一个标志，用于检查缓存文件是否有效
            if os.path.exists(frame_dis_file):  # 如果frame_dis文件存在
                frame_dis: np.ndarray = np.load(frame_dis_file).astype(np.float32)  # 读取文件，并转换为float32类型的numpy数组
                if frame_dis.shape[0] == frame_dis.shape[1] == len(
                        frame_files):  # 检查读取的矩阵是否是一个N×N的对称矩阵，且N等于frame_files的长度
                    cache_right = True  # 如果检查通过，标记缓存文件有效
            if not cache_right:  # 如果缓存文件无效
                # 读取scene下所有帧的世界坐标
                frame_poses = []  # 初始化一个空列表，用于存储每个帧的位置信息
                loop = tqdm(frame_files, total=len(frame_files), leave=False, dynamic_ncols=True)  # 使用tqdm显示进度条
                loop.set_description(
                    f'Building \'frame_dis.npy\' | Dataset No.{i + 1} | Scene No.{j + 1}')  # 设置进度条的描述信息
                for frame_file in loop:  # 遍历每个帧文件
                    with np.load(frame_file, allow_pickle=True) as npz:  # 加载npz文件
                        frame_pose = npz['ego_translation'].squeeze(1).astype(
                            np.float32)  # 提取位置信息并转换为float32类型，形状为(3, 1)
                        frame_poses.append(frame_pose)  # 将位置信息添加到frame_poses列表中
                frame_poses = np.stack(frame_poses, axis=0)  # 将所有帧的位置信息堆叠成一个数组，形状为(N, 3)
                frame_dis = np.linalg.norm(  # 计算每对帧之间的欧几里得距离，生成一个距离矩阵
                    x=(np.expand_dims(frame_poses, axis=1) - np.expand_dims(frame_poses, axis=0)), ord=2,
                    axis=-1)  # 扩展维度后计算两帧之间的距离，得到(N, N)矩阵

                np.save(file=frame_dis_file, arr=frame_dis)  # 将计算得到的距离矩阵保存到frame_dis.npy文件中
                logger.info(f'File \'frame_dis\' has been saved in {frame_dis_file}')  # 打印日志，提示文件保存成功

            frame_dis = torch.from_numpy(frame_dis).half()  # 将numpy数组转换为PyTorch的半精度浮点张量
            dataset_frame_dis.append(frame_dis)  # 将当前场景的帧距离矩阵添加到dataset_frame_dis列表中
        frame_distance.append(dataset_frame_dis)  # 将当前数据集的帧距离矩阵列表添加到frame_distance列表中
    return frame_distance  # 返回包含所有数据集的帧距离矩阵的列表


def get_adjacent_frame_mask(dataset_list: List[BasicDataset], max_dist: float) -> List[
    List[torch.Tensor]]:  # 定义一个函数，用于获取所有场景内各帧的邻近帧掩码矩阵
    """
    获取所有场景内各帧的邻近帧掩码矩阵

    :param dataset_list: 各数据集
    :param max_dist: 判定邻近帧的最近距离
    :return: dataset[scene[frame]]
    """
    adjacent_frame_mask = []  # 初始化一个空列表，用于存储每个数据集中每个场景的邻近帧掩码矩阵
    for i, dataset in enumerate(dataset_list):  # 遍历每个数据集
        dataset_frame_mask = []  # 初始化一个空列表，用于存储当前数据集的所有场景的邻近帧掩码矩阵
        for j, scene in enumerate(dataset.scene_list):  # 遍历数据集中的每个场景
            frame_files = []  # 初始化一个空列表，用于存储当前场景中所有agent的帧文件路径
            for agent in scene.agent_list:  # 遍历场景中的每个agent
                frame_files += agent.file_list  # 将agent的帧文件列表追加到frame_files中

            # 检查场景目录下有无frame_dis文件，若有则读取并检查，否则计算后写入
            frame_dis_file = os.path.join(scene.root, 'frame_dis.npy')  # 构建frame_dis.npy文件的路径
            cache_right = False  # 初始化一个标志，用于检查缓存文件是否有效
            if os.path.exists(frame_dis_file):  # 如果frame_dis文件存在
                frame_dis: np.ndarray = np.load(frame_dis_file).astype(np.float32)  # 读取文件，并转换为float32类型的numpy数组
                if frame_dis.shape[0] == frame_dis.shape[1] == len(
                        frame_files):  # 检查读取的矩阵是否是一个N×N的对称矩阵，且N等于frame_files的长度
                    cache_right = True  # 如果检查通过，标记缓存文件有效
            if not cache_right:  # 如果缓存文件无效
                # 读取scene下所有帧的世界坐标
                frame_poses = []  # 初始化一个空列表，用于存储每个帧的位置信息
                loop = tqdm(frame_files, total=len(frame_files), leave=False, dynamic_ncols=True)  # 使用tqdm显示进度条
                loop.set_description(
                    f'Building \'frame_dis.npy\' | Dataset No.{i + 1} | Scene No.{j + 1}')  # 设置进度条的描述信息
                for frame_file in loop:  # 遍历每个帧文件
                    with np.load(frame_file, allow_pickle=True) as npz:  # 加载npz文件
                        frame_pose = npz['ego_translation'].squeeze(1).astype(
                            np.float32)  # 提取位置信息并转换为float32类型，形状为(3, 1)
                        frame_poses.append(frame_pose)  # 将位置信息添加到frame_poses列表中
                frame_poses = np.stack(frame_poses, axis=0)  # 将所有帧的位置信息堆叠成一个数组，形状为(N, 3)
                frame_dis = np.linalg.norm(  # 计算每对帧之间的欧几里得距离，生成一个距离矩阵
                    x=(np.expand_dims(frame_poses, axis=1) - np.expand_dims(frame_poses, axis=0)), ord=2,
                    axis=-1)  # 扩展维度后计算两帧之间的距离，得到(N, N)矩阵

                np.save(file=frame_dis_file, arr=frame_dis)  # 将计算得到的距离矩阵保存到frame_dis.npy文件中
                logger.info(f'File \'frame_dis\' has been saved in {frame_dis_file}')  # 打印日志，提示文件保存成功

            frame_dis_mask = torch.from_numpy(frame_dis) <= max_dist  # 创建邻近帧掩码，标记距离在max_dist以内的帧对
            dataset_frame_mask.append(frame_dis_mask)  # 将当前场景的邻近帧掩码矩阵添加到dataset_frame_mask列表中
        adjacent_frame_mask.append(dataset_frame_mask)  # 将当前数据集的邻近帧掩码矩阵列表添加到adjacent_frame_mask列表中
    return adjacent_frame_mask  # 返回包含所有数据集的邻近帧掩码矩阵的列表
