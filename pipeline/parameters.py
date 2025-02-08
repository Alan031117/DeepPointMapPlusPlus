from easydict import EasyDict  # 导入EasyDict模块，用于将字典转换为对象
from collections import Iterable  # 导入Iterable模块，用于判断对象是否为可迭代对象
import argparse  # 导入argparse模块，用于解析命令行参数
import colorlog as logging  # 导入colorlog模块并将其命名为logging，用于带有颜色的日志记录
logger = logging.getLogger(__name__)  # 创建一个日志记录器对象，名称为当前模块名
logger.setLevel(logging.INFO)  # 设置日志记录器的日志级别为INFO


def str_to_bool(s):  # 定义一个将字符串转换为布尔值的函数
    if s.lower() == 'true':  # 如果字符串为'true'（不区分大小写），返回True
        return True
    elif s.lower() == 'false':  # 如果字符串为'false'（不区分大小写），返回False
        return False
    else:  # 如果字符串既不是'true'也不是'false'，则抛出类型错误异常
        raise TypeError(f'str {s} can not convert to bool.')


def update_args(args, cfg: dict):  # 定义一个函数，用于更新命令行参数args并从配置文件cfg中加载参数
    def subdict2edict(iterable_ob):  # 定义一个递归函数，将嵌套的字典或可迭代对象转换为EasyDict对象
        for i, element in enumerate(iterable_ob):  # 遍历可迭代对象中的每个元素
            if isinstance(element, dict):  # 如果元素是字典，则将其转换为EasyDict对象
                iterable_ob[i] = EasyDict(element)
            elif isinstance(element, Iterable) and not isinstance(element, str):  # 如果元素是可迭代对象且不是字符串，递归调用自己
                subdict2edict(element)

    for key, value in cfg.items():  # 遍历配置文件中的每个键值对
        if not hasattr(args, key):  # 如果命令行参数args中不存在该键（参数），记录一个警告日志
            logger.warning(f'Found unknown parameter in yaml file: {key}')
        if isinstance(value, dict):  # 如果值是字典，则将其转换为EasyDict对象
            value = EasyDict(value)
        elif isinstance(value, Iterable) and not isinstance(value, str):  # 如果值是可迭代对象且不是字符串，递归调用subdict2edict函数
            subdict2edict(value)
        setattr(args, key, value)  # 将配置文件中的值设置为args对象的属性
    return args  # 返回更新后的args对象


parser = argparse.ArgumentParser(description='DeepPointMap SLAM algorithm')  # 解析命令行参数

# 添加命令行参数
parser.add_argument('--name',                   default='DeepPointMap',         type=str,
                    help='Name of the model')  # 添加name参数，默认值为'DeepPointMap'，类型为字符串，表示模型的名称
parser.add_argument('--version',                default='v1.0',                 type=str,
                    help='Version of the model')  # 添加version参数，默认值为'v1.0'，类型为字符串，表示模型的版本
parser.add_argument('--mode',                   default='train',                type=str,
                    choices=['train', 'infer'])  # 添加mode参数，默认值为'train'，类型为字符串，可选值为'train'或'infer'
parser.add_argument('--checkpoint', '-ckpt',    default='',                     type=str,
                    help='Training checkpoint file')  # 添加checkpoint参数，默认值为空字符串，类型为字符串，表示训练的检查点文件路径
parser.add_argument('--weight', '-w',           default='',                     type=str,
                    help='Model pre-training weight')  # 添加weight参数，默认值为空字符串，类型为字符串，表示模型预训练权重文件路径
parser.add_argument('--yaml_file', '-yaml',     default='',                     type=str,
                    help='Yaml file which contains config parameters')  # 包含配置参数的yaml文件路径
parser.add_argument('--num_workers',            default=4,                      type=int,
                    help='Number of threads used for parallel data loading')  # 用于并行数据加载的线程数，默认为4
parser.add_argument('--thread_safety',          default=False,                  action='store_true',
                    help='Whether the data loading method is thread safety')  # 数据加载方法是否线程安全，默认False
parser.add_argument('--use_cuda',               default='True',                 type=str_to_bool,
                    help='Using cuda to accelerate calculations')  # 是否使用CUDA，默认值为'True'
parser.add_argument('--gpu_index',              default='0',                    type=str,
                    help='Index of gpu')  # 添加gpu_index参数，默认值为'0'，类型为字符串，表示要使用的GPU索引
parser.add_argument('--use_ddp',                default=False,                  action='store_true',
                    help='Use distributed data parallel during training')  # 添加use_ddp参数，默认值为False，表示是否在训练中使用分布式数据并行
parser.add_argument('--local_rank',             default=0,                      type=int,
                    help='Local device id on current node, only valid in DDP mode before torch1.9.0')  # 添加local_rank参数，默认值为0，类型为整数，表示在当前节点上的本地设备ID，仅在DDP模式下有效
parser.add_argument('--world_size',              default=1,                      type=int,
                    help='Total number of GPUs used in DDP mode')  # 添加word_size参数，默认值为1，类型为整数，表示在DDP模式下使用的GPU总数
parser.add_argument('--infer_src',              default=[],                     type=list,
                    help='Sequential pcd data director list for inference')  # 用于推理的连续PCD数据目录列表
parser.add_argument('--infer_tgt',              default='',                     type=str,
                    help='Inference output director')  # 添加infer_tgt参数，默认值为空字符串，类型为字符串，表示推理输出目录
parser.add_argument('--multi_agent', '-ma',     default=False,                  action='store_true',
                    help='Multi agent SLAM mode')  # 添加multi_agent参数，默认值为False，表示是否启用多智能体SLAM模式
parser.add_argument('--multi_thread', '-mt',    default=False,                  action='store_true',
                    help='Using multi-thread asynchronous pipeline to accelerating inference')  # 是否多线程异步，默认为False
parser.add_argument('--use_ros', '-ros',        default=False,                  action='store_true',
                    help='Inference on ros or not (experimental)')  # 添加use_ros参数，默认值为False，表示是否在ROS上进行推理（实验性功能）
parser.add_argument('--half',                   default=False,                  action='store_true',
                    help='FP16')  # 添加half参数，默认值为False，表示是否使用FP16精度

# 推荐使用配置文件导入的参数，均为dict类型
parser.add_argument('--dataset',                help='Dataset used for training or inference')  # 添加dataset参数，用于指定训练或推理使用的数据集
parser.add_argument('--transforms',             help='Data transformation methods, including preprocessing and augment')  # 添加transforms参数，用于指定数据转换方法，包括预处理和数据增强
parser.add_argument('--encoder',                help='Parameters for DMP Encoder network structure')  # 添加encoder参数，用于指定DMP编码器网络结构的参数
parser.add_argument('--decoder',                help='Parameters for DMP Decoder network structure')  # 添加decoder参数，用于指定DMP解码器网络结构的参数
parser.add_argument('--label_decoder',          help='Parameters for Label Decoder network structure')  # 添加label decoder参数
parser.add_argument('--train',                  help='Parameters for controlling the training method')  # 添加train参数，用于控制训练方法的参数
parser.add_argument('--loss',                   help='Parameters for calculating overall loss')  # 添加loss参数，用于指定计算整体损失的参数
parser.add_argument('--slam_system',            help='Parameters for full slam system, including frontend and backend')  # 添加slam_system参数，用于指定完整SLAM系统的参数，包括前端和后端

