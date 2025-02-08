import os
import sys
import yaml

from parameters import *  # 导入参数模块中的所有内容

args = parser.parse_args()  # args: 模型参数(命令行 + 默认参数)
os.environ['NUMEXPR_MAX_THREADS'] = '16'
if not args.use_ddp and args.use_cuda:  # 未使用分布式数据并行，使用CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index  # 设置CUDA可见的设备
sys.path.insert(1, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))  # 将父目录添加到Python路径中，以便模块导入

import colorlog as logging

logging.basicConfig(level=logging.INFO)  # 日志
logger = logging.getLogger(__name__)  # 日志
logger.setLevel(logging.INFO)  # 日志
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告信息

import torch
import torch.distributed as dist
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')  # 设置多进程共享策略为文件系统，以避免文件句柄共享问题

from dataloader.body import SlamDatasets
from dataloader.transforms import PointCloudTransforms
from network.encoder.encoder import Encoder
from network.decoder.decoder import Decoder
from network.loss import RegistrationLoss
from modules.model_pipeline import DeepPointModelPipeline
from modules.trainer import Trainer
from network.encoder.label_decoder import Label_Decoder


def main():
    """任务入口"""

    '''参数解析与设置'''
    global args  # 全局变量 args
    if not os.path.exists(args.yaml_file):  # 如果YAML文件不存在
        raise FileNotFoundError(f'yaml_file \'{args.yaml_file}\' is not found!')
    logger.info(f'Loading config from \'{args.yaml_file}\'...')  # 日志
    with open(args.yaml_file, 'r', encoding='utf-8') as f:  # 以只读方式打开YAML
        cfg = yaml.load(f, yaml.FullLoader)  # 加载 YAML 文件内容并解析为字典
    args = update_args(args, cfg)  # 使用YAML文件中的配置更新命令行参数
    if not args.thread_safety:  # 如果 未启用线程安全
        torch.multiprocessing.set_start_method('spawn')  # 设置 多进程启动方法为'spawn'
        logger.warning(f'The start method of torch.multiprocessing has been set to \'spawn\'')  # 日志
    if args.use_ddp and torch.cuda.is_available():
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', rank=args.local_rank, world_size=args.world_size)
        torch.cuda.set_device(args.device)
    elif args.use_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        gpus = list(range(torch.cuda.device_count()))
        torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    else:  # 如果不使用CUDA
        args.device = torch.device('cpu')  # 设备为CPU

    '''数据变换、加载数据集'''

    logger.info('正在准备数据...')  # 日志
    transforms = PointCloudTransforms(args=args, mode='train')  # 创建PointCloudTransforms对象，进行点云数据的变换处理，模式为'train'
    if args.use_ddp:  # GPU
        if args.local_rank == 0:
            dataset = SlamDatasets(args=args, data_transforms=transforms)
            print(dataset)
        dist.barrier()
        if args.local_rank != 0:
            dataset = SlamDatasets(args=args, data_transforms=transforms)
    else:  # CPU
        dataset = SlamDatasets(args=args, data_transforms=transforms)  # dataset 用于储存加载的数据集
        print(dataset)

    '''模型与损失函数'''
    logger.info('正在准备模型...')  # 记录日志
    encoder = Encoder(args=args)  # 创建Encoder对象，定义编码器网络结构
    decoder = Decoder(args=args)  # 创建Decoder对象，定义解码器网络结构
    label_decoder = Label_Decoder(**args.label_decoder)
    criterion = RegistrationLoss(args=args)  # 创建RegistrationLoss对象，定义损失函数
    model = DeepPointModelPipeline(args=args, encoder=encoder, decoder=decoder, label_decoder=label_decoder,
                                   criterion=criterion)  # 创建DeepPointModelPipeline对象，构建整个模型流水线

    '''训练流程'''
    logger.info('正在启动训练器...')  # 记录日志，提示正在启动训练器
    # 训练器
    trainer = Trainer(args=args, dataset=dataset, model=model)  # 创建Trainer对象，传入参数、数据集和模型，定义训练流程
    trainer.run()  # 运行训练流程


if __name__ == "__main__":  # 判断是否直接运行此脚本
    main()  # 如果是，则调用main函数，启动整个流程
    logger.info('Done.')  # 记录日志，提示任务完成
