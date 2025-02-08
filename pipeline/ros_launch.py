import os
import sys
import yaml
from parameters import *

args = parser.parse_args()
if args.use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(1, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import colorlog as logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import warnings

warnings.filterwarnings("ignore")

import torch
import numpy as np

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

# import ros (start)
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
# import ros (end)

from system.core import SlamSystem
from dataloader.transforms import PointCloudTransforms, PointCloud
from network.encoder.encoder import Encoder
from network.decoder.decoder import Decoder



def main():
    # Load yaml and prepare platform
    global args
    if not os.path.exists(args.yaml_file):
        raise FileNotFoundError(f'yaml_file is not found: {args.yaml_file}')
    logger.info(f'Loading config from \'{args.yaml_file}\'...')
    with open(args.yaml_file, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, yaml.FullLoader)
    args = update_args(args, cfg)
    if not args.thread_safety:
        torch.multiprocessing.set_start_method('spawn')
        logger.warning(f'The start method of torch.multiprocessing has been set to \'spawn\'')
    if args.use_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.cuda.set_device('cuda:0')
    else:
        args.device = torch.device('cpu')
    args.use_ros = True

    # Create result dir and save yaml
    save_root = os.path.join(args.infer_tgt)
    os.makedirs(save_root, exist_ok=True)
    with open(os.path.join(save_root, 'settings.yaml'), 'w+', encoding='utf-8') as arg_file:
        args_dict = sorted(args._get_kwargs())
        for k, v in args_dict:
            arg_file.write(f'{k}: {v}\n')

    # Init data-loaders and data-transform
    logger.info('Preparing data...')
    transforms = PointCloudTransforms(args=args, mode='infer')

    # Init models and load weights
    logger.info('Preparing model...')
    encoder = Encoder(args=args)
    decoder = Decoder(args=args)
    if(os.path.exists(args.weight) == False):
        logger.warning(f'weight file not exists: {args.weight}, model will be random initialized.')
    else:
        logger.info(f'Load weight from \'{args.weight}\'')
        weights = torch.load(args.weight, map_location='cpu')
        encoder.load_state_dict(weights['encoder'])
        decoder.load_state_dict(weights['decoder'])
        logger.info(f'Initialization completed, device = \'{args.device}\'')


    # init rospy and slam_system
    rospy.init_node('DeepPointMap')
    slam_system = SlamSystem(args=args, dpm_encoder=encoder, dpm_decoder=decoder, system_id=0, logger_dir=save_root)
    ros_scan_drop_counter = -1
    def on_lidar_scan(msg):
        nonlocal ros_scan_drop_counter
        ros_scan_drop_counter += 1
        if (ros_scan_drop_counter % 5 != 0):
            return
        points = np.asarray(list(point_cloud2.read_points(msg)), dtype=np.float32)  # (N, xyz+), contains nan
        points = points[np.isnan(points).sum(1) == 0][:, :3] # (N, xyz), without any nan
        data = transforms(PointCloud(xyz=points))  # data = [point_cloud, R, T, padding_mask, original_scan], all type = torch.Tensor
        for i in range(len(data)):
            data[0].unsqueeze_(i)
        slam_system.MT_Step(data)
        return

    # Feed data! (Multi-Thread / MT)
    slam_system.MT_Init()
    try:
        rospy.Subscriber(args.infer_src.lidar_topic, PointCloud2, callback=on_lidar_scan)
        rospy.spin()
    except:
        logger.critical('ROS node shutdown, closing system...')
        slam_system.MT_Done()
        slam_system.MT_Wait()

    slam_system.result_logger.save_trajectory('trajectory')
    slam_system.result_logger.save_posegraph('trajectory')
    slam_system.result_logger.draw_trajectory('trajectory', draft=False)
    slam_system.result_logger.save_map('trajectory')


if __name__ == "__main__":
    main()
    logger.info('Done.')
