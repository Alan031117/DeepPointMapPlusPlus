import os
import pickle
import random
import sys
import yaml
from parameters import *

args = parser.parse_args()
assert args.use_ddp == False
if args.use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
sys.path.insert(1, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import colorlog as logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import warnings

warnings.filterwarnings("ignore")

import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from glob import glob
import numpy as np

from system.core import SlamSystem
from dataloader.body import BasicAgent
from dataloader.transforms import PointCloudTransforms
from network.encoder.encoder import Encoder
from network.decoder.decoder import Decoder

TRIAL_EACH_FOLDER = 30
GT_DISTANCE_THRESHOLD = 25


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
        gpus = list(range(torch.cuda.device_count()))
        torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    else:
        args.device = torch.device('cpu')

    # Init models and load weights
    logger.info('Preparing model...')
    encoder = Encoder(args=args)
    decoder = Decoder(args=args)
    if (os.path.exists(args.weight) == False):
        logger.warning(f'weight file not exists: {args.weight}, model will be random initialized.')
    else:
        logger.info(f'Load weight from \'{args.weight}\'')
        weights = torch.load(args.weight, map_location='cpu')
        encoder.load_state_dict(weights['encoder'])
        decoder.load_state_dict(weights['decoder'])
        logger.info(f'Initialization completed, device = \'{args.device}\'')
    encoder = encoder.to(args.device).eval()
    decoder = decoder.to(args.device).eval()

    # Init data-transform
    logger.info('Preparing data...')

    os.makedirs(args.infer_tgt, exist_ok=True)
    if (args.train_src is None or os.path.isdir(args.train_src) == False):
        logger.warning(f'{args.train_src=}, skip training stage')
    elif (os.path.isfile(os.path.join(args.infer_tgt, 'encoder.pt')) and os.path.isfile(os.path.join(args.infer_tgt, 'decoder.pt'))):
        encoder.load_state_dict(torch.load(os.path.join(args.infer_tgt, 'encoder.pt'), map_location=args.device).to(args.device))
        decoder.load_state_dict(torch.load(os.path.join(args.infer_tgt, 'decoder.pt'), map_location=args.device).to(args.device))
        logger.warning(f'found pervious weight, loading:')
        logger.warning(os.path.join(args.infer_tgt, 'encoder.pt'))
        logger.warning(os.path.join(args.infer_tgt, 'decoder.pt'))
    else:
        decoder = decoder.train()
        sequence_dirs = glob(os.path.join(args.train_src, '*'))
        sequence_dirs = [i for i in sequence_dirs if os.path.isdir(os.path.join(i, 'pointcloud_20m_10overlap'))]
        descriptors = {}
        FINE_TUNING_EPOCH = 30
        BS = 128

        for data_root in tqdm(sequence_dirs, desc='Preparing Dir...'):
            sequence_idx = os.path.basename(data_root)
            with open(os.path.join(data_root, 'pointcloud_locations_20m_10overlap.csv'), 'r') as f:
                gt_pos = f.readlines()[1:]
            gt_pos = [i.strip().split(',') for i in gt_pos]

            gt_pos = {time_stamp: (float(northing), float(easting)) for time_stamp, northing, easting in gt_pos}
            bin_files = {time_stamp: os.path.join(data_root, 'pointcloud_20m_10overlap', f'{time_stamp}.bin') for time_stamp in gt_pos.keys()}
            if (os.path.isfile(os.path.join(data_root, 'descriptors'))):
                descriptors[sequence_idx] = pickle.load(open(os.path.join(data_root, 'descriptors'), 'rb'))
                continue
            descriptors[sequence_idx] = {}
            for i, file in tqdm(bin_files.items(), desc=f'Extracting {sequence_idx}'):
                pcd = torch.from_numpy(np.fromfile(file, dtype=np.float64).reshape(-3, 3)).float().T
                pointcloud = pcd[None].to(args.device)
                mask = torch.zeros(pointcloud.shape[2], dtype=torch.bool)[None].to(args.device)
                with torch.inference_mode():
                    coor, fea, padding = encoder(points=pointcloud, points_padding=mask)
                coor = coor * 60.0
                descriptors[sequence_idx][i] = (gt_pos[i], torch.concat([fea, coor], dim=1)[0].cpu(), padding[0])  # (fea+xyz, N), ( N)
            pickle.dump(descriptors[sequence_idx], open(os.path.join(data_root, 'descriptors'), 'wb+'))

        # optim = torch.optim.AdamW(params=decoder.loop_head.parameters(),lr=0.001,weight_decay=0.0001)
        optim = torch.optim.AdamW(params=decoder.parameters(), lr=0.00005, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=FINE_TUNING_EPOCH, eta_min=0.00001)
        loss = torch.nn.BCELoss()
        for epoch in range(FINE_TUNING_EPOCH):
            tq = tqdm(descriptors.items(), total=len(descriptors))
            loss_list = []
            for sequence_idx, sequence_dict in tq:
                values = sequence_dict.values()
                pos = torch.stack([torch.tensor(i[0]) for i in values], dim=0)  # B, 2
                des = torch.stack([i[1] for i in values], dim=0)  # B, fea+xyz, N
                pad = torch.stack([i[2] for i in values], dim=0)  # B, N

                distance_mat = torch.norm(pos.unsqueeze(0) - pos.unsqueeze(1), p=2, dim=-1)  # B, Bs
                positive_pair_indexes = torch.nonzero(distance_mat <= GT_DISTANCE_THRESHOLD)
                negitive_pair_indexes = torch.nonzero(distance_mat > GT_DISTANCE_THRESHOLD)
                assert positive_pair_indexes.shape[0] < negitive_pair_indexes.shape[0]

                random_index_pos = torch.randperm(n=positive_pair_indexes.shape[0])[:int(positive_pair_indexes.shape[0] // (BS // 2) * (BS // 2))]
                random_index_neg = torch.randperm(n=negitive_pair_indexes.shape[0])[:int(negitive_pair_indexes.shape[0] // (BS // 2) * (BS // 2))]
                for i in range(len(sequence_dict) // (BS // 2)):
                    pos_iindexs = random_index_pos[i * (BS // 2):i * (BS // 2) + (BS // 2)]
                    pos_index = positive_pair_indexes[pos_iindexs, :]  # bs//2, ij
                    neg_iindexs = random_index_neg[i * (BS // 2):i * (BS // 2) + (BS // 2)]
                    neg_index = negitive_pair_indexes[neg_iindexs, :]  # bs//2, ij

                    batch_a = torch.concat([pos_index[:, 0], neg_index[:, 0]])
                    pos_a, des_a, pad_a = pos[batch_a, :], des[batch_a, :, :], pad[batch_a, :]  # Bs, 2 | # Bs, fea+xyz, N | Bs, N

                    batch_b = torch.concat([pos_index[:, 1], neg_index[:, 1]])
                    pos_b, des_b, pad_b = pos[batch_b, :], des[batch_b, :, :], pad[batch_b, :]  # Bs, 2 | # Bs, fea+xyz, N | Bs, N

                    gt_label = (torch.norm(pos_a - pos_b, p=2, dim=-1) <= GT_DISTANCE_THRESHOLD).float().to(args.device)  # Bs,
                    loop_prob = decoder.loop_detection_forward(src_descriptor=des_a.to(args.device),
                                                               dst_descriptor=des_b.to(args.device),
                                                               src_padding_mask=pad_a.to(args.device),
                                                               dst_padding_mask=pad_b.to(args.device))

                    optim.zero_grad()
                    l = loss(loop_prob, gt_label)
                    l.backward()
                    optim.step()

                    loss_list.append(l.item())
                    tq.set_description(f'epoch = {epoch}, loss = {sum(loss_list)/len(loss_list):.3f}, lr = {scheduler.get_lr()[0]:.8f}')
            scheduler.step()
        torch.save(encoder.state_dict(), os.path.join(args.infer_tgt, 'encoder.pt'))
        torch.save(decoder.state_dict(), os.path.join(args.infer_tgt, 'decoder.pt'))
    decoder = decoder.eval()
    sequence_dirs = glob(os.path.join(args.infer_src, '*'))
    sequence_dirs = [i for i in sequence_dirs if os.path.isdir(os.path.join(i, 'pointcloud_20m'))]

    result_csv = open(os.path.join(args.infer_tgt, f'result.csv'), 'w+')
    # For each sequence...
    result_list = []
    for i, data_root in enumerate(sequence_dirs):
        with open(os.path.join(data_root, 'pointcloud_locations_20m.csv'), 'r') as f:
            gt_pos = f.readlines()[1:]
        gt_pos = [i.strip().split(',') for i in gt_pos]
        gt_pos = {time_stamp: (float(northing), float(easting)) for time_stamp, northing, easting in gt_pos}
        pcds = {i: torch.from_numpy(np.fromfile(os.path.join(data_root, 'pointcloud_20m', f'{i}.bin'), dtype=np.float64).reshape(-3, 3)).float().T for i in gt_pos.keys()}
        descriptors = {}

        for i, pcd in tqdm(pcds.items(), desc='Extracting...'):
            pointcloud = pcd[None].to(args.device)
            mask = torch.zeros(pointcloud.shape[2], dtype=torch.bool)[None].to(args.device)
            with torch.inference_mode():
                coor, fea, padding = encoder(points=pointcloud, points_padding=mask)
            coor = coor * 60.0
            descriptors[i] = torch.concat([fea, coor], dim=1)[0].cpu()  # (fea+xyz, N)

        bins = list(pcds.keys())
        random.shuffle(bins)

        for i in bins[:TRIAL_EACH_FOLDER]:
            src_descriptor, src_gt_pos = descriptors[i], gt_pos[i]
            tq = tqdm(bins, leave=False, desc=f'Trial {i}')
            query_result = []
            for j in tq:
                dst_descriptor, dst_gt_pos = descriptors[j], gt_pos[j]
                with torch.inference_mode():
                    loop_prob = decoder.loop_detection_forward(src_descriptor=src_descriptor.to(args.device), dst_descriptor=dst_descriptor.to(args.device))
                loop_prob = loop_prob.item()
                gt_distance = ((src_gt_pos[0] - dst_gt_pos[0])**2 + (src_gt_pos[1] - dst_gt_pos[1])**2)**0.5

                query_result.append((loop_prob, gt_distance))
                tq.set_description_str(f'{i} <-> {j} | gt_dist = {gt_distance:>5.2f} prob = {loop_prob:.2f}')
            query_result = sorted(query_result, key=lambda x: x[0], reverse=True)
            result_list.append([os.path.basename(data_root), query_result.copy()])
            query_result = query_result[:int(len(query_result) / 100)]
            query_result = list(filter(lambda x: (x[1] < GT_DISTANCE_THRESHOLD), query_result))
            top_1percent_recall = len([prob for prob, distance in query_result if (prob > args.slam_system.loop_detection_confidence_acpt_threshold)]) / len(query_result)
            print(f'Recall = {top_1percent_recall}')
            result_csv.write(f'Recall = {top_1percent_recall:.5f}, seq = {os.path.basename(data_root)}\n')
    pickle.dump(result_list, open(os.path.join(args.infer_tgt, f'result_list.pkl'), 'wb+'))
    result_csv.close()


if __name__ == "__main__":
    main()
    logger.info('Done.')
