# raise RuntimeError(f'THIS SCRIPT HAS A BUG (COORDINATE X-Z-Y), DO NOT USE')
from pyquaternion import Quaternion
import numpy as np
import os
import open3d as o3d
import torchvision
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import sys
import shutil
from glob import glob


KITTI_VERSION = 'v4.0'
print(f'{sys.platform=}')
if sys.platform == 'linux':
    DATA_ROOT = r'/root/dataset/KITTI_CARLA'
    POSE_ROOT = r'/root/dataset/KITTI_CARLA'
    CALIB_ROOT = r'/root/dataset/KITTI_CARLA'
    OUTPUT_ROOT = r'/root/dataset/KITTICarla_v4.0_NoVisual'
else:
    DATA_ROOT = r'E:\Dataset\original_KITTICarla_Mini'
    IMAGE_ROOT = r'E:\Dataset\original_KITTICarla_Mini'
    POSE_ROOT = r'E:\Dataset\original_KITTICarla_Mini'
    CALIB_ROOT = r'E:\Dataset\original_KITTICarla_Mini'
    OUTPUT_ROOT = r'E:\Dataset\KITTICarla_v4.0_NoVisual'

print(f'Load original KITTI-Carla from {os.path.abspath(DATA_ROOT)}')
print(f'Will create new KITTI-Carla dataset at {os.path.abspath(OUTPUT_ROOT)}')

scene_path = glob(os.path.join(DATA_ROOT, 'Town*'))
print(f'find {len(scene_path)} scenes: {scene_path}')
# while (input('press Enter to continue, use ctrl+C to abort:') != ''):
#     pass


def get_label(file_path) -> np.ndarray:
    label = (np.fromfile(file_path, dtype=np.uint32) & 0xFFFF)[:, np.newaxis]  # (N, )
    return label


def get_lidar(file_path):
    pcd = o3d.io.read_point_cloud(file_path)

    points = np.array(pcd.points)  # (N,3)
    return points


def show_pcd(pcds, colors=None, window_name="PCD", normals=False):
    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name=window_name)
    for i in range(len(pcds)):
        # 创建点云对象
        pcd_o3d = o3d.open3d.geometry.PointCloud()
        # 将点云数据转换为Open3d可以直接使用的数据类型
        if (isinstance(pcds[i], np.ndarray)):
            pcd_points = pcds[i][:, :3]
        elif (isinstance(pcds[i], torch.Tensor)):
            pcd_points = pcds[i][:, :3].detach().cpu().numpy()
        else:
            pcd_points = np.array(pcds[i][:, :3])
        pcd_o3d.points = o3d.open3d.utility.Vector3dVector(pcd_points)
        # pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
        # 设置点的颜色
        if colors is not None:
            pcd_o3d.paint_uniform_color(colors[i])
        # 将点云加入到窗口中
        vis.add_geometry(pcd_o3d)

    vis.run()
    vis.destroy_window()


# MIN_PCD_DIST = 0.002


class PoseTool(object):
    @classmethod
    def SE3(cls, R, t):
        if (isinstance(R, np.ndarray)):
            R = torch.tensor(R, dtype=torch.float32).reshape(3, 3)
        if (isinstance(t, np.ndarray)):
            t = torch.tensor(t, dtype=torch.float32).reshape(3, 1)
        mat = torch.eye(4)
        mat[:3, :3] = R
        mat[:3, 3:4] = t
        return mat

    @classmethod
    def Rt(cls, SE3):
        '''
        R: torch.Tensor(3, 3)
        t: torch.Tensor(3, 1)
        '''
        R = SE3[:3, :3]
        t = SE3[:3, 3:]
        return (R, t)


def make_npz_warpper(frame_id, scene_root):
    sample_file = os.path.join(scene_path, 'generated', 'frames', f'frame_{frame_id:04d}.ply')
    if (os.path.exists(sample_file) == False):
        return

    lidar = torch.from_numpy(get_lidar(sample_file)).T  # 3, N
    velodyne_to_cam_transform = cam_to_velo.inverse().float()

    # images, lidar_proj = make_visual_info(df)
    # images = images[np.newaxis, :]

    # new_pose_SE3 = velodyne_to_cam_transform.inverse() @ cam00_pose @ velodyne_to_cam_transform
    new_pose_SE3 = dG @ cam00_pose @ velodyne_to_cam_transform
    pose_r, pose_t = PoseTool.Rt(new_pose_SE3)

    traj_show_list.append(new_pose_SE3)  # 3, 1
    # pcd_show_list.append((pose_r@lidar+ pose_t).T)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(lidar.T.numpy())
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=50))
    # lidarnorm = np.asanyarray(pcd.normals)

    if (frame_id % 500 == 0):
        plt.figure(figsize=(20, 20), dpi=200)

        plt.subplot(221)
        plt.axis('equal')
        plt.title('Point Cloud')
        plt.scatter(lidar[0, :], lidar[1, :], c=lidar[2, :], s=0.1)

        # if (USE_VISUAL):
        #     plt.subplot(222)
        #     plt.title('Image Proj')
        #     c, h, w = images.shape
        #     plt.imshow(images.transpose(1, 2, 0))
        #     vis_mask = lidar_proj[2, :] > 0  # N,
        #     plt.scatter(lidar_proj[0, vis_mask] * w, lidar_proj[1, vis_mask] * h, c=lidar_proj[2, vis_mask], s=0.1)

        #     plt.subplot(223)
        #     plt.axis('equal')
        #     plt.title('Points with Visual')
        #     plt.scatter(lidar[0, vis_mask], lidar[1, vis_mask], c=lidar[2, vis_mask], s=0.1)

        plt.subplot(224)
        plt.axis('equal')
        plt.title('Current Traj')
        traj_show_list_np = np.stack(traj_show_list, axis=0)
        plt.plot(traj_show_list_np[:, 0, 3], traj_show_list_np[:, 1, 3])
        plt.scatter(traj_show_list_np[-1, 0, 3], traj_show_list_np[-1, 1, 3], s=1, c='red')

        plt.tight_layout()
        plt.savefig(os.path.join(check_root, f'{str(frame_id).zfill(2)}_{frame_id}_check.png'))
        plt.close()

    np.savez(
        os.path.join(scene_root, str(frame_id)),
        lidar_pcd=lidar.T.numpy().astype(np.float32),  # N, 3
        # lidar_seg=label.astype(np.int64),  # N, 1
        # lidar_norm=lidarnorm.astype(np.float32),  # N, 3
        # lidar_proj=lidar_proj.astype(np.float32),  # (N, 4) -> [img_id, u, v]
        # image=images.astype(np.uint8),  # (1, 3, H, W)
        ego_rotation=pose_r.numpy().astype(np.float32),  # 3, 3
        ego_translation=pose_t.numpy().astype(np.float32),  # 3, 1
        time_stamp = float(time_stamp),
    )

dG = torch.tensor([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float()

scene_tq = tqdm(scene_path, desc='loading scenes...')
for scene_path in scene_tq:
    pose_file = os.path.join(scene_path, 'generated', 'full_poses_lidar.txt')
    calib_file = os.path.join(scene_path, 'generated', 'lidar_to_cam0.txt')
    scene_name = os.path.split(scene_path)[-1]
    poses = []
    frame_id = -1
    for l in tqdm(open(pose_file, 'r'),leave=False,desc='reading pose'):
        if l.startswith('#'):
            continue
        frame_id += 1
        if (frame_id % 100 == 0):
            line = l.strip().split(' ')
            cam00_pose = torch.eye(4)
            cam00_pose[:3, :] = torch.tensor([float(i) for i in line[0:-1]]).view(3, 4)
            time_stamp = float(line[-1])
            poses.append((frame_id//100, cam00_pose, time_stamp))

    sample_tq = tqdm(poses, desc=f'loading samples in scene {scene_path}')

    scene_root = os.path.join(OUTPUT_ROOT, scene_name, '0')
    check_root = os.path.join(OUTPUT_ROOT, f'_check', str(scene_name))
    if (os.path.exists(scene_root)):
        # shutil.rmtree(scene_root)
        print(f'passing dir {scene_root}')
        continue
    os.makedirs(scene_root, exist_ok=True)
    os.makedirs(check_root, exist_ok=True)

    cam_to_velo = torch.eye(4)
    f = open(calib_file, 'r').readlines()
    f = f[1].strip().split(' ')
    f = [float(i) for i in f]
    cam_to_velo[:3, :] = torch.tensor(f).view(3, 4)

    traj_show_list = []
    pcd_show_list = []

    for frame_id, cam00_pose, time_stamp in sample_tq:
        make_npz_warpper(frame_id, scene_root)
    traj_show_list = torch.stack(traj_show_list)

    plt.figure(figsize=(20, 20))
    plt.title(f'Traj {scene_name}')
    plt.axis('equal')
    plt.plot(traj_show_list[:, 0, :], traj_show_list[:, 1, :])
    plt.scatter(traj_show_list[0, 0, :], traj_show_list[0, 1, :], marker='s', label='start')
    plt.scatter(traj_show_list[-1, 0, :], traj_show_list[-1, 1, :], marker='o', label='end')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(check_root, f'Traj {scene_name}.png'))
    plt.close()

# T @ PCD =
