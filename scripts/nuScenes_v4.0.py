# %%
import sys
import numpy as np
import open3d as o3d
import nuscenes
from pyquaternion import Quaternion
import struct
import os
from os.path import join as ospj
import pickle
import torchvision
import torch
from tqdm import tqdm

FAST_NUSCENES_VERSION = 'v4.0'

if sys.platform == 'linux':
    DATA_ROOT = r'/root/dataset/original_nuScenes'
    OUTPUT_ROOT = r'/root/dataset/nuScenes_v4.0_Visual'
else:
    DATA_ROOT = r'E:\original_nuScenes'
    OUTPUT_ROOT = r'E:/nuScenes_v4.0_Visual'





print(f'Load original nuScenes from {os.path.abspath(DATA_ROOT)}')
print(f'Will create new nuScenes dataset at {os.path.abspath(OUTPUT_ROOT)}')

os.makedirs(OUTPUT_ROOT, exist_ok=True)
#nusc = nuscenes.NuScenes(version='v1.0-mini', dataroot=DATA_ROOT)
nusc = nuscenes.NuScenes(version='v1.0-trainval', dataroot=DATA_ROOT)
sceneList = nusc.scene
print(f'load {len(sceneList)} scenes')


class Pose(object):
    def __init__(self, pos_xyz=None, rot=None, rot_type='quat') -> None:
        self.pos = np.array(pos_xyz).reshape(3, 1)
        if (rot_type.lower() == 'quat'):
            self.rot = Quaternion(np.array(rot).reshape(4)).rotation_matrix
        elif (rot_type.lower() == 'rot_mat'):
            self.rot = np.array(rot).reshape(3, 3)
        else:
            raise RuntimeError(f"rot para in Pose class not defined. found {rot_type}")
        pass

    @property
    def position_xyz(self):
        '''
        get 3x3 rotation matrix
        '''
        return self.pos

    @property
    def rotation_mat(self):
        '''
        get 3x1 translation vector
        '''
        return self.rot

    def __add__(self, other):
        '''
        Apply two transf in order

        Q = R2 x (R1 x Q + t1) + t2
          = R2 x R1 x Q + R2 x t1 + t2
            ^~~~~~~       ^~~~~~~~~~~~
            New R         New t
        '''
        rot = other.rotation_mat @ self.rotation_mat.copy()
        pos = other.rotation_mat @ self.position_xyz.copy() + other.pos
        ret = Pose(pos, rot, rot_type='rot_mat')
        return ret

    def inv(self):
        rot = self.rotation_mat.copy().T
        pos = -(rot @ self.position_xyz.copy())
        ret = Pose(pos, rot, rot_type='rot_mat')
        return ret

    def homogeneous(self):
        '''
        Return 4x4 homogeneous transf matrix
        '''
        ret = np.eye(4)
        ret[:3, :3] = self.rotation_mat
        ret[:3, 3] = self.position_xyz.flatten()
        return ret


class DataFrame:
    LIDAR_COORDINATE_CHOICE = ['sensor', 'ego', 'global']
    RADAR_COORDINATE_CHOICE = ['ego', 'global']
    BBOX_COORDINATE_CHOICE = ['ego', 'global']

    def __init__(self, sample_token: str, data_root: str, nusc):
        self.SEG_INDEX_LABEL_MAP = {c['index']: c['name'] for c in nusc.category}
        self.SEG_LABEL_INDEX_MAP = {c['name']: c['index'] for c in nusc.category}

        self.nusc = nusc
        self.sample = self.nusc.get('sample', sample_token)
        self.data_root = data_root
        '''Camera Data'''
        self.cam_extrinsics, self.cam_intrinsics = [], []

        self.CAM_FRONT = self.nusc.get('sample_data', self.sample['data']['CAM_FRONT'])
        extri, intri = self._ReadCameraIntrinsicExtrinsic(self.CAM_FRONT['calibrated_sensor_token'])
        self.cam_extrinsics.append(extri)
        self.cam_intrinsics.append(intri)

        self.CAM_FRONT_LEFT = self.nusc.get('sample_data', self.sample['data']['CAM_FRONT_LEFT'])
        extri, intri = self._ReadCameraIntrinsicExtrinsic(self.CAM_FRONT_LEFT['calibrated_sensor_token'])
        self.cam_extrinsics.append(extri)
        self.cam_intrinsics.append(intri)

        self.CAM_BACK_LEFT = self.nusc.get('sample_data', self.sample['data']['CAM_BACK_LEFT'])
        extri, intri = self._ReadCameraIntrinsicExtrinsic(self.CAM_BACK_LEFT['calibrated_sensor_token'])
        self.cam_extrinsics.append(extri)
        self.cam_intrinsics.append(intri)

        self.CAM_BACK = self.nusc.get('sample_data', self.sample['data']['CAM_BACK'])
        extri, intri = self._ReadCameraIntrinsicExtrinsic(self.CAM_BACK['calibrated_sensor_token'])
        self.cam_extrinsics.append(extri)
        self.cam_intrinsics.append(intri)

        self.CAM_BACK_RIGHT = self.nusc.get('sample_data', self.sample['data']['CAM_BACK_RIGHT'])
        extri, intri = self._ReadCameraIntrinsicExtrinsic(self.CAM_BACK_RIGHT['calibrated_sensor_token'])
        self.cam_extrinsics.append(extri)
        self.cam_intrinsics.append(intri)

        self.CAM_FRONT_RIGHT = self.nusc.get('sample_data', self.sample['data']['CAM_FRONT_RIGHT'])
        extri, intri = self._ReadCameraIntrinsicExtrinsic(self.CAM_FRONT_RIGHT['calibrated_sensor_token'])
        self.cam_extrinsics.append(extri)
        self.cam_intrinsics.append(intri)

        self.cam_extrinsics = np.stack(self.cam_extrinsics).astype(np.float32)
        self.cam_intrinsics = np.stack(self.cam_intrinsics).astype(np.float32)
        '''Lidar Data'''
        self.LIDAR_TOP = nusc.get('sample_data', self.sample['data']['LIDAR_TOP'])
        '''Radar Data'''
        self.RADAR_BACK_LEFT = nusc.get('sample_data', self.sample['data']['RADAR_BACK_LEFT'])
        self.RADAR_BACK_RIGHT = nusc.get('sample_data', self.sample['data']['RADAR_BACK_RIGHT'])
        self.RADAR_FRONT = nusc.get('sample_data', self.sample['data']['RADAR_FRONT'])
        self.RADAR_FRONT_LEFT = nusc.get('sample_data', self.sample['data']['RADAR_FRONT_LEFT'])
        self.RADAR_FRONT_RIGHT = nusc.get('sample_data', self.sample['data']['RADAR_FRONT_RIGHT'])
        '''Annos'''
        self.bounding_boxs = [nusc.get('sample_annotation', anno_token) for anno_token in self.sample['anns']]
        self.lidar_seg = nusc.get('lidarseg', self.LIDAR_TOP['token'])
        '''Calculate timestamp (in us, not SECOND)'''
        timestamps = [
            self.CAM_BACK['timestamp'], self.CAM_BACK_LEFT['timestamp'], self.CAM_BACK_RIGHT['timestamp'], self.CAM_FRONT['timestamp'], self.CAM_FRONT_LEFT['timestamp'],
            self.CAM_FRONT_RIGHT['timestamp'], self.LIDAR_TOP['timestamp'], self.RADAR_BACK_LEFT['timestamp'], self.RADAR_BACK_RIGHT['timestamp'], self.RADAR_FRONT['timestamp'],
            self.RADAR_FRONT_LEFT['timestamp'], self.RADAR_FRONT_RIGHT['timestamp']
        ]
        self.timestamp = np.mean(timestamps)
        self.timestamp_std = np.std(timestamps)
        '''Self pose (use pos information of lidar)'''
        ego_pose = nusc.get('ego_pose', nusc.get('sample_data', self.sample['data']['LIDAR_TOP'])['ego_pose_token'])
        self.ego_translation = ego_pose['translation']  # delete soon
        self.ego_rotation = ego_pose['rotation']  # delete soon
        self.ego_pose = Pose(ego_pose['translation'], ego_pose['rotation'])

    def get_lidar(self, coordinate: str = 'ego', label_list: list = None, use_image_color=False) -> tuple:
        assert coordinate in DataFrame.LIDAR_COORDINATE_CHOICE, \
            f'Coordinate {coordinate} not implemented, choice: {self.LIDAR_COORDINATE_CHOICE}.'
        lidar = self._ReadVelodynePoindcloud(os.path.join(self.data_root, self.LIDAR_TOP['filename']))
        label = np.fromfile(os.path.join(self.data_root, self.lidar_seg['filename']), dtype=np.uint8)
        # filter out unchosen label points
        if label_list is not None:
            mask = np.zeros_like(label, dtype=np.bool8)
            label_in_seg = set([self.SEG_LABEL_INDEX_MAP[cate] for cate in label_list])  # building
            for l in label_in_seg:
                mask |= (label == l)
            lidar = lidar[mask, :]
            label = label[mask]

        # apply coordinate
        if coordinate == 'sensor':
            return lidar, label[:, np.newaxis]

        sensor_coord = self.nusc.get('calibrated_sensor', self.LIDAR_TOP['calibrated_sensor_token'])
        lidar_ego = self._ApplyRotationTranslation(lidar, sensor_coord['translation'], sensor_coord['rotation'])
        if coordinate == 'ego':
            return lidar_ego, label[:, np.newaxis]

        lidar_global = self._ApplyRotationTranslation(lidar_ego, self.ego_translation, self.ego_rotation)
        if coordinate == 'global':
            return lidar_global, label[:, np.newaxis]

    def get_radar_fused(self, coordinate: str = 'ego') -> np.ndarray:
        assert coordinate in DataFrame.RADAR_COORDINATE_CHOICE, \
            f'Coordinate {coordinate} not implemented, choice: {self.RADAR_COORDINATE_CHOICE}.'

        radar_f = np.asarray(o3d.io.read_point_cloud(os.path.join(self.data_root, self.RADAR_FRONT['filename'])).points)
        radar_f_coord = self.nusc.get('calibrated_sensor', self.RADAR_FRONT['calibrated_sensor_token'])
        ego_radar_f = self._ApplyRotationTranslation(radar_f, radar_f_coord['translation'], radar_f_coord['rotation'])

        radar_fl = np.asarray(o3d.io.read_point_cloud(os.path.join(self.data_root, self.RADAR_FRONT_LEFT['filename'])).points)
        radar_fl_coord = self.nusc.get('calibrated_sensor', self.RADAR_FRONT_LEFT['calibrated_sensor_token'])
        ego_radar_fl = self._ApplyRotationTranslation(radar_fl, radar_fl_coord['translation'], radar_fl_coord['rotation'])

        radar_fr = np.asarray(o3d.io.read_point_cloud(os.path.join(self.data_root, self.RADAR_FRONT_RIGHT['filename'])).points)
        radar_fr_coord = self.nusc.get('calibrated_sensor', self.RADAR_FRONT_RIGHT['calibrated_sensor_token'])
        ego_radar_fr = self._ApplyRotationTranslation(radar_fr, radar_fr_coord['translation'], radar_fr_coord['rotation'])

        radar_bl = np.asarray(o3d.io.read_point_cloud(os.path.join(self.data_root, self.RADAR_BACK_LEFT['filename'])).points)
        radar_bl_coord = self.nusc.get('calibrated_sensor', self.RADAR_BACK_LEFT['calibrated_sensor_token'])
        ego_radar_bl = self._ApplyRotationTranslation(radar_bl, radar_bl_coord['translation'], radar_bl_coord['rotation'])

        radar_br = np.asarray(o3d.io.read_point_cloud(os.path.join(self.data_root, self.RADAR_BACK_RIGHT['filename'])).points)
        radar_br_coord = self.nusc.get('calibrated_sensor', self.RADAR_BACK_RIGHT['calibrated_sensor_token'])
        ego_radar_br = self._ApplyRotationTranslation(radar_br, radar_br_coord['translation'], radar_br_coord['rotation'])

        ego_radar_fuse = np.concatenate([ego_radar_f, ego_radar_fl, ego_radar_fr, ego_radar_bl, ego_radar_br])

        if coordinate == 'ego':
            return ego_radar_fuse
        if coordinate == 'global':
            glo_radar_fuse = self._ApplyRotationTranslation(ego_radar_fuse, self.ego_translation, self.ego_rotation)
            return glo_radar_fuse

    def get_ego_global_pose(self):
        trans = self.ego_translation
        rot = self.ego_rotation
        return MakeTransFormList(trans, rot)

    def get_bbox(self, coordinate: str = 'ego') -> list:
        assert coordinate in DataFrame.BBOX_COORDINATE_CHOICE, \
            f'Coordinate {coordinate} not implemented, choice: {self.BBOX_COORDINATE_CHOICE}.'
        '''Bbox format: (x,y,z, )'''
        if coordinate == 'global':
            return self.bounding_boxs
        elif coordinate == 'ego':
            raise NotImplementedError('get_bbox() not implemented yet')

    def get_images(self):
        """
        get 6 camera image of this data frame
        shape [c, h, w], scaled [0, 1]    
        """
        # imgs = []
        img = torchvision.io.read_image(os.path.join(self.data_root, self.CAM_FRONT['filename']))  # [c, h, w]
        # imgs.append(torchvision.io.read_image(os.path.join(self.data_root, self.CAM_FRONT_LEFT['filename'])))  # [h, w, c]
        # imgs.append(torchvision.io.read_image(os.path.join(self.data_root, self.CAM_BACK_LEFT['filename'])))  # [h, w, c]
        # imgs.append(torchvision.io.read_image(os.path.join(self.data_root, self.CAM_BACK['filename'])))  # [h, w, c]
        # imgs.append(torchvision.io.read_image(os.path.join(self.data_root, self.CAM_BACK_RIGHT['filename'])))  # [h, w, c]
        # imgs.append(torchvision.io.read_image(os.path.join(self.data_root, self.CAM_FRONT_RIGHT['filename'])))  # [h, w, c]
        # imgs = torch.stack(imgs)  # [6, c, h, w]
        return img / 255.0

    def _ReadVelodynePoindcloud(self, path: str):
        pc_list = []
        with open(path, 'rb') as f:
            content = f.read()
            pc_iter = struct.iter_unpack('fffff', content)
            for idx, point in enumerate(pc_iter):
                pc_list.append([point[0], point[1], point[2]])
        return np.asarray(pc_list, dtype=np.float32)

    def _ReadLidarSegAnnotation(self, path: str):
        return np.fromfile(path, dtype=np.uint8)

    def _ReadCameraIntrinsicExtrinsic(self, cam_calibrated_token):
        '''
        get intrinsic and extrinsic matric of camera 
        cam_calibrated_token:str, calibrated token of camera
        return extrinsic, intrinsic
        '''
        cam_calibrated = self.nusc.get('calibrated_sensor', cam_calibrated_token)
        I = np.array(cam_calibrated['camera_intrinsic']).astype(np.float32)
        T = np.array(cam_calibrated['translation']).astype(np.float32)
        R = np.array(Quaternion(cam_calibrated['rotation']).rotation_matrix).astype(np.float32)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R.T
        extrinsic[:3, 3] = -R.T @ T
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = I
        return extrinsic, intrinsic

    def _ApplyRotationTranslation(self, pcd: np.ndarray, translation_offset: np.ndarray, rotation_quaternion: np.ndarray):
        rotate_matrix = Quaternion(np.array(rotation_quaternion)).rotation_matrix
        # ! pch with shape (n,3) NOT (3,n)
        # ! A' = (R x A^T)^T = A x R^T
        return np.matmul(pcd, rotate_matrix.T) + np.array(translation_offset)


# %%
def show_pcd(pcds, colors=None, window_name="PCD", normals=False):
    '''
    pcds: List(ArrayLike), points to be shown, shape (K, xyz+)
    colors: List[Tuple], color list, shape (r,g,b) scaled 0~1
    '''
    import open3d as o3d
    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name=window_name)
    # 设置点云大小
    # vis.get_render_option().point_size = 1
    # 设置颜色背景为黑色
    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([0, 0, 0])

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


# %% [markdown]
# ## Visual Info

# %%
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_classes import Box
import matplotlib.pyplot as plt
from nuscenes.utils.geometry_utils import view_points

MIN_PCD_DIST = 2

frame_token = sceneList[0]['first_sample_token']
df = DataFrame(frame_token, DATA_ROOT, nusc)


def make_visual_info(df):

    images = df.get_images()  # [2, c, h, w]
    points = df.get_lidar()[0].T  # (3,N)
    points = np.concatenate((points, np.ones((1, points.shape[1]))))  # (4,N)

    cam_in_left = torch.from_numpy(df.cam_intrinsics).float()[0, :, :]  # (4,4)
    velodyne_to_cam = torch.from_numpy(df.cam_extrinsics).float()[0, :, :]  # (4,4)

    # 点云转换至左相机坐标系
    pcd_cam = cam_in_left @ velodyne_to_cam @ points  # (4(x,y,k,l),N)
    depth = pcd_cam[2, :]
    pcd_cam = pcd_cam / pcd_cam[3, :] / depth  # 点的坐标归一化为(u,v,1,1)
    pcd_cam[2, :] = depth  # 恢复点的深度 (u,v,d,1)

    pcd_cam = pcd_cam[:3, :]  # uvd, N

    # 丢弃超出视野范围外的点
    vis_mask = torch.ones(depth.shape, dtype=torch.bool)  # N,1

    vis_mask &= depth > 0  # 丢弃摄像机后方的点
    vis_mask &= pcd_cam[0, :] > 1
    vis_mask &= pcd_cam[0, :] < images.shape[2] - 1
    vis_mask &= pcd_cam[1, :] > 1
    vis_mask &= pcd_cam[1, :] < images.shape[1] - 1

    pcd_cam[:, ~vis_mask] = 0
    pcd_cam[2, ~vis_mask] = -1

    # u,v坐标归一化
    C, H, W = images.shape
    pcd_cam[0, :] /= W
    pcd_cam[1, :] /= H

    return images, pcd_cam


# %% [markdown]
# ## Target Info

# %%
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_classes import Box
import matplotlib.pyplot as plt

frame_token = sceneList[0]['first_sample_token']

frame = DataFrame(frame_token, DATA_ROOT, nusc)


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


def make_target_list(frame, min_pcd_points=10):
    lidar, lidarseg = frame.get_lidar()  # (34688, 3)
    lidar = lidar.astype(np.float32)
    lidarseg = lidarseg.astype(np.float32)

    R = frame.ego_pose.rotation_mat
    T = frame.ego_pose.position_xyz
    inv_R = R.T
    inv_T = -R.T @ T

    glo_lidar = (R @ lidar.T + T).T  # (34688, 3)
    dets = []

    for target_idx in range(len(frame.bounding_boxs)):
        det = frame.bounding_boxs[target_idx]
        box = nusc.get_box(det['token'])
        if (int(det['num_lidar_pts']) < min_pcd_points):
            continue

        det_token = det['instance_token']  # str
        det_label = det['category_name']  # str
        det_vis = det['visibility_token']  # str
        det_center = box.center[:, np.newaxis]  # 3, 1
        det_rot = box.orientation.rotation_matrix  # 3, 3
        det_wlh = box.wlh[:, np.newaxis]  # 3, 1
        det_lwh = np.array([det_wlh[1, :], det_wlh[0, :], det_wlh[2, :]])
        del det_wlh

        det_lidar_pcd_glo = glo_lidar[points_in_box(box, glo_lidar.T), :]  # (34688, 3)

        det_center_ego = inv_R @ det_center + inv_T  # 3, 1 - ego
        det_rot_ego = inv_R @ det_rot  # 3, 3 - ego
        det_lidar_pcd_ego = (inv_R @ det_lidar_pcd_glo.T + inv_T).T  # (34688, 3)
        det_lidar_pcd_local = ((det_rot_ego.T @ det_lidar_pcd_ego.T) - det_rot_ego.T @ det_center_ego).T

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(det_lidar_pcd_local)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=50))
        det_lidar_norm_local = np.asanyarray(pcd.normals)

        #pcd_o3d = o3d.open3d.geometry.PointCloud()
        #pcd_o3d.points = o3d.open3d.utility.Vector3dVector(local_crop)
        #pcd_o3d.normals = o3d.open3d.utility.Vector3dVector(local_norm)
        #o3d.visualization.draw_geometries([pcd_o3d])

        dets.append((det_token, det_label, det_vis, det_center_ego.astype(np.float32), det_rot_ego.astype(np.float32), det_lwh.astype(np.float32), det_lidar_pcd_local.astype(np.float32),
                     det_lidar_norm_local.astype(np.float32)))
    return dets
    #show_pcd([det_lidar_pcd_local])

    #ax = plt.gca()
    #ax.axis('equal')

    #ax.scatter(x=det_lidar_pcd_local[:, 0], y=det_lidar_pcd_local[:, 1], s=0.5)
    #ax.scatter(x=det_center[0, :], y=det_center[1, :], marker='x')
    #plt.scatter(x=det_center[0, 0] - det_lwh[0, 0] / 2, y=det_center[1, 0] - det_lwh[1, 0] / 2)
    #plt.scatter(x=det_center[0, 0] + det_lwh[0, 0] / 2, y=det_center[1, 0] + det_lwh[1, 0] / 2)


# %%
USE_VISUAL = True

tq = tqdm(sceneList, desc='converting scenes...')
for scenejson in tq:
    scene_token = scenejson['token']
    tq.set_description(scene_token)
    scene_name = scenejson['name']
    scene_description = scenejson['description']
    scene_root = ospj(OUTPUT_ROOT, scene_token)
    check_root = ospj(OUTPUT_ROOT, '_check')

    os.makedirs(scene_root, exist_ok=True)
    os.makedirs(check_root, exist_ok=True)
    try:
        sample_token = scenejson['first_sample_token']
        sample_nums = scenejson['nbr_samples']

        samples = []
        index = 0
        traj_show_list = []
        while sample_token != '':
            df = DataFrame(sample_token, DATA_ROOT, nusc)

            # targets = make_target_list(df)

            image, lidar_proj = make_visual_info(df)
            image = image
            lidar_proj = lidar_proj.T  # (N, 3)
            c, h, w = image.shape

            image = (image * 255).numpy().astype(np.uint8)

            samples.append(df)
            sample_token = nusc.get('sample', sample_token)['next']

            lidar, lidarseg = df.get_lidar()
            lidar = lidar.astype(np.float32)
            lidarseg = lidarseg.astype(np.float32)

            pose_cam_SE3 = PoseTool.SE3(df.ego_pose.rotation_mat, df.ego_pose.position_xyz)
            traj_show_list.append(pose_cam_SE3)  # 4, 4

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(lidar)
            # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=50))
            # lidarnorm = np.asanyarray(pcd.normals)

            if (index % 10 == 0):
                plt.figure(figsize=(20, 20), dpi=200)

                plt.subplot(221)
                plt.axis('equal')
                plt.title('Point Cloud')
                plt.scatter(lidar[:, 0], lidar[:, 1], c=lidar[:, 2], s=0.1)

                if (USE_VISUAL):
                    plt.subplot(222)
                    plt.title('Image Proj')
                    c, h, w = image.shape
                    plt.imshow(image.transpose(1, 2, 0))
                    vis_mask = lidar_proj[:, 2] > 0  # N,
                    plt.scatter(lidar_proj[vis_mask, 0] * w, lidar_proj[vis_mask, 1] * h, c=lidar_proj[vis_mask, 2], s=1)

                    plt.subplot(223)
                    plt.axis('equal')
                    plt.title('Points with Visual')
                    plt.scatter(lidar[vis_mask, 0], lidar[vis_mask, 1], c=lidar[vis_mask, 2], s=0.1)

                plt.subplot(224)
                plt.axis('equal')
                plt.title('Current Traj')
                traj_show_list_np = np.stack(traj_show_list, axis=0)  # N, 4, 4
                plt.plot(traj_show_list_np[:, 0, 3], traj_show_list_np[:, 1, 3])
                plt.scatter(traj_show_list_np[-1, 0, 3], traj_show_list_np[-1, 1, 3], s=1, c='red')

                plt.tight_layout()
                plt.savefig(os.path.join(check_root, f'{scene_token}_{index}_check.png'))
                plt.close()

            save_root = ospj(scene_root, '0')
            os.makedirs(save_root, exist_ok=True)
            np.savez(
                ospj(save_root, str(index)),
                lidar_pcd=lidar.astype(np.float32),  # N, 3
                lidar_seg=lidarseg.astype(np.int64),  # N, 1
                # lidar_norm=lidarnorm.astype(np.float32),  # N, 3
                lidar_proj=lidar_proj.numpy().astype(np.float32),  # (N, 3) -> [ u, v, d]
                image=image.astype(np.uint8),  # (6, 3, H, W)
                ego_rotation=df.ego_pose.rotation_mat.astype(np.float32),  # 3, 3
                ego_translation=df.ego_pose.position_xyz.astype(np.float32),  # 3, 1
                # targets=targets,  # list[(token, label, vis, rotation, translation, wlh, cropped_pcd, cropped_norm)]
            )
            index += 1
    except:
        print(f'{scene_token} Error!')
        os.rename(ospj(OUTPUT_ROOT, scene_token), ospj(OUTPUT_ROOT, '_' + scene_token))
        continue

    traj_show_list = torch.stack(traj_show_list)  # N, 4, 4
    plt.figure(figsize=(20, 20), dpi=600)
    plt.axis('equal')
    plt.title(f'Traj {scene_token}')
    plt.axis('equal')
    plt.plot(traj_show_list[:, 0, 3], traj_show_list[:, 1, 3])
    plt.savefig(os.path.join(check_root, f'Traj {scene_token}.png'))
    plt.close()

    with open(os.path.join(check_root, f'Traj {scene_token}_xy_gt.txt'), 'w+') as f:
        for pose in traj_show_list:
            f.write(' '.join([str(float(i)) for i in pose[:3, :].flatten().tolist()]) + '\n')
