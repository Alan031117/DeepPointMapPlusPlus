# 抛出一个运行时错误，提示此脚本有一个关于坐标系X-Z-Y的错误，建议不要使用
raise RuntimeError(f'THIS SCRIPT HAS A BUG (COORDINATE X-Z-Y), DO NOT USE')

# 导入pyquaternion库中的Quaternion类，用于四元数计算
from pyquaternion import Quaternion

# 导入numpy库，用于数值计算
import numpy as np

# 导入os模块，用于文件和路径的操作
import os

# 导入open3d库，用于处理3D数据
import open3d as o3d

# 导入torchvision库，用于处理图像数据
import torchvision

# 导入tqdm库，用于显示进度条
from tqdm import tqdm

# 导入torch库，用于深度学习操作
import torch

# 导入matplotlib.pyplot库，用于绘图
import matplotlib.pyplot as plt

# 导入sys模块，用于与Python解释器交互
import sys

# 导入shutil模块，用于高级文件操作
import shutil

# 导入glob模块，用于文件路径的模式匹配
from glob import glob

# 设置KITTI数据集的版本为v4.0
KITTI_VERSION = 'v4.0'

# 定义一个包含版本说明的多行字符串
Version_Note = '''
version: V4.0

elem:
    pointcloud xyz: (N, xyz), float32  # 点云的XYZ坐标，数据类型为float32
    pointcloud label: (N, c), long  # 点云的标签，数据类型为long
    pointcloud proj: (N, uvd), float32. d = -1 if has no visual info  # 点云投影坐标，数据类型为float32，若无视觉信息则d为-1
    image: (3, H, W), uint8  # 图像数据，3通道，高度H，宽度W，数据类型为uint8
    ego_rotation: (3, 3), float32  # 自身旋转矩阵，3x3，数据类型为float32
    ego_translation: (3, 1), float32  # 自身平移向量，3x1，数据类型为float32

gt: kitti official, convert to x-y plane (aka. lidar coordinate), using Tr.inv() @ Pose_cam @ Tr  # 地面真值，官方KITTI数据集，转换为X-Y平面（即激光雷达坐标系），通过Tr.inv() @ Pose_cam @ Tr计算
label: semantic kitti  # 标签数据，使用语义KITTI数据集
'''

# 设置是否使用视觉数据为False
USE_VISUAL = False

# 设置场景数量为22
SCENE_NUM = 22

# 根据操作系统类型设置数据路径
if sys.platform == 'linux':
    DATA_ROOT = r'/root/dataset/original_KITTI_Odometry/dataset/sequences'  # 数据路径在Linux下的根目录
    POSE_ROOT = r'/root/dataset/original_KITTI_Odometry/dataset/poses_SUMA'  # 位姿数据路径
    CALIB_ROOT = r'/root/dataset/original_KITTI_Odometry/dataset/sequences'  # 校准数据路径
    OUTPUT_ROOT = r'/root/dataset/KITTI_v4.0_SUMA_Visual'  # 输出数据路径
else:
    DATA_ROOT = r'E:\original_KITTI_Odometry\dataset\sequences'  # 数据路径在Windows下的根目录
    IMAGE_ROOT = r'E:\original_KITTI_Odometry\dataset\sequences'  # 图像数据路径
    POSE_ROOT = r'E:\original_KITTI_Odometry\dataset\poses'  # 位姿数据路径
    CALIB_ROOT = r'E:\original_KITTI_Odometry\dataset\sequences'  # 校准数据路径
    OUTPUT_ROOT = r'E:/KITTI_v4.0_NoVisual'  # 输出数据路径

# 打印加载原始KITTI数据集的路径
print(f'Load original KITTI from {os.path.abspath(DATA_ROOT)}')

# 打印将要创建新KITTI数据集的路径
print(f'Will create new KITTI dataset at {os.path.abspath(OUTPUT_ROOT)}')

# 定义一个字典来存储EGO_POSE，即自身位姿
EGO_POSE = {}

# 通过读取POSE_ROOT目录下的txt文件来加载场景的位姿信息
for file_name in sorted(glob(os.path.join(POSE_ROOT, '*.txt'))):
    scene_pose = []  # 定义一个列表来存储场景的每帧位姿
    for line in open(os.path.join(POSE_ROOT, file_name), 'r', encoding='UTF-8'):
        frame_line = np.asarray([eval(x) for x in line.strip('\n').split(' ')]).reshape(3, 4)  # 解析每一行位姿数据，并转换为3x4的数组
        frame_pose = {"translation": frame_line[:, 3].flatten(), "rotation": frame_line[:, :3].flatten()}  # 提取平移和旋转矩阵，并将其存入字典中
        scene_pose.append(frame_pose)  # 将每帧位姿数据添加到场景位姿列表中
    EGO_POSE[os.path.basename(file_name).split('.')[0]] = scene_pose  # 将场景位姿列表存储到EGO_POSE字典中，键为场景名称
print(f"already loaded {len(EGO_POSE)} scenes poses")  # 打印已经加载的场景位姿数量

# 定义一个字典来存储每个场景的时间戳信息
TIMESTAMPS = {}

# 通过读取CALIB_ROOT目录下的times.txt文件来加载场景的时间戳信息
for file_name in sorted(os.listdir(CALIB_ROOT)):
    scene_time = []  # 定义一个列表来存储场景的每帧时间戳
    for line in open(os.path.join(CALIB_ROOT, file_name, 'times.txt'), 'r', encoding='UTF-8'):
        time = eval(line.strip('\n'))  # 解析每一行时间戳数据
        scene_time.append(time)  # 将每帧时间戳添加到场景时间列表中
    TIMESTAMPS[file_name] = scene_time  # 将场景时间列表存储到TIMESTAMPS字典中，键为场景名称
print(f"already loaded {len(TIMESTAMPS)} scenes timestamps")  # 打印已经加载的场景时间戳数量

# 定义一个字典来存储校准信息
CALIB = {}

# 通过读取CALIB_ROOT目录下的calib.txt文件来加载场景的校准信息
for file_name in sorted(os.listdir(CALIB_ROOT)):
    intrinsics = {}  # 定义一个字典来存储相机的内参信息
    for line in open(os.path.join(CALIB_ROOT, file_name, 'calib.txt'), 'r', encoding='UTF-8'):
        line = line.strip('\n').split(':')  # 解析每一行校准数据
        if (line[0] == 'Tr'):
            TRANSFORM = np.eye(4)  # 初始化4x4的单位矩阵
            TRANSFORM[:3, :] = np.asarray([eval(x) for x in line[1].split(" ")[1:]]).reshape(3, 4)  # 填充转换矩阵的前3行4列
        else:
            intrinsics[line[0]] = np.asarray([eval(x) for x in line[1].split(" ")[1:]]).reshape(3, 4)  # 将相机内参信息存储到intrinsics字典中
    CALIB[file_name] = (TRANSFORM.copy(), intrinsics)  # 将转换矩阵和内参信息存储到CALIB字典中，键为场景名称
    del TRANSFORM  # 删除转换矩阵，释放内存
print(f"already loaded {len(CALIB)} scenes calib")  # 打印已经加载的场景校准信息数量

# 定义Pose类，用于表示位姿
class Pose(object):
    # 初始化函数，接受位置坐标和旋转信息
    def __init__(self, pos_xyz=None, rot=None, rot_type='quat') -> None:
        self.pos = np.array(pos_xyz).reshape(3, 1)  # 将位置坐标转换为3x1的数组
        if (rot_type.lower() == 'quat'):
            self.rot = Quaternion(np.array(rot).reshape(4)).rotation_matrix  # 如果旋转类型是四元数，则将其转换为旋转矩阵
        elif (rot_type.lower() == 'rot_mat'):
            self.rot = np.array(rot).reshape(3, 3)  # 如果旋转类型是旋转矩阵，则直接转换为3x3的矩阵
        else:
            raise RuntimeError(f"rot para in Pose class not defined. found {rot_type}")  # 如果旋转类型不支持，抛出运行时错误
        pass  # 占位符，无操作

    # 定义position_xyz属性，返回位置坐标
    @property
    def position_xyz(self):
        '''
        获取3x3的旋转矩阵
        '''
        return self.pos

    # 定义rotation_mat属性，返回旋转矩阵
    @property
    def rotation_mat(self):
        '''
        获取3x1的平移向量
        '''
        return self.rot

    # 重载加法运算符，实现两个位姿的组合
    def __add__(self, other):
        '''
        按顺序应用两个变换

        Q = R2 x (R1 x Q + t1) + t2
          = R2 x R1 x Q + R2 x t1 + t2
            ^~~~~~~       ^~~~~~~~~~~~
            新的旋转矩阵     新的平移向量
        '''
        rot = other.rotation_mat @ self.rotation_mat.copy()  # 计算新的旋转矩阵
        pos = other.rotation_mat @ self.position_xyz.copy() + other.pos  # 计算新的平移向量
        ret = Pose(pos, rot, rot_type='rot_mat')  # 创建新的Pose对象
        return ret

    # 定义逆变换方法，返回当前变换的逆
    def inv(self):
        rot = self.rotation_mat.copy().T  # 计算旋转矩阵的转置，得到逆矩阵
        pos = -(rot @ self.position_xyz.copy())  # 计算平移向量的相反数
        ret = Pose(pos, rot, rot_type='rot_mat')  # 创建新的Pose对象
        return ret

    # 定义homogeneous方法，返回4x4的齐次变换矩阵
    def homogeneous(self):
        '''
        返回4x4的齐次变换矩阵
        '''
        ret = np.eye(4)  # 初始化为4x4的单位矩阵
        ret[:3, :3] = self.rotation_mat  # 填充旋转矩阵
        ret[:3, 3] = self.position_xyz.flatten()  # 填充平移向量
        return ret  # 返回齐次变换矩阵



# 定义一个名为DataFrame的类，用于处理场景中的数据帧
class DataFrame:
    # 初始化函数，接受scene_token（场景标识符）、sample_token（样本标识符）和data_root（数据根目录）作为参数
    def __init__(self, scene_token: int, sample_token: int, data_root: str):
        self.data_root = data_root  # 保存数据根目录路径
        self.scene_token = str(scene_token).zfill(2)  # 将场景标识符转换为两位数的字符串，前面补0
        self.sample_token = sample_token  # 保存样本标识符
        '''相机内参信息'''
        # 从CALIB字典中提取当前场景的相机内参，并存储在cam_intrinsics字典中
        self.cam_intrinsics = {"color_left": CALIB[self.scene_token][1]['P2'], "color_right": CALIB[self.scene_token][1]['P3']}
        '''激光雷达到相机的转换矩阵'''
        # 从CALIB字典中提取当前场景的激光雷达到相机的转换矩阵
        self.velodyne_to_cam_transform = CALIB[self.scene_token][0]
        # self.cam_to_velodyne_transform = np.linalg.inv(TRANSFORM)  # 注释掉的代码，用于计算相机到激光雷达的转换矩阵（反矩阵）
        '''时间戳信息'''
        # 从TIMESTAMPS字典中提取当前样本的时间戳
        self.timestamps = TIMESTAMPS[self.scene_token][self.sample_token]
        '''自身位姿信息（激光雷达位置信息）'''
        # 检查EGO_POSE字典中是否存在当前场景的位姿信息
        if self.scene_token not in EGO_POSE.keys():
            raise  # 如果不存在，则抛出异常
        else:
            # 如果存在，则从EGO_POSE字典中提取当前样本的位姿信息，并初始化Pose对象
            self.ego_pose = Pose(EGO_POSE[self.scene_token][self.sample_token]['translation'], EGO_POSE[self.scene_token][self.sample_token]['rotation'], rot_type='rot_mat')

    # 定义一个方法get_label，用于获取标签数据
    def get_label(self) -> np.ndarray:
        # 构建标签文件的路径
        label_path = os.path.join(self.data_root, self.scene_token, "labels", f"{str(self.sample_token).zfill(6)}.label")
        # 从标签文件中读取数据，并提取低16位作为标签
        label = (np.fromfile(label_path, dtype=np.uint32) & 0xFFFF)[:, np.newaxis]  # (N, )
        return label  # 返回标签数组

    # 定义一个方法get_lidar，用于获取激光雷达点云数据
    def get_lidar(self):
        # 构建激光雷达文件的路径
        lidar_path = os.path.join(self.data_root, self.scene_token, "velodyne", f"{str(self.sample_token).zfill(6)}.bin")
        # 从激光雷达文件中读取点云数据，转换为(N, 3)的形状，N为点的数量
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]  # (N,3)
        return points  # 返回点云数据

    # 定义一个方法get_images，用于获取相机图像数据
    def get_images(self):
        """
            获取当前数据帧的两张相机图像
            返回形状为[2, c, h, w]的数组，值范围为[0, 1]
        """
        imgs = []  # 定义一个列表用于存储图像数据
        # 根据操作系统类型选择不同的图像路径
        if sys.platform == 'linux':
            # 在Linux系统中，从data_root目录读取图像
            imgs.append(torchvision.io.read_image(os.path.join(self.data_root, self.scene_token, "image_2", f"{str(self.sample_token).zfill(6)}.png")))  # [c, h, w]
            imgs.append(torchvision.io.read_image(os.path.join(self.data_root, self.scene_token, "image_3", f"{str(self.sample_token).zfill(6)}.png")))  # [c, h, w]
        else:
            # 在其他系统中（如Windows），从IMAGE_ROOT目录读取图像
            imgs.append(torchvision.io.read_image(os.path.join(IMAGE_ROOT, self.scene_token, "image_2", f"{str(self.sample_token).zfill(6)}.png")))  # [c, h, w]
            imgs.append(torchvision.io.read_image(os.path.join(IMAGE_ROOT, self.scene_token, "image_3", f"{str(self.sample_token).zfill(6)}.png")))  # [c, h, w]
        imgs = torch.stack(imgs).numpy()  # 将图像数据堆叠为[2, c, h, w]的数组，并转换为numpy数组
        return imgs  # 返回图像数据


# 定义一个函数show_pcd，用于显示点云数据
def show_pcd(pcds, colors=None, window_name="PCD", normals=False):
    # 创建可视化窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name=window_name)
    for i in range(len(pcds)):
        # 创建点云对象
        pcd_o3d = o3d.open3d.geometry.PointCloud()
        # 将输入的点云数据转换为Open3D可以直接使用的数据类型
        if (isinstance(pcds[i], np.ndarray)):
            pcd_points = pcds[i][:, :3]  # 提取点的XYZ坐标
        elif (isinstance(pcds[i], torch.Tensor)):
            pcd_points = pcds[i][:, :3].detach().cpu().numpy()  # 如果是PyTorch张量，先转为numpy数组
        else:
            pcd_points = np.array(pcds[i][:, :3])  # 其他情况直接转为numpy数组
        pcd_o3d.points = o3d.open3d.utility.Vector3dVector(pcd_points)  # 将点云数据赋给Open3D点云对象
        # pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))  # 注释掉的代码，用于估计点云的法向量
        # 如果提供了颜色信息，则为点云设置统一的颜色
        if colors is not None:
            pcd_o3d.paint_uniform_color(colors[i])
        # 将点云对象添加到可视化窗口中
        vis.add_geometry(pcd_o3d)

    vis.run()  # 启动可视化窗口，显示点云
    vis.destroy_window()  # 关闭可视化窗口



# 定义一个名为PoseTool的类，用于处理位姿变换相关的工具方法
class PoseTool(object):
    # 定义一个类方法SE3，用于将旋转矩阵和平移向量构造成齐次变换矩阵
    @classmethod
    def SE3(cls, R, t):
        # 如果输入的R是numpy数组，将其转换为PyTorch张量并调整形状为3x3
        if (isinstance(R, np.ndarray)):
            R = torch.tensor(R, dtype=torch.float32).reshape(3, 3)
        # 如果输入的t是numpy数组，将其转换为PyTorch张量并调整形状为3x1
        if (isinstance(t, np.ndarray)):
            t = torch.tensor(t, dtype=torch.float32).reshape(3, 1)
        # 创建一个4x4的单位矩阵
        mat = torch.eye(4)
        # 将输入的旋转矩阵R放入齐次变换矩阵的左上3x3部分
        mat[:3, :3] = R
        # 将输入的平移向量t放入齐次变换矩阵的左侧3x1部分
        mat[:3, 3:4] = t
        # 返回构造好的4x4齐次变换矩阵
        return mat

    # 定义一个类方法Rt，用于从齐次变换矩阵中提取旋转矩阵和平移向量
    @classmethod
    def Rt(cls, SE3):
        '''
        R: torch.Tensor(3, 3)  # 输出的旋转矩阵
        t: torch.Tensor(3, 1)  # 输出的平移向量
        '''
        # 从输入的4x4齐次变换矩阵中提取左上角的3x3旋转矩阵
        R = SE3[:3, :3]
        # 从输入的4x4齐次变换矩阵中提取左侧的3x1平移向量
        t = SE3[:3, 3:]
        # 返回旋转矩阵和平移向量
        return (R, t)

# 定义一个函数make_visual_info，用于将点云数据与相机图像对齐并生成视觉信息
def make_visual_info(df: DataFrame):
    # 从DataFrame对象中获取两张相机图像数据，[2, c, h, w]
    images = df.get_images()
    # 从DataFrame对象中获取激光雷达点云数据并转置为(3,N)的形状
    points = df.get_lidar().T  # (3,N)
    # 将点云数据扩展为4行（添加一行全为1的行），使其适应齐次坐标的计算
    points = np.concatenate((points, np.ones((1, points.shape[1]))))  # (4,N)

    # 创建一个4x4的单位矩阵，用于表示相机内参矩阵
    cam_in_left = np.eye(4)
    # 将左相机的内参矩阵填入单位矩阵的前三行，数据缩放为米级单位（除以1000）
    cam_in_left[:3, :] = df.cam_intrinsics['color_left'] / 1000  # (4,4)
    # 获取激光雷达到相机的转换矩阵
    velodyne_to_cam = df.velodyne_to_cam_transform  # (4,4)

    # 将点云转换到左相机坐标系下
    pcd_cam = cam_in_left @ velodyne_to_cam @ points  # (4(x,y,k,l),N)
    # 提取转换后的点云深度信息（Z轴数据）
    depth = pcd_cam[2, :]
    # 将点的齐次坐标归一化为(u, v, 1, 1)的形式，并恢复深度信息
    pcd_cam = pcd_cam / pcd_cam[3, :] / depth  # 点的坐标归一化为(u,v,1,1)
    pcd_cam[2, :] = depth  # 恢复点的深度 (u,v,d,1)

    # 只保留前3行数据，即(u, v, d)表示的点云数据
    pcd_cam = pcd_cam[:3, :]  # uvd, N

    # 丢弃超出视野范围的点，创建一个全为True的布尔掩码，用于标记点是否在视野范围内
    vis_mask = np.ones(depth.shape, dtype=bool)  # N,1
    # 丢弃位于摄像机后方的点（深度为负的点）
    vis_mask = np.logical_and(vis_mask, depth > 0)  # 丢弃摄像机后方的点
    # 丢弃超出图像宽度范围的点（u坐标小于1或大于图像宽度减1）
    vis_mask = np.logical_and(vis_mask, pcd_cam[0, :] > 1)
    vis_mask = np.logical_and(vis_mask, pcd_cam[0, :] < images[0].shape[2] - 1)
    # 丢弃超出图像高度范围的点（v坐标小于1或大于图像高度减1）
    vis_mask = np.logical_and(vis_mask, pcd_cam[1, :] > 1)
    vis_mask = np.logical_and(vis_mask, pcd_cam[1, :] < images[0].shape[1] - 1)

    # 将不在视野范围内的点的坐标置为0，深度置为-1，表示无效点
    pcd_cam[:, ~vis_mask] = 0
    pcd_cam[2, ~vis_mask] = -1

    # 将(u, v)坐标进行归一化处理
    _, C, H, W = images.shape  # 提取图像的通道数、高度和宽度
    pcd_cam[0, :] /= W  # u坐标归一化，除以图像宽度
    pcd_cam[1, :] /= H  # v坐标归一化，除以图像高度

    # 返回左相机图像和处理后的点云数据
    return images[0, :, :, :], pcd_cam

# 定义一个齐次变换矩阵dG，用于表示某种旋转和平移
dG = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).float()

# 使用tqdm库显示加载场景的进度条
scene_tq = tqdm(range(SCENE_NUM), desc='loading scenes...')
# 创建输出目录，如果目录不存在则创建
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 打开或创建一个名为'note.txt'的文件，用于记录版本信息
with open(os.path.join(OUTPUT_ROOT, 'note.txt'), 'w+') as f:
    f.write(Version_Note)  # 将版本说明写入文件


# 对每个场景的ID进行迭代
for scene_id in scene_tq:
    # 列出当前场景下激光雷达文件夹中的所有文件，并创建一个进度条
    sample_tq = tqdm(range(len(os.listdir(os.path.join(DATA_ROOT, str(scene_id).zfill(2), 'velodyne')))), desc=f'loading samples in scene {scene_id}')

    # 定义当前场景的输出根目录（存储数据）和检查根目录（用于存储检查图像）
    scene_root = os.path.join(OUTPUT_ROOT, str(scene_id).zfill(2), '0')
    check_root = os.path.join(OUTPUT_ROOT, f'_check', str(scene_id).zfill(2))

    # 如果当前场景的输出目录已经存在，跳过该场景的处理
    if (os.path.exists(scene_root)):
        # shutil.rmtree(scene_root)  # 如果想要重新处理，可以取消注释，删除现有的目录
        continue  # 跳过当前场景
    # 创建输出目录和检查目录，如果目录不存在则创建
    os.makedirs(scene_root, exist_ok=True)
    os.makedirs(check_root, exist_ok=True)

    # 初始化一个列表，用于存储轨迹显示数据
    traj_show_list = []

    # 对每个数据帧的ID进行迭代
    for frame_id, sample_id in enumerate(sample_tq):
        # 如果到达倒数第10帧，可以取消注释代码用于显示点云数据和轨迹数据并退出
        # if (frame_id == len(sample_tq) - 10):
        #     # show_pcd(pcd_show_list)
        #     # traj_show_list = np.concatenate(traj_show_list, axis=1)

        #     plt.figure()
        #     plt.axis('equal')
        #     traj_show_list_np = np.concatenate(traj_show_list, axis=1)
        #     plt.scatter(traj_show_list_np[0, :], traj_show_list_np[1, :], s=0.1)
        #     plt.show()
        #     exit()

        # 创建DataFrame对象，获取当前数据帧的信息
        df = DataFrame(scene_id, sample_id, DATA_ROOT)
        # 获取激光雷达点云数据并转换为PyTorch张量，形状为(3, N)
        lidar = torch.from_numpy(df.get_lidar()).T  # 3, N
        # 获取激光雷达到相机的转换矩阵，并转换为PyTorch张量
        velodyne_to_cam_transform = torch.tensor(df.velodyne_to_cam_transform).float()
        # 如果使用视觉数据，则生成视觉信息
        if (USE_VISUAL):
            images, lidar_proj = make_visual_info(df)
            images = images  # C, h, w

        # 获取自身位姿的旋转矩阵和平移向量，并转换为PyTorch张量
        pose_r = torch.tensor(df.ego_pose.rotation_mat.astype(np.float32))
        pose_t = torch.tensor(df.ego_pose.position_xyz.astype(np.float32))

        # 将位姿转换为齐次变换矩阵
        pose_cam_SE3 = PoseTool.SE3(pose_r, pose_t)
        # 使用齐次变换矩阵和激光雷达到相机的变换矩阵计算激光雷达的位姿
        # pose_lidar_SE3 = velodyne_to_cam_transform.inverse() @ pose_cam_SE3 @ velodyne_to_cam_transform

        # 将激光雷达到相机的转换矩阵扩展维度，以适应批量矩阵运算
        velodyne_to_cam_transform = velodyne_to_cam_transform.unsqueeze(0)  # 1, 4, 4
        # 计算激光雷达到相机的逆变换
        left = np.einsum("...ij,...jk->...ik", np.linalg.inv(velodyne_to_cam_transform.numpy()), pose_cam_SE3.numpy())
        # 计算激光雷达的齐次变换矩阵
        right = np.einsum("...ij,...jk->...ik", left, velodyne_to_cam_transform.numpy())
        pose_lidar_SE3 = torch.from_numpy(right[0, :, :])

        # 从齐次变换矩阵中提取旋转矩阵和平移向量
        pose_r, pose_t = PoseTool.Rt(pose_lidar_SE3)

        # 将当前数据帧的位姿添加到轨迹显示列表中
        traj_show_list.append(pose_lidar_SE3)  # 4, 4

        # 取消注释以下代码以估计点云的法向量（目前被注释掉）
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(lidar.T.numpy())
        # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=50))
        # lidarnorm = np.asanyarray(pcd.normals)

        # 获取当前数据帧的标签数据
        label = df.get_label()

        # 每300帧可视化一次结果并保存图像
        if (frame_id % 300 == 0):
            plt.figure(figsize=(20, 20), dpi=200)

            # 绘制点云数据的散点图
            plt.subplot(221)
            plt.axis('equal')
            plt.title('Point Cloud')
            plt.scatter(lidar[0, :], lidar[1, :], c=lidar[2, :], s=0.1)

            # 如果使用视觉数据，则绘制图像投影和视觉点云
            if (USE_VISUAL):
                plt.subplot(222)
                plt.title('Image Proj')
                c, h, w = images.shape
                plt.imshow(images.transpose(1, 2, 0))  # 显示相机图像
                vis_mask = lidar_proj[2, :] > 0  # 筛选出深度大于0的点
                plt.scatter(lidar_proj[0, vis_mask] * w, lidar_proj[1, vis_mask] * h, c=lidar_proj[2, vis_mask], s=0.1)  # 在图像上投影点云

                plt.subplot(223)
                plt.axis('equal')
                plt.title('Points with Visual')
                plt.scatter(lidar[0, vis_mask], lidar[1, vis_mask], c=lidar[2, vis_mask], s=0.1)  # 绘制视觉点云的散点图

            # 绘制当前轨迹的图像
            plt.subplot(224)
            plt.axis('equal')
            plt.title('Current Traj')
            traj_show_list_np = np.stack(traj_show_list, axis=0)  # 将轨迹列表转换为numpy数组
            plt.plot(traj_show_list_np[:, 0, 3], traj_show_list_np[:, 1, 3])  # 绘制轨迹路径
            plt.scatter(traj_show_list_np[-1, 0, 3], traj_show_list_np[-1, 1, 3], s=1, c='red')  # 标记当前帧的位置

            # 调整布局并保存检查图像
            plt.tight_layout()
            plt.savefig(os.path.join(check_root, f'{str(scene_id).zfill(2)}_{frame_id}_check.png'))
            plt.close()

        # 保存当前数据帧的处理结果
        if (USE_VISUAL):
            # 保存带有视觉信息的数据
            np.savez(
                os.path.join(scene_root, str(sample_id)),
                lidar_pcd=lidar.T.numpy().astype(np.float32),  # 保存点云数据，形状为(N, 3)
                lidar_seg=label.astype(np.int64),  # 保存点云标签数据，形状为(N, 1)
                # lidar_norm=lidarnorm.astype(np.float32),  # 保存点云法向量数据，形状为(N, 3)（目前被注释掉）
                lidar_proj=lidar_proj.T.astype(np.float32),  # 保存点云在相机图像中的投影坐标，形状为(N, 3)
                image=images.astype(np.uint8),  # 保存相机图像数据，形状为(3, H, W)
                ego_rotation=pose_r.numpy().astype(np.float32),  # 保存位姿的旋转矩阵，形状为(3, 3)
                ego_translation=pose_t.numpy().astype(np.float32),  # 保存位姿的平移向量，形状为(3, 1)
            )
        else:
            # 保存不带视觉信息的数据
            np.savez(
                os.path.join(scene_root, str(sample_id)),
                lidar_pcd=lidar.T.numpy().astype(np.float32),  # 保存点云数据，形状为(N, 3)
                lidar_seg=label.astype(np.int64),  # 保存点云标签数据，形状为(N, 1)
                # lidar_norm=lidarnorm.astype(np.float32),  # 保存点云法向量数据，形状为(N, 3)（目前被注释掉）
                # lidar_proj=lidar_proj.T.astype(np.float32),  # 保存点云在相机图像中的投影坐标，形状为(N, 3)（目前被注释掉）
                # image=images.astype(np.uint8),  # 保存相机图像数据，形状为(3, H, W)（目前被注释掉）
                ego_rotation=pose_r.numpy().astype(np.float32),  # 保存位姿的旋转矩阵，形状为(3, 3)
                ego_translation=pose_t.numpy().astype(np.float32),  # 保存位姿的平移向量，形状为(3, 1)
            )

    # 将轨迹列表转换为PyTorch张量，形状为(N, 4, 4)
    traj_show_list = torch.stack(traj_show_list)  # N, 4, 4
    # 绘制完整轨迹的图像并保存
    plt.figure(figsize=(20, 20), dpi=600)
    plt.axis('equal')
    plt.title(f'Traj {str(scene_id).zfill(2)}')
    plt.axis('equal')
    plt.plot(traj_show_list[:, 0, 3], traj_show_list[:, 1, 3])  # 绘制轨迹路径
    plt.savefig(os.path.join(check_root, f'Traj {str(scene_id).zfill(2)}.png'))  # 保存轨迹图像
    plt.close()

    # 保存轨迹的位置信息到文本文件中
    with open(os.path.join(check_root, f'Traj {str(scene_id).zfill(2)}_xy_gt.txt'), 'w+') as f:
        for pose in traj_show_list:
            f.write(' '.join([str(float(i)) for i in pose[:3, :].flatten().tolist()]) + '\n')  # 将位姿的3x4部分保存为文本格式
