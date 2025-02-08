import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d


# imu_to_lidar_matrix = np.eye(3)
imu_to_lidar_se3 = np.array([
    [0.125073,  -0.886741, 0.445025,  -0.0468777],
    [0.00183693, 0.448754, 0.893653,  -0.089225],
    [-0.992146, -0.110955, 0.0577561, -0.0886047],
    [0,          0,        0,         1]
])
imu_to_lidar_matrix = imu_to_lidar_se3[:3, :3]
lidar_time_offset = -0.0015327
G = 9.7946  # 重力加速度


def remove_motion_distortion(point_cloud: np.ndarray, imu_data:np.ndarray, init_velocity: np.ndarray) -> np.ndarray:
    """
    基于IMU的点云运动畸变去除

    :param point_cloud: 点云数据 [timestamp, x, y, z]
    :param imu_data: IMU数据 [timestamp, qx, qy, qz, qw, ax, ay, az, lx, ly, lz]
    :param init_velocity: 初始速度 [vx, vy, vz]
    :return: 去除畸变后的点云 [x, y, z]
    """
    point_cloud_timestamps = point_cloud[:, 0]
    point_cloud_xyz = point_cloud[:, 1:]

    imu_timestamps = imu_data[:, 0]
    imu_quaternions = imu_data[:, 1:5]
    imu_angular_velocity = imu_data[:, 5:8]
    imu_linear_acceleration = imu_data[:, 8:]

    # 根据时间戳进行排序和对齐
    sorted_indices = np.argsort(point_cloud_timestamps)
    point_cloud_timestamps = point_cloud_timestamps[sorted_indices]
    point_cloud_xyz = point_cloud_xyz[sorted_indices]

    interpolate_quaternions = interp1d(imu_timestamps, imu_quaternions, axis=0, fill_value='extrapolate')
    imu_quaternions = interpolate_quaternions(point_cloud_timestamps)
    interpolate_angular_velocity = interp1d(imu_timestamps, imu_angular_velocity, axis=0, fill_value='extrapolate')
    imu_angular_velocity = interpolate_angular_velocity(point_cloud_timestamps)
    interpolate_linear_acceleration = interp1d(imu_timestamps, imu_linear_acceleration, axis=0,
                                               fill_value='extrapolate')
    imu_linear_acceleration = interpolate_linear_acceleration(point_cloud_timestamps)

    # 获取每个时刻的方位角
    imu_rotations = Rotation.from_quat(imu_quaternions).as_matrix()

    # IMU坐标系转雷达坐标系
    lidar_rotations = imu_rotations @ imu_to_lidar_matrix
    lidar_linear_acceleration = imu_linear_acceleration @ imu_to_lidar_matrix

    # 计算相对于扫描结束时刻方位的相对旋转矩阵
    lidar_relative_rotations = lidar_rotations.T @ lidar_rotations[-1]

    # 加速度对时间的积分求出每个时间戳的速度增量，再与初始速度相加得到每个时间戳的速度
    delta_time = np.diff(point_cloud_timestamps)
    delta_velocity = np.cumsum(lidar_linear_acceleration[1:] * delta_time[:, np.newaxis], axis=0)
    velocities = np.vstack((np.zeros(3), delta_velocity)) + init_velocity

    # 速度对时间的积分求出每个时间戳的位姿增量，得到每个点的位移
    flip_velocities_lidar = np.flip(velocities, axis=0)  # 对齐到结束时刻，需要反向积分求各点的位移
    flip_delta_time = np.flip(delta_time, axis=0)
    point_translation = np.cumsum(flip_velocities_lidar[1:] * flip_delta_time[:, np.newaxis], axis=0)
    point_translation = np.vstack((np.zeros(3), point_translation))
    point_translation = np.flip(point_translation, axis=0)

    # 根据每个点的旋转和位移，对齐到结束时刻
    # TODO 检查这里的旋转和位姿补偿方向是否相反，点的补偿位移应该与计算的位移互为相反数，旋转补偿应与相对结束时刻的旋转互逆
    transformed_point_cloud = point_cloud_xyz - point_translation
    transformed_point_cloud = lidar_relative_rotations.inv().as_matrix() @ transformed_point_cloud[:, :, np.newaxis]
    transformed_point_cloud = transformed_point_cloud.squeeze(-1)

    return transformed_point_cloud


def load_imu(file_path):
    with open(file_path, 'r') as f:
        data_list = f.readlines()
    data_list = [i.strip() for i in data_list if len(i.strip()) > 0]
    assert len(data_list) % 22 == 0
    imu_data_len = int(len(data_list) / 22)
    imu_data = []
    for i in range(imu_data_len):
        current_data = data_list[i * 22: (i + 1) * 22]
        timestamp = float(current_data[3][6:].strip()) + float('0.' + current_data[4][7:].strip())
        qx = float(current_data[7][3:])
        qy = float(current_data[8][3:])
        qz = float(current_data[9][3:])
        qw = float(current_data[10][3:])
        ax = float(current_data[13][3:])
        ay = float(current_data[14][3:])
        az = float(current_data[15][3:])
        lx = float(current_data[18][3:])
        ly = float(current_data[19][3:])
        lz = float(current_data[20][3:])
        imu_data.append([timestamp, qx, qy, qz, qw, ax, ay, az, lx, ly, lz])
    imu_data = np.array(imu_data)
    return imu_data


if __name__ == '__main__':
    # 使用示例：
    pcd_path = r'/Volumes/DZH/fudan/230712/velodyne/000001.bin'
    imu_path = r'/Volumes/DZH/fudan/230712/imu.txt'
    points = np.fromfile(pcd_path, dtype=np.float64).reshape(-1, 6)
    points = points[np.isnan(points).sum(1) == 0]
    points[:, -1:] += lidar_time_offset
    timestamp = points[:, -1:]
    point_cloud = np.concatenate([timestamp, points[:, :3]], axis=1)
    imu_data = load_imu(imu_path)
    min_t, max_t = timestamp.min(), timestamp.max()
    min_time_diff = imu_data[:, 0] - min_t
    max_time_diff = imu_data[:, 0] - max_t
    left_index = max((min_time_diff < 0).sum() - 1, 0)
    right_index = min((max_time_diff < 0).sum(), min_time_diff.shape[0] - 1) + 1

    init_velocity = np.array([0.0, 0.0, 0.0])  # 1 x 3的numpy数组

    transformed_point_cloud = remove_motion_distortion(point_cloud, imu_data[left_index: right_index], init_velocity)

    import open3d as o3d
    color = ((timestamp - timestamp.min()) * 10).repeat(3, 1)
    pcd_ori = o3d.geometry.PointCloud()
    pcd_ori.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd_ori_color = color.copy()
    pcd_ori_color[:, 1:] = 0
    pcd_ori.colors = o3d.utility.Vector3dVector(pcd_ori_color)
    pcd_undistortion = o3d.geometry.PointCloud()
    pcd_undistortion.points = o3d.utility.Vector3dVector(transformed_point_cloud)
    pcd_undistortion_color = color.copy()
    pcd_undistortion_color[:, [0, 2]] = 0
    pcd_undistortion.colors = o3d.utility.Vector3dVector(pcd_undistortion_color)

    o3d.visualization.draw_geometries([pcd_ori, pcd_undistortion])
