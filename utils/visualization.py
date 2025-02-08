from typing import List
import open3d as o3d
import numpy as np
import torch


def show_pcd(pcds: List, colors: List = None, window_name: str = "PCD",
             has_normals: bool = False, estimate_normals: bool = False, estimate_kwargs: dict = None,
             filter: bool = False) -> None:
    """
    点云可视化展示

    Args:
        pcds: [Array1, Array2, ...] Array.shape = (N, 3+)
        colors: [RGB1, RGB2, ...] RGB.shape = (3,), like [1, 0.5, 0] for R=1, G=0.5, B=0
        window_name: 窗口展示的名称
        has_normals: 输入pcd是否已包含法线
        estimate_normals: 是否估计法线
        estimate_kwargs: 法线估计的参数字典
        filter: 是否滤波

    Returns:
        None
    """
    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name=window_name, width=2880, height=1620)
    # 设置点云大小
    # vis.get_render_option().point_size = 1
    # 设置颜色背景为黑色
    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([0, 0, 0])

    print(f'{window_name:*<30}')
    for i in range(len(pcds)):
        # 创建点云对象
        pcd_o3d = o3d.open3d.geometry.PointCloud()
        # 将点云数据转换为Open3d可以直接使用的数据类型
        if isinstance(pcds[i], np.ndarray):
            pcd_points = pcds[i][:, :3]
        elif isinstance(pcds[i], torch.Tensor):
            pcd_points = pcds[i][:, :3].detach().cpu().numpy()
        else:
            pcd_points = np.array(pcds[i][:, :3])
        pcd_o3d.points = o3d.open3d.utility.Vector3dVector(pcd_points)

        if has_normals:
            if pcds[i].shape[1] < 6:
                print('Normals is NOT found')
            else:
                if isinstance(pcds[i], np.ndarray):
                    pcd_normals = pcds[i][:, 3:6]
                elif isinstance(pcds[i], torch.Tensor):
                    pcd_normals = pcds[i][:, 3:6].detach().cpu().numpy()
                else:
                    pcd_normals = np.array(pcds[i][:, 3:6])
                pcd_o3d.normals = o3d.open3d.utility.Vector3dVector(pcd_normals)

        if filter:
            pcd_o3d = pcd_o3d.remove_statistical_outlier(nb_neighbors=20, std_ratio=3)[0]

        if estimate_normals:
            if estimate_kwargs is None:
                radius, max_nn = 1, 30
            else:
                assert 'radius' in estimate_kwargs.keys() and 'max_nn' in estimate_kwargs.keys()
                radius, max_nn = estimate_kwargs['radius'], estimate_kwargs['max_nn']
            pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))

        # 设置点的颜色
        if colors is not None:
            pcd_o3d.paint_uniform_color(colors[i])
        # 将点云加入到窗口中
        vis.add_geometry(pcd_o3d)
        print(pcd_o3d)
    print('*' * 30)

    vis.run()
    vis.destroy_window()
