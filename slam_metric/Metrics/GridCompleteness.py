import torch
import open3d as o3d


class GirdCompletenessMetric():
    def __init__(self, grid_size=10) -> None:
        self.grid_size = grid_size
        self.pcd_map_o3d = o3d.geometry.PointCloud()
        pass

    def eval(self, pointcloud_map: torch.Tensor):
        '''
        pointcloud_map: [fea+xyz, N]
        '''
        coors = pointcloud_map[-3:, :]  # (3, N)
        coors[2, :] = 0  #set z-axis to 0
        self.pcd_map_o3d.points = o3d.open3d.utility.Vector3dVector(coors.numpy().T)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd_map_o3d, voxel_size=self.grid_size)
        comp_index = len(voxel_grid.get_voxels()) / ((voxel_grid.get_max_bound() - voxel_grid.get_min_bound()) / self.grid_size).prod()
        return comp_index