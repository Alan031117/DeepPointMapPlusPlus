import os
import shutil
from glob import glob


def generate_file(root):
    for file in os.listdir(root):
        file_path = os.path.join(root, file)
        if not os.path.isdir(file_path):
            os.remove(file_path)  # 移除非目录文件
            print(f'remove \'{file_path}\'')

    scenes_root = os.path.join(root, 'Scenes')
    # 遍历所有场景
    for scenes in sorted(os.listdir(scenes_root)):
        agent_root = os.path.join(scenes_root, scenes)
        agent0 = os.path.join(agent_root, '0')
        # 每个场景的点云文件归为0号agent所有
        if os.path.isdir(agent_root) and not os.path.exists(agent0):
            os.mkdir(agent0)
            for pcd_file in glob(os.path.join(agent_root, '*.npz')):
                pcd_file_name = os.path.split(pcd_file)[1]  # 点云文件名
                dst = os.path.join(agent0, pcd_file_name)
                shutil.move(pcd_file, dst)
                print(f'move \'{pcd_file}\' to \'{dst}\'')

    # 每个scenes移动到更高一级文件
    for scenes in sorted(os.listdir(scenes_root)):
        src = os.path.join(scenes_root, scenes)
        dst = os.path.join(root, scenes)
        shutil.move(src, dst)
        print(f'move \'{src}\' to \'{dst}\'')

    os.rmdir(scenes_root)
    print(f'remove \'{scenes_root}\'')


if __name__ == "__main__":
    # generate_file('/root/dataset/nuScenes_v2.0')
    generate_file('/root/dataset/KITTI_v2.0')
    generate_file('/root/dataset/Carla_v2.0')
