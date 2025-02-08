import torch
import matplotlib.pyplot as plt

dG = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).float()


def txt2se3(calib: str, txt_file: str):
    calib = open(calib, 'r').readlines()
    for l in calib:
        if (l.startswith('Tr:')):
            TRANSFORM = torch.eye(4)
            c = [float(i) for i in l.split()[1:]]
            c = torch.tensor(c).reshape(3, 4)
            TRANSFORM[:3, :] = c  # 4, 4

    txt_data = open(txt_file, 'r').readlines()
    SE3 = []
    for l in txt_data:
        pose = [float(i) for i in l.split()]
        pose = torch.tensor(pose).reshape(3, 4)
        pose = torch.concat([pose, torch.Tensor([0, 0, 0, 1]).float().view(1, 4)], dim=0)

        pose = dG @ pose @ TRANSFORM
        SE3.append(pose)
    SE3 = torch.stack(SE3)
    N, _, _ = SE3.shape  # N, 4, 4
    return SE3


SE3 = txt2se3('./calib.txt', './00.txt')
plt.scatter(SE3[:, 0, 3], SE3[:, 1, 3])
plt.show()