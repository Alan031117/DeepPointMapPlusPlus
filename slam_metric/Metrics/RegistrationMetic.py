import os
import torch
import argparse

parser = argparse.ArgumentParser(description='Evaluate registration metric RRE & RTE')

# 基本参数
parser.add_argument('--prediction_file', '-pred', type=str, help='KITTI type SE3 prediction file path')
parser.add_argument('--gt_file', '-gt', type=str, help='KITTI type SE3 ground truth file path')
parser.add_argument('--recall_R_threshold', '-r', type=float, default=5, help='RRE threshold for recall, degree system')
parser.add_argument('--recall_T_threshold', '-t', type=float, default=2, help='RTE threshold for recall, metric system')


def Mat_to_RPY(rot_matrix):
    """
    convert a rotation matrix to rpy (Roll-Pitch-Yaw) angle
    rpy_angle: torch.Tensor, shape [bs, 3, 3]

    return: torch.Tensor, shape [bs, 3]
    """
    def isRotationMatrix(R):
        bs = R.shape[0]
        Rt = R.permute(0, 2, 1)
        shouldBeIdentity = torch.bmm(Rt, R)
        I = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device=R.device)
        n = torch.linalg.norm(I - shouldBeIdentity) / bs
        return n < 1e-2

    batch = True
    if (rot_matrix.ndim == 2):
        batch = False
        rot_matrix = rot_matrix.unsqueeze(0)
    assert (isRotationMatrix(rot_matrix)), f'{rot_matrix} is not orthogonal matrix'
    sy = torch.sqrt(rot_matrix[:, 0, 0] * rot_matrix[:, 0, 0] + rot_matrix[:, 1, 0] * rot_matrix[:, 1, 0])
    x = torch.atan2(rot_matrix[:, 2, 1], rot_matrix[:, 2, 2])
    y = torch.atan2(-rot_matrix[:, 2, 0], sy)
    z = torch.atan2(rot_matrix[:, 1, 0], rot_matrix[:, 0, 0])

    Rs = torch.stack([x, y, z], dim=1)
    if (batch == False):
        Rs = Rs[0]
    return Rs


def eval_reg(gt_file_path, pred_file_path):
    command = f'python "{__file__}" -pred "{pred_file_path}" -gt "{gt_file_path}"'
    reg_result = ParseReg(os.popen(command).read())
    return reg_result


def ParseReg(evo_response):
    '''
    prase response of registration metric system call
    (e.g, python toolbox/eval_registration.py -pred xxx.txt -gt xxx.txt)
    evo_response: str, result of system call
    '''
    result = dict()
    for line in evo_response.splitlines():
        line_strip = line.strip()
        if (line_strip.startswith("RRE")):
            result['RRE'] = float(line.split()[1])
        elif (line_strip.startswith("RTE")):
            result['RTE'] = float(line.split()[1])
        elif (line_strip.startswith("RR")):
            result['RR'] = float(line.split()[1])
    return result


def read_rt(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'{file_path} is not found')
    R_list, T_list = [], []
    with open(file_path, 'r') as f:
        for se3 in f.readlines():
            RT = torch.tensor([float(i) for i in se3.split()], dtype=torch.float).reshape(3, 4)
            R = RT[:, :3]
            T = RT[:, -1]
            R_list.append(R)
            T_list.append(T)
    R_list = torch.stack(R_list, dim=0)
    T_list = torch.stack(T_list, dim=0)
    return R_list, T_list


def calculate_relative_rt(R_list, T_list):
    """
    R @ pre + T = next
    pre_R @ pre + pre_T = next_R @ next + next_T
    pre_R @ pre + pre_T - next_T = next_R @ next
    next_R.T(pre_R @ pre + pre_T - next_T) = next
    (next_R.T @ pre_R) @ pre + next_R.T @ (pre_T - next_T) = next

    R = next_R.T @ pre_R
    T = next_R.T @ (pre_T - next_T)
    """
    R_pre, T_pre = R_list[:-1], T_list[:-1]
    R_next, T_next = R_list[1:], T_list[1:]
    RR = R_next.transpose(1, 2) @ R_pre
    RT = R_next.transpose(1, 2) @ (T_pre - T_next).unsqueeze(-1)
    return RR, RT.squeeze(-1)


def calculate_relative_error(RR_pred, RT_pred, RR_gt, RT_gt, recall_R_threshold=5, recall_T_threshold=2):
    angle = Mat_to_RPY(RR_gt.transpose(1, 2) @ RR_pred)
    angle = angle * 180 / torch.pi
    RRE = torch.sum(torch.abs(angle), dim=-1)
    RTE = torch.norm(RT_gt - RT_pred, p=2, dim=-1) * 100
    recall = (RRE < recall_R_threshold) & (RTE < (recall_T_threshold * 100))
    RR = torch.sum(recall) / max(recall.shape[0], 1)
    return RRE.mean(), RTE.mean(), RR


if __name__ == "__main__":
    args = parser.parse_args()
    prediction_file = args.prediction_file
    gt_file = args.gt_file
    recall_R_threshold = args.recall_R_threshold
    recall_T_threshold = args.recall_T_threshold

    # 读取R T
    R_pred, T_pred = read_rt(prediction_file)
    R_gt, T_gt = read_rt(gt_file)
    assert R_pred.shape[0] > 1, 'At least two pose nodes are required to calculate the relative error'

    # 计算任意相邻帧间的相对R T
    RR_pred, RT_pred = calculate_relative_rt(R_pred, T_pred)
    RR_gt, RT_gt = calculate_relative_rt(R_gt, T_gt)

    # 计算配准指标RRE RTE RR
    RRE, RTE, RR = calculate_relative_error(RR_pred, RT_pred, RR_gt, RT_gt, recall_R_threshold, recall_T_threshold)
    print(f'''
        RRE\t{RRE:.10} degree
        RTE\t{RTE:.10} cm
        RR \t{RR:.10} 
    ''')
