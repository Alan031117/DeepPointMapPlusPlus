# 位姿矩阵相关功能
import numpy as np
from torch import Tensor as Tensor
from typing import Tuple


def rt_global_to_relative(center_R: Tensor, center_T: Tensor, other_R: Tensor, other_T: Tensor) -> Tuple[Tensor, Tensor]:
    """
    全局rt转相对rt
    R @ src + T = dst
    src_R @ src + src_T = dst_R @ dst + dst_T
    dst_R.T(src_R @ src + src_T - dst_T) = dst
    R = dst_R.T @ src_R
    T = dst_R.T @ (src_T - dst_T)

    :param center_R: 相对变化的中心帧旋转矩阵
    :param center_T: 相对变化的中心帧平移矩阵
    :param other_R: 相对变化的其他帧旋转矩阵
    :param other_T: 相对变化的其他帧旋转矩阵
    :return:
        relative_R, relative_T
        relative_R.shape == other_r.shape
        relative_T.shape == other_t.shape
    """
    relative_R = center_R.transpose(-1, -2) @ other_R
    relative_T = center_R.transpose(-1, -2) @ (other_T - center_T)
    return relative_R, relative_T


def rt_global_to_relative_np(center_R: np.ndarray, center_T: np.ndarray, other_R: np.ndarray, other_T: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    全局rt转相对rt
    R @ src + T = dst
    src_R @ src + src_T = dst_R @ dst + dst_T
    dst_R.T(src_R @ src + src_T - dst_T) = dst
    R = dst_R.T @ src_R
    T = dst_R.T @ (src_T - dst_T)

    :param center_R: 相对变化的中心帧旋转矩阵
    :param center_T: 相对变化的中心帧平移矩阵
    :param other_R: 相对变化的其他帧旋转矩阵
    :param other_T: 相对变化的其他帧旋转矩阵
    :return:
        relative_R, relative_T
        relative_R.shape == other_r.shape
        relative_T.shape == other_t.shape
    """
    relative_R = np.swapaxes(center_R, -1, -2) @ other_R
    relative_T = np.swapaxes(center_R, -1, -2) @ (other_T - center_T)
    return relative_R, relative_T

