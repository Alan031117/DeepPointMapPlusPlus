from typing import Any  # 导入 Any 类型，用于指定函数的返回值类型可以是任意类型
import torch  # 导入 PyTorch 库，用于张量操作
import numpy as np  # 导入 NumPy 库，用于数组操作
from functools import partial  # 导入 partial 函数，用于部分应用函数的参数

def _single_move_to_device(x, device='cpu', detach: bool = True, non_blocking: bool = False):
    # 定义一个内部函数，将单个张量或数组移动到指定设备（如 GPU 或 CPU）

    if isinstance(x, torch.Tensor) and x.device != device:  # 如果 x 是一个 PyTorch 张量且当前设备与目标设备不同
        if detach:  # 如果需要将张量从计算图中分离（detach）
            return x.detach().to(device, non_blocking=non_blocking)  # 分离张量并移动到目标设备
        else:  # 如果不需要分离张量
            return x.to(device, non_blocking=non_blocking)  # 直接移动张量到目标设备

    if isinstance(x, np.ndarray):  # 如果 x 是一个 NumPy 数组
        if x.dtype == np.int16 or x.dtype == np.int32 or x.dtype == np.int64 or x.dtype == np.int128:
            dtype = torch.int64  # 如果 NumPy 数组的数据类型是整数类型，转换为 PyTorch 的 int64 类型
        elif x.dtype == np.float32 or x.dtype == np.float64:
            dtype = torch.float32  # 如果 NumPy 数组的数据类型是浮点类型，转换为 PyTorch 的 float32 类型
        elif x.dtype == np.bool8:
            dtype = torch.bool  # 如果 NumPy 数组的数据类型是布尔类型，转换为 PyTorch 的 bool 类型
        else:
            raise TypeError()  # 如果数据类型不匹配，抛出类型错误

        if detach:  # 如果需要分离张量
            return torch.tensor(x, dtype=dtype, device=device).detach()  # 将 NumPy 数组转换为 PyTorch 张量并分离
        else:  # 如果不需要分离张量
            return torch.tensor(x, dtype=dtype, device=device)  # 直接将 NumPy 数组转换为 PyTorch 张量并移动到目标设备

    else:  # 如果 x 既不是张量也不是数组
        return x  # 直接返回原始值

def detach_to_device(x, device, non_blocking: bool = False) -> Any:
    # 定义一个函数，将数据移动到指定设备，并在需要时分离张量（detach）

    if x is None:  # 如果 x 为 None
        return None  # 直接返回 None

    if isinstance(x, tuple):  # 如果 x 是一个元组
        return (_single_move_to_device(i, device, non_blocking=non_blocking) for i in x)  # 将元组中的每个元素移动到设备上
    if isinstance(x, list):  # 如果 x 是一个列表
        return [_single_move_to_device(i, device, non_blocking=non_blocking) for i in x]  # 将列表中的每个元素移动到设备上
    else:  # 如果 x 既不是元组也不是列表
        return _single_move_to_device(x, device, non_blocking=non_blocking)  # 直接移动单个元素到设备上

def move_to_device(x, device, non_blocking: bool = False):
    # 定义一个函数，将数据移动到指定设备，但不分离张量（不使用 detach）

    if x is None:  # 如果 x 为 None
        return None  # 直接返回 None

    if isinstance(x, tuple):  # 如果 x 是一个元组
        return (_single_move_to_device(i, device, detach=False, non_blocking=non_blocking) for i in x)  # 将元组中的每个元素移动到设备上，且不分离张量
    if isinstance(x, list):  # 如果 x 是一个列表
        return [_single_move_to_device(i, device, detach=False, non_blocking=non_blocking) for i in x]  # 将列表中的每个元素移动到设备上，且不分离张量
    else:  # 如果 x 既不是元组也不是列表
        return _single_move_to_device(x, device, detach=False, non_blocking=non_blocking)  # 直接移动单个元素到设备上，且不分离张量

detach_to_cpu = partial(detach_to_device, device='cpu')  # 创建一个新的函数 detach_to_cpu，它是 detach_to_device 的部分应用，固定设备为 'cpu'

