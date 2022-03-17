import os
import torch
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')
data_folder_out = "./results"


def save_path(num_model=None):
    if num_model is None:
        now = datetime.now()
        folder_str = now.strftime("%Y_%m_%d__%H_%M_%S")
    else:
        folder_str = str(num_model)

    print("save_path: ", folder_str)

    folder_path_models = os.path.join(data_folder_out, folder_str)
    return folder_path_models


def my_zeros(shape):
    return torch.zeros(shape, device=device, dtype=torch.float32)


def my_ones(shape):
    return torch.ones(shape, device=device, dtype=torch.float32)


def my_eye(n):
    return torch.eye(n, device=device, dtype=torch.float32)


def my_tensor(arr):
    return torch.tensor(arr, device=device, dtype=torch.float32)


def my_as_tensor(arr):
    return torch.as_tensor(arr, device=device, dtype=torch.float32)


def my_as_long_tensor(arr):
    return torch.as_tensor(arr, device=device, dtype=torch.long)


def my_long_tensor(arr):
    return torch.tensor(arr, device=device, dtype=torch.long)


def my_range(start, end, step=1):
    return torch.arange(start=start, end=end, step=step, device=device, dtype=torch.float32)


def my_linspace(start, end, steps=1):
    return torch.linspace(start=start, end=end, steps=steps, device=device, dtype=torch.float32)


def my_rand(shape):
    return torch.rand(shape, device=device, dtype=torch.float32)


def my_randn(shape):
    return torch.randn(shape, device=device, dtype=torch.float32)


def my_randperm(n):
    return torch.randperm(n, device=device, dtype=torch.long)


def my_randint(low, up, size):
    return torch.randint(low, up, size, device=device, dtype=torch.long)


def dist_mat(x, y, inplace=True):
    d = torch.mm(x, y.transpose(0, 1))
    v_x = torch.sum(x ** 2, 1).unsqueeze(1)
    v_y = torch.sum(y ** 2, 1).unsqueeze(0)
    d *= -2
    if inplace:
        d += v_x
        d += v_y
    else:
        d = d + v_x
        d = d + v_y

    return d


def nn_search(y, x):
    d = dist_mat(x, y)
    return torch.argmin(d, dim=1)


def print_memory_status(pos=None):
    if torch.torch.cuda.is_available():
        tot = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        res = torch.cuda.memory_reserved(0) // (1024**2)
        all = torch.cuda.memory_allocated(0) // (1024**2)
        free = res - all
        if pos is not None:
            print("Memory status at pos", pos, ": tot =", tot, ", res =", res, ", all =", all, ", free =", free)
        else:
            print("Memory status: tot =", tot, ", res =", res, ", all =", all, ", free =", free)
        return torch.cuda.memory_reserved(0) / 1024**2
    else:
        print("Memory status: CUDA NOT AVAILABLE")