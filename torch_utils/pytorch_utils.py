"""
Replaced using new version of rlkit
2e8097f2a2c4d4d5079f12b8210f8ac8f250aee0
"""
import torch
import numpy as np
import os
import abc


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

"""
GPU wrappers
"""

_use_gpu = False
device = None
_gpu_id = 0


def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    if _use_gpu and gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")
    device = torch.device("cuda:" + str(0) if _use_gpu else "cpu")


def gpu_enabled():
    return _use_gpu


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


# noinspection PyPep8Naming
def FloatTensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.FloatTensor(*args, **kwargs, device=torch_device)


def from_numpy(np_array, requires_grad=False):
    # tensor_array = torch.from_numpy(np_array).float().to(device)
    tensor_array = torch.from_numpy(np_array.astype(np.float32)).to(device) # float() creates a new tensor, making it much more slow
    tensor_array.requires_grad_(requires_grad=requires_grad)
    return tensor_array


def get_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.to("cpu").detach().numpy()


def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)


def ones_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones_like(*args, **kwargs, device=torch_device)


def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs, device=torch_device)


def rand(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.rand(*args, **kwargs, device=torch_device)


def zeros_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros_like(*args, **kwargs, device=torch_device)


def full(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.full(*args, **kwargs, device=torch_device)


def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs, device=torch_device)


def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)


def tf_like_gather(x: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    # perform gather like tensorflow
    return torch.cat([x[j][None] for j in i])


def tf_like_gather_np(x: np.ndarray, i: np.ndarray) -> np.ndarray:
    return np.concatenate([x[j][None] for j in i])


def print_grad_hook(grad):
    # for debug
    print(grad)


def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(_elem_or_tuple_to_variable(e) for e in elem_or_tuple)
    return from_numpy(elem_or_tuple, requires_grad=False).float()


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    return {
        k: _elem_or_tuple_to_variable(x)
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype("O")  # ignore object (e.g. dictionaries)
    }

        
class ScalarSchedule(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_value(self, t):
        pass


class LinearSchedule(ScalarSchedule):
    """
    Linearly interpolate and then stop at a final value.
    """
    def __init__(
            self,
            init_value,
            final_value,
            ramp_duration,
    ):
        self._init_value = init_value
        self._final_value = final_value
        self._ramp_duration = ramp_duration

    def get_value(self, t):
        return (
            self._init_value
            + (self._final_value - self._init_value)
            * min(1.0, t * 1.0 / self._ramp_duration)
        )