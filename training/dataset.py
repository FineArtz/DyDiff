# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import gym
import d4rl
from copy import deepcopy

try:
    import pyspng
except ImportError:
    pyspng = None


# ----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 name,  # Name of the dataset.
                 raw_shape,  # Shape of the raw image data (NCHW).
                 max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
                 use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
                 xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
                 random_seed=0,  # Random seed to use when applying max_size.
                 cache=False,  # Cache images in CPU memory?
                 ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict()  # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        # assert isinstance(image, np.ndarray)
        # assert list(image.shape) == self.image_shape
        # assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]
        return deepcopy(image), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        # assert len(self.image_shape) == 3 # CHW
        # assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


# ----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
                 path,  # Path to directory or zip.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 use_pyspng=True,  # Use pyspng if available?
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in
                                os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if self._use_pyspng and pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels


# ----------------------------------------------------------------------------

def get_trajectories(name, max_length_of_trajectory=7, include_action=False):
    env = gym.make(name)
    dataset = env.get_dataset()
    observations = np.array(dataset['observations'])
    actions = np.array(dataset["actions"]) if include_action else None
    datas = np.concatenate([observations, actions], axis=1) if include_action else observations
    terminals = np.array(dataset['terminals']) | np.array(dataset["timeouts"])
    # terminals[-1] = 1

    # 使用布尔索引找到终止点
    terminal_indices = np.where(terminals == 1)[0]

    trajectories = []
    raw_lengths = []
    start_idx = 0
    for end_idx in terminal_indices:
        one_trajectory = datas[start_idx:end_idx + 1]
        trajectories.append(one_trajectory)
        raw_lengths.append(len(one_trajectory))
        start_idx = end_idx + 1

    # 索引器构建
    indexer = {}
    dataset_index = 0
    for i, trajectory in enumerate(trajectories):
        traj_len = len(trajectory)
        if traj_len <= max_length_of_trajectory:
            indexer[dataset_index] = (i, 0)
            dataset_index += 1
        else:
            for j in range(traj_len - max_length_of_trajectory + 1):
                indexer[dataset_index] = (i, j)
                dataset_index += 1

    return env, trajectories, raw_lengths, indexer, datas


class MujocoDataset(Dataset):
    def __init__(self,
                 name,  # the name of the mujoco env
                 datas,  # all the datas in the datasets
                 trajectories,  # the trajectories of the mujoco env
                 lengths,  # the length of each trajectory
                 indexer,  # the reflection from dataset index to raw index
                 normalizer="LimitsNormalizer",  # the normalizer to use
                 dim=None,  # the dim of the mujoco env
                 max_length_of_trajectory=7,  # the max length of a trajectory
                 observation_dim=None,  # the observation dim of the mujoco env
                 normalize=True,  # normalize the data or not
                 use_cond=False,  # use condition or not
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):

        self.trajectories = trajectories
        self.lengths = lengths
        self.dim = dim
        self.env_name = name
        self.indexer = indexer
        self.observation_dim = observation_dim
        self.datas = datas
        self.mean, self.std = np.mean(datas, axis=0), np.std(datas, axis=0)
        self.max_length_of_trajectory = max_length_of_trajectory
        assert len(self.trajectories) == len(self.lengths), "lengths of trajectories not equal to lengths"
        self.usenormalize = normalize
        self.use_cond = use_cond
        super().__init__(name=name, raw_shape=(len(indexer), max_length_of_trajectory, dim), **super_kwargs)
        if normalizer == "GaussianNormalizer":
            self.normalizer = GaussianNormalizer(datas)
        elif normalizer == "LimitsNormalizer":
            self.normalizer = LimitsNormalizer(datas)
        else:
            raise NotImplementedError("normalizer not implemented")

    def _clip_padding_trajectory(self, trajectory: np.ndarray, local_idx, eps=1e-4):
        # return a trajectory with shape of [max_length_of_trajectory, dim]
        if len(trajectory) > self.max_length_of_trajectory:
            i_start = local_idx
            return trajectory[i_start:i_start + self.max_length_of_trajectory]
        elif len(trajectory) < self.max_length_of_trajectory:
            assert local_idx == 0, "local index must be zero for trajectories whose length is shorter than max_length_of_trajectory"
            return np.concatenate([trajectory, np.zeros(
                (self.max_length_of_trajectory - len(trajectory), trajectory.shape[1])) - 1 - eps])
        else:
            return trajectory

    def _load_raw_image(self, idx):
        raw_idx, local_idx = self.indexer[idx]
        data = self.trajectories[raw_idx]
        data = self.normalize(data)
        data = self._clip_padding_trajectory(data, local_idx)
        if self.use_cond:
            return data, self.get_conditions(data)
        else:
            return data

    def _load_raw_labels(self):
        return None

    def normalize(self, data, *args, **kwargs):
        return self.normalizer.normalize(data, *args, **kwargs)

    def unnormalize(self, data, *args, **kwargs):
        return self.normalizer.unnormalize(data, *args, **kwargs)

    def get_conditions(self, data):
        """
            condition on current observation for planning
        """
        if self.observation_dim is not None:
            return {0: data[0, :self.observation_dim]}
        else:
            return {0: data[0]}

class Normalizer:
    '''
        parent class, subclass by defining the `normalize` and `unnormalize` methods
    '''

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)
        self.mins_torch = torch.tensor(self.mins)
        self.maxs_torch = torch.tensor(self.maxs)

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()


class GaussianNormalizer(Normalizer):
    '''
        normalizes to zero mean and unit variance
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.means = self.X.mean(axis=0)
        self.stds = self.X.std(axis=0)
        self.z = 1

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    '''
            f'''means: {np.round(self.means, 2)}\n    '''
            f'''stds: {np.round(self.z * self.stds, 2)}\n'''
        )

    def normalize(self, x, idx=None):
        if idx is None:
            idx = np.arange(x.shape[-1])
        return (x[..., idx] - self.means[idx]) / self.stds[idx]

    def unnormalize(self, x, idx=None):
        if idx is None:
            idx = np.arange(x.shape[-1])
        return x[..., idx] * self.stds[idx] + self.means[idx]


class LimitsNormalizer(Normalizer):
    '''
        maps [ xmin, xmax ] to [ -1, 1 ]
    '''

    def normalize_numpy(self, x, idx=None):
        x = np.array(x)
        if idx is None:
            idx = np.arange(x.shape[-1])
        x = (x - self.mins[idx]) / (self.maxs[idx] - self.mins[idx])
        x = 2 * x - 1
        return x

    def to(self, device):
        self.mins_torch = self.mins_torch.to(device)
        self.maxs_torch = self.maxs_torch.to(device)

    def normalize_torch(self, x: torch.Tensor, idx=None):
        x = torch.clone(x)
        self.to(x.device)
        if idx is None:
            idx = np.arange(x.shape[-1])
        x = (x - self.mins_torch[idx]) / (self.maxs_torch[idx] - self.mins_torch[idx])
        x = 2 * x - 1
        return x

    def unnormalize_numpy(self, x, idx=None, eps=1e-4):
        x = np.array(x)
        if idx is None:
            idx = np.arange(x.shape[-1])
        if x.max() > 1 + eps or x.min() < -1 - eps:
            x = np.clip(x, -1, 1)
        x = (x + 1) / 2.
        return x * (self.maxs[idx] - self.mins[idx]) + self.mins[idx]

    def unnormalize_torch(self, x, idx=None, eps=1e-4):
        x = torch.clone(x)
        self.to(x.device)
        if idx is None:
            idx = np.arange(x.shape[-1])
        if x.max() > 1 + eps or x.min() < -1 - eps:
            x = torch.clip(x, -1, 1)
        x = (x + 1) / 2.
        return x * (self.maxs_torch[idx] - self.mins_torch[idx]) + self.mins_torch[idx]

    def normalize(self, x, idx=None):
        if isinstance(x, torch.Tensor):
            return self.normalize_torch(x, idx)
        else:
            return self.normalize_numpy(x, idx)

    def unnormalize(self, x, idx=None, eps=1e-4):
        if idx is None:
            idx = np.arange(x.shape[-1])
        if isinstance(x, torch.Tensor):
            return self.unnormalize_torch(x, idx)
        else:
            return self.unnormalize_numpy(x, idx)


class CondMujocoDataset(MujocoDataset):
    def __init__(self,
                 name,  # the name of the mujoco env
                 datas,  # all the datas in the datasets
                 trajectories,  # the trajectories of the mujoco env
                 lengths,  # the length of each trajectory
                 indexer,  # the reflection from dataset index to raw index
                 normalizer="LimitsNormalizer",  # the normalizer to use
                 dim=None,  # the dim of the mujoco env
                 max_length_of_trajectory=7,  # the max length of a trajectory
                 observation_dim=None,  # the observation dim of the mujoco env
                 normalize=True,  # normalize the data or not
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        super().__init__(name=name, datas=datas, trajectories=trajectories, lengths=lengths, indexer=indexer,
                         normalizer=normalizer, dim=dim, max_length_of_trajectory=max_length_of_trajectory,
                         observation_dim=observation_dim, normalize=normalize, use_cond=True, **super_kwargs)