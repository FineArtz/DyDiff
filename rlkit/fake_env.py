from functools import reduce
from typing import Callable, Dict, List, Tuple
import torch
import numpy as np

from rlkit.utils.normalizer import ModelEnvNormalizer
import torch_utils.pytorch_utils as ptu

        
class TrajFakeEnv:
    def __init__(self,
        model,
        sampler: Callable,
        observation_dim: int,
        action_dim: int,
        include_action: bool,
        is_terminal: Callable,
        normalizer: ModelEnvNormalizer,
    ):
        self.model = model
        if isinstance(model, EnsembleModel):
            self.sampler = EnsembleSampler(sampler, normalizer)
        else:
            self.sampler = sampler
        self.observation_dim = observation_dim
        self.data_dim = observation_dim + action_dim if include_action else observation_dim
        self.is_terminal = is_terminal
        self.normalizer = normalizer
        
    def step(self, obs: np.ndarray, sample_len: int = 1, **kwargs) -> Tuple[np.array, float, bool, Dict]:
        if single := (obs.ndim == 1):
            obs = obs[None]
        if "cond_actions" in kwargs:
            kwargs["cond_actions"] = self.normalizer.act_env_to_model(kwargs['cond_actions'])
        # obs: env scaled obs
        obs = self.normalizer.env_to_model(obs)
        # obs: model scaled obs
        obs = ptu.from_numpy(obs)
        samples = self.step_tensor(obs, sample_len, **kwargs)
        if log_stats := kwargs.get('log_stats', False):
            samples, log_stats = samples
        samples = ptu.get_numpy(samples)
        obs, act = samples[..., :self.observation_dim], samples[..., self.observation_dim:]
        # obs: model scaled obs
        # act: model scaled act
        obs = self.normalizer.model_to_env(obs)
        # obs: env scaled obs
        act = self.normalizer.act_model_to_env(act)
        # act: env scaled act
        samples = np.concatenate([obs, act], axis=-1)
        if log_stats:
            return samples, log_stats
        return samples
        
    def step_tensor(self, obs: torch.Tensor, sample_len: int = 1, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        batch_size = obs.shape[0]
        noise = ptu.randn(size=(batch_size, sample_len, self.data_dim))
        samples = self.sampler(self.model, noise, conditions={0: obs}, observation_dim=self.observation_dim, **kwargs)
        return samples
    

class EnsembleModel:
    def __init__(self, models: List[torch.nn.Module]):
        self.models = models
        self.num_models = len(models)
        self._check_consistency('dataset_mean', 'dataset_std', 'rl_setting', 'dataset.env_name', 'dataset.max_length_of_trajectory')
        self.dataset = self.models[0].dataset
        for model in self.models[1:]:
            del model.dataset
            if not hasattr(model, "use_cond"):
                model.use_cond = False

    def _check_consistency(self, *attrs: List[str]):
        for attr in attrs:
            sub_attrs = attr.split('.')
            val0 = reduce(getattr, sub_attrs, self.models[0])
            for model in self.models[1:]:
                val = reduce(getattr, sub_attrs, model)
                eq = None
                if isinstance(val0, np.ndarray):
                    eq = np.all(val == val0)
                elif isinstance(val0, torch.Tensor):
                    eq = torch.all(val == val0)
                else:
                    eq = val == val0
                assert eq, f"{attr} is not consistent among models."

    def gen_model_idx(self, batch_size: int) -> np.ndarray:
        return np.random.randint(self.num_models, size=batch_size)
    
    def normalize(self, inputs: np.ndarray) -> np.ndarray:
        return self.dataset.normalize(inputs)
    
    def unnormalize(self, inputs: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        return self.dataset.unnormalize(inputs)
    
    def to(self, device: torch.device):
        for model in self.models:
            model.to(device)
        return self
    

class EnsembleSampler:
    def __init__(self, sampler: Callable, normalizer: ModelEnvNormalizer):
        self.sampler = sampler
        self.normalizer = normalizer
    
    def _get_wrapped_policy(self, gn: Callable):
        def _wrapped_policy(obs: torch.Tensor, *args, **kwargs):
            # obs: model scaled obs
            obs = self.normalizer.model_to_env(obs)
            # obs: env scaled obs
            outputs = list(gn(obs, *args, **kwargs))
            act = outputs[0]
            # act: env scaled act
            act = self.normalizer.act_env_to_model(act)
            # act: model scaled act
            outputs[0] = act
            return tuple(outputs)
        return _wrapped_policy

    def __call__(self, ens_model: EnsembleModel, noise: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        samples = None
        batch_size = noise.shape[0]
        model_idxs = ens_model.gen_model_idx(batch_size)
        samples = ptu.zeros_like(noise) # batch_size * sample_len * data_dim
        for i, model in enumerate(ens_model.models):
            model_i_kwargs = kwargs.copy()
            if 'conditions' in kwargs:
                model_i_condition = kwargs['conditions'][0][model_idxs == i]
                model_i_kwargs['conditions'] = {0: model_i_condition}
            model_samples = self.sampler(model, noise[model_idxs == i], *args, **model_i_kwargs)
            samples[model_idxs == i] = model_samples.to(torch.float32)
        return samples