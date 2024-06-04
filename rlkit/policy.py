import copy
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from torch_utils import persistence
from training.networks import MLP
import torch_utils.pytorch_utils as ptu

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6


@persistence.persistent_class
class TanhGaussianPolicy(MLP):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256), activation='relu',
                 max_action=1):
        super().__init__(
            input_dim=state_dim, 
            output_dim=2 * action_dim, 
            hidden_dims=hidden_dims, 
            activation=activation)
        self.max_action = max_action

    def forward(self, obs: torch.Tensor, deterministic: bool = False, return_log_prob: bool = False) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = super().forward(obs)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        log_prob = None
        if deterministic:
            action = torch.tanh(mean)
            log_std = None
        else:
            normal = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
            if return_log_prob:
                action = normal.rsample()
                log_prob = torch.sum(normal.log_prob(action), dim=-1, keepdim=True)
            else:
                action = normal.rsample()
        action = self.max_action * action
        return action, mean, log_std, log_prob
    
    def get_actions(self, obs: np.ndarray, deterministic=False) -> np.ndarray:
        obs = ptu.from_numpy(obs)
        actions, _, _, _ = self.forward(obs, deterministic)
        return ptu.get_numpy(actions)
    
    def get_action(self, obs: np.ndarray, deterministic=False) -> Tuple[np.ndarray, Dict]:
        actions = self.get_actions(obs[None], deterministic)
        return actions[0], {}
    
    def copy(self):
        return copy.deepcopy(self)


@persistence.persistent_class
class MakeDeterministic(nn.Module):
    def __init__(self, stochastic_policy: nn.Module):
        super().__init__()
        self.stochastic_policy = stochastic_policy
    
    def forward(self, *args, **kwargs):
        return self.stochastic_policy(*args, **kwargs, deterministic=True)
    
    def get_action(self, *args, **kwargs):
        return self.stochastic_policy.get_action(*args, **kwargs, deterministic=True)

    def get_actions(self, *args, **kwargs):
        return self.stochastic_policy.get_actions(*args, **kwargs, deterministic=True)

    def __getattribute__(self, __name: str):
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            p = super().__getattribute__("stochastic_policy")
            return getattr(p, __name)