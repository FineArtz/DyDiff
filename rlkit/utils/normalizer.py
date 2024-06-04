import numpy as np
import torch
import inspect


class RunningMeanStd:
    """Calulates the running mean and std of a data stream.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(self, mean=0.0, std=1.0) -> None:
        self.mean, self.var = mean, std
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        """Add a batch of item into RMS with the same shape, modify mean/var/count."""
        batch_mean, batch_var = np.mean(x, axis=0), np.var(x, axis=0)
        batch_count = len(x)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean, self.var = new_mean, new_var
        self.count = total_count


class ModelEnvNormalizer:

    def __init__(self, model, env) -> None:
        self.model = model
        self.env = env
        self.obs_len = env.observation_space.shape[0]
        self.act_len = env.action_space.shape[0]

    def _model_to_env_np(self, obs: np.ndarray) -> np.ndarray:
        return self.env._apply_normalize_obs(self.model.dataset.unnormalize(obs, idx=np.arange(self.obs_len)))
    
    def _env_to_model_np(self, obs: np.ndarray) -> np.ndarray:
        return self.model.dataset.normalize(self.env._apply_unnormalize_obs(obs), idx=np.arange(self.obs_len))
    
    def _model_to_env_torch(self, obs: torch.Tensor) -> torch.Tensor:
        return self.env._apply_normalize_obs_torch(self.model.dataset.unnormalize(obs, idx=np.arange(self.obs_len)))
    
    def _env_to_model_torch(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model.dataset.normalize(self.env._apply_unnormalize_obs_torch(obs), idx=np.arange(self.obs_len))

    def model_to_env(self, obs):
        if isinstance(obs, tuple):
            marker = obs[1]
            assert marker == 'model'
            obs = self.model_to_env(obs[0])
            return obs, 'env'
        if isinstance(obs, np.ndarray):
            return self._model_to_env_np(obs)
        elif isinstance(obs, torch.Tensor):
            return self._model_to_env_torch(obs)
        else:
            raise NotImplementedError
        
    def env_to_model(self, obs):
        if isinstance(obs, tuple):
            marker = obs[1]
            assert marker == 'env'
            obs = self.env_to_model(obs[0])
            return obs, 'model'
        if isinstance(obs, np.ndarray):
            return self._env_to_model_np(obs)
        elif isinstance(obs, torch.Tensor):
            return self._env_to_model_torch(obs)
        else:
            raise NotImplementedError

    def env_to_raw(self, obs):
        if isinstance(obs, tuple):
            marker = obs[1]
            assert marker == 'env'
            obs = self.env_to_raw(obs[0])
            return obs, 'raw'
        if isinstance(obs, np.ndarray):
            return self.env._apply_unnormalize_obs(obs)
        elif isinstance(obs, torch.Tensor):
            return self.env._apply_unnormalize_obs_torch(obs)
        else:
            raise NotImplementedError
        
    def raw_to_env(self, obs):
        if isinstance(obs, tuple):
            marker = obs[1]
            assert marker == 'raw'
            obs = self.raw_to_env(obs[0])
            return obs, 'env'
        if isinstance(obs, np.ndarray):
            return self.env._apply_normalize_obs(obs)
        elif isinstance(obs, torch.Tensor):
            return self.env._apply_normalize_obs_torch(obs)
        else:
            raise NotImplementedError

    def model_to_raw(self, obs):
        if isinstance(obs, tuple):
            marker = obs[1]
            assert marker == 'model'
            obs = self.model_to_raw(obs[0])
            return obs, 'raw'
        if isinstance(obs, np.ndarray):
            return self.model.dataset.unnormalize(obs, idx=np.arange(self.obs_len))
        elif isinstance(obs, torch.Tensor):
            return self.model.dataset.unnormalize(obs, idx=np.arange(self.obs_len))
        else:
            raise NotImplementedError
            
    def raw_to_model(self, obs):
        if isinstance(obs, tuple):
            marker = obs[1]
            assert marker == 'raw'
            obs = self.raw_to_model(obs[0])
            return obs, 'model'
        if isinstance(obs, np.ndarray):
            return self.model.dataset.normalize(obs, idx=np.arange(self.obs_len))
        elif isinstance(obs, torch.Tensor):
            return self.model.dataset.normalize(obs, idx=np.arange(self.obs_len))
        else:
            raise NotImplementedError
            
    def act_model_to_env(self, act):
        if isinstance(act, tuple):
            marker = act[1]
            assert marker == 'model'
            act = self.act_model_to_env(act[0])
            return act, 'env'
        return self.model.dataset.unnormalize(act, idx=np.arange(self.obs_len, self.obs_len + self.act_len))
    
    def act_env_to_model(self, act):
        if isinstance(act, tuple):
            marker = act[1]
            assert marker == 'env'
            act = self.act_env_to_model(act[0])
            return act, 'model'
        return self.model.dataset.normalize(act, idx=np.arange(self.obs_len, self.obs_len + self.act_len))