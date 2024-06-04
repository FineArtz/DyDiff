from collections import OrderedDict
from typing import Dict, List, Union
from itertools import count
import os
import torch
import torch.nn as nn
import pickle
import numpy as np

from rlkit.utils.buffer import ReplayBuffer
from rlkit.logger import logger
from training.networks import BNN
import torch_utils.pytorch_utils as ptu


class DynamicsTrainer(torch.nn.Module):
    def __init__(
        self,
        model: BNN,
        replay_buffer: ReplayBuffer,
        optimizer = torch.optim.Adam,
        creterion = nn.MSELoss,
        lr: float = 1e-3,
        weight_decays: Union[float, List] = None,
        num_elites: int = None,
        reward_scale: float = 1.0,
        batch_size: int = 256,
        train_steps: int = None,  # if not setting max_epochs, training will stop following the early stopping scheme
        holdout_ratio: float = 0.2,
        max_holdout: int = 5000,
        early_stopping_patience: int = 5,  # for early stopping
        log_freq: int = 100,  # log frequency, -1 for silence
        save_path: str = './logs/dynamics_model',
        save_name: str = 'model',
        debug: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.buffer = replay_buffer
        self.weight_decays = weight_decays
        num_hidden = len(self.model.layers) - 1
        if self.weight_decays is None:
            self.weight_decays = (
                [2.5e-5, 5e-5] + [7.5e-5] * (num_hidden - 2) + [1e-4]
                if num_hidden > 2
                else [2.5e-5, 1e-4]
            )
        elif isinstance(self.weight_decays, float):
            self.weight_decays = [self.weight_decays] * (num_hidden + 1)
        elif isinstance(self.weight_decays, list):
            assert len(self.weight_decays) == num_hidden + 1
        self.num_nets = self.model.num_nets
        self.num_elites = self.num_nets if num_elites is None else num_elites
        self.num_elites = min(self.num_elites, self.num_nets)
        self.best_idx = None

        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.holdout_ratio = holdout_ratio
        self.max_holdout = max_holdout
        self.early_stopping_patience = early_stopping_patience
        self.log_freq = log_freq
        self._state: Dict[int, List[OrderedDict[str : torch.Tensor]]] = {}
        self.normalize_obs = kwargs.get('normalize_obs', False)
        self.use_gaussian = kwargs.get('use_gaussian', False)
        self.pred_reward = kwargs.get('pred_reward', False)

        opt_param_groups = [
            {"params": fc.parameters(), "weight_decay": wd}
            for fc, wd in zip(self.model.layers, self.weight_decays)
        ]
        opt_param_groups += [{"params": [self.model.min_log_var, self.model.max_log_var]}]
        self.optimizer = optimizer(opt_param_groups, lr=lr)
        self.params_dict = dict(
            lr=lr,
            weight_decay=weight_decays,
            batch_size=batch_size,
            holdout_ratio=holdout_ratio,
            max_holdout=max_holdout,
            early_stopping_patience=early_stopping_patience,
            normalize_obs=self.normalize_obs,
            use_gaussian=self.use_gaussian,
            pred_reward=self.pred_reward,
        )
        self.save_file = os.path.join(save_path, f'{save_name}.pkl')
        self.is_loaded = False
        self.debug = debug

    def _compute_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor, add_var_loss: bool = True
    ) -> torch.Tensor:
        if self.use_gaussian:
            mean, log_var = self.model(inputs, return_log_var=True)
            inv_var = torch.exp(-log_var)
            if add_var_loss:
                loss = torch.mean((mean - targets) ** 2 * inv_var, dim=[-2, -1])
                var_loss = torch.mean(log_var, dim=[-2, -1])
                loss = loss + var_loss
            else:
                loss = torch.mean((mean - targets) ** 2, dim=[-2, -1])
        else:
            mean = self.model(inputs)
            loss = torch.mean(self.loss_fn(mean, targets), dim=[-2, -1])
        return loss

    def train(self):
        if self.is_loaded:
            logger.log(f'Dynamics model already loaded, skipping training')
            return
        data = self.buffer.get_all()
        data = ptu.np_to_pytorch_batch(data)
        obs = data["observations"]
        act = data["actions"]
        next_obs = data["next_observations"]
        # not include the reward
        inputs = torch.cat((obs, act), dim=-1)
        if self.pred_reward:
            reward = self.reward_scale * data["rewards"]
            targets = torch.cat((reward, next_obs - obs), dim=-1)
        else:
            targets = next_obs - obs

        num_ho = min(self.max_holdout, int(len(obs) * self.holdout_ratio))
        ho_idx = np.random.choice(len(obs), num_ho, replace=False)
        ho = np.zeros(len(obs), dtype=bool)
        ho[ho_idx] = True
        ho_inputs = inputs[ho]
        ho_inputs = torch.tile(ho_inputs[None], dims=[self.num_nets, 1, 1])
        ho_targets = targets[ho]
        ho_targets = torch.tile(ho_targets[None], dims=[self.num_nets, 1, 1])
        tr_inputs = inputs[~ho]
        tr_targets = targets[~ho]
        best_loss = [float("inf") for _ in range(self.num_nets)]
        patience = 0
        train_iter = range(self.train_steps) if self.train_steps else count()
        total_epoch = 0
        idxs = np.random.randint(tr_inputs.shape[0], size=[self.num_nets, tr_inputs.shape[0]])
        for i in train_iter:
            # shuffle
            idx = np.random.permutation(len(tr_inputs))
            tr_inputs = tr_inputs[idx]
            tr_targets = tr_targets[idx]
            total_epoch += 1
            st = 0
            while st < len(tr_inputs):
                ed = min(st + self.batch_size, len(tr_inputs))
                inputs = ptu.tf_like_gather(tr_inputs, idxs[:, st:ed])
                targets = ptu.tf_like_gather(tr_targets, idxs[:, st:ed])

                self.optimizer.zero_grad()
                mse_loss = torch.mean(self._compute_loss(inputs, targets, add_var_loss=True))
                loss = (
                    mse_loss
                    + 0.01 * torch.mean(self.model.max_log_var)
                    - 0.01 * torch.mean(self.model.min_log_var)
                )
                loss.backward()
                self.optimizer.step()
                st = ed

            with torch.inference_mode():
                pred_loss = self._compute_loss(ho_inputs, ho_targets, add_var_loss=False)
                pred_loss = ptu.get_numpy(pred_loss)
                update = False
                for net_id in range(self.num_nets):
                    if pred_loss[net_id] < best_loss[net_id]:
                        best_loss[net_id] = pred_loss[net_id]
                        self._save_state(net_id)
                        update = True
                if update:
                    patience = 0
                else:
                    patience += 1
                if patience > self.early_stopping_patience:
                    break
            if self.log_freq > 0 and i % self.log_freq == 0:
                logger.log(f"Dynamics model training | Iter {i}, Loss: {loss.item()}, Holdout Loss: {pred_loss.mean()}")
            if self.debug:
                logger.log(f"Dynamics model training breaks for DEBUG")
                break

        self._set_state()
        with torch.inference_mode():
            final_loss = self._compute_loss(ho_inputs, ho_targets, add_var_loss=False)
            final_loss = ptu.get_numpy(final_loss)
        self.params_dict["loss"] = final_loss
        for layer_id in range(len(self.model.layers)):
            for net_id in range(self.model.num_nets):
                for k in self._state[net_id][layer_id].keys():
                    self._state[net_id][layer_id][k] = self._state[net_id][layer_id][k].cpu()
        self.params_dict["model_params"] = self._state
        
        self.best_idx = np.argsort(final_loss)[:self.num_elites].tolist()
        self.params_dict["best_idx"] = self.best_idx
        final_loss.sort()
        val_loss = final_loss[:self.num_elites].mean()
        logger.log(f"Dynamics model | Final validation loss: {val_loss}")
        with open(self.save_file, "wb") as f:
            pickle.dump(self.params_dict, f)
        logger.log(f'Epoch {total_epoch} | Dynamics model saved to {self.save_file}')

    def forward(self, obs: torch.Tensor, act: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        inputs = torch.cat((obs, act), dim=-1)
        mean, variance = self.model.predict(inputs, factored=True)
        if self.pred_reward:
            mean[:, :, 1:] += obs
        else:
            mean += obs
        std = torch.sqrt(variance)
        samples = mean if deterministic else mean + std * torch.randn_like(mean)
        
        model_idx = np.random.choice(self.best_idx, size=samples.shape[1])
        next_obs = samples[model_idx, np.arange(samples.shape[1])]
        if self.pred_reward:
            return next_obs[:, 1:], next_obs[:, 0]
        else:
            return next_obs
    
    def predict(self, obs: np.ndarray, act: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if single := (obs.ndim == 1):
            obs = obs[None]
            act = act[None]
        obs = ptu.from_numpy(obs)
        act = ptu.from_numpy(act)
        if self.pred_reward:
            next_obs, reward = self.forward(obs, act, deterministic=deterministic)
            next_obs = ptu.get_numpy(next_obs)
            reward = ptu.get_numpy(reward)
            if single:
                next_obs = next_obs[0]
                reward = reward[0]
            return next_obs, reward
        else:
            next_obs = self.forward(obs, act, deterministic=deterministic)
            next_obs = ptu.get_numpy(next_obs)
            if single:
                next_obs = next_obs[0]
            return next_obs
    
    def load_model(self, path):
        logger.log(f"Loading dynamics model from {path}")
        with open(path, "rb") as f:
            self.params_dict = pickle.load(f)
        self._state = self.params_dict["model_params"]
        self._set_state()
        self.normalize_obs = self.params_dict.get('normalize_obs', self.normalize_obs)
        self.use_gaussian = self.params_dict.get('use_gaussian', self.use_gaussian)
        self.pred_reward = self.params_dict.get('pred_reward', self.pred_reward)
        self.best_idx = self.params_dict.get('best_idx', np.arange(self.num_nets))
        logger.log(f'Dynamics model loaded')
        self.is_loaded = True
        
    def get_snapshot(self):
        return dict(dynamics_model=self.model)
    
    @property
    def networks(self):
        return [self.model]

    def to(self, device):
        self.model.to(device)
        
    def _save_state(self, net_id: int) -> None:
        self._state[net_id] = [
            OrderedDict({"weight": fc.weight.data, "bias": fc.bias.data})
            for fc in self.model.layers
        ]

    def _set_state(self):
        for layer_id in range(len(self.model.layers)):
            for net_id in range(self.model.num_nets):
                layer = self.model.layers[layer_id]
                layer.load_state_dict(self._state[net_id][layer_id])
