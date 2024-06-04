from itertools import count
import os
import torch
import torch.nn as nn
import pickle
import numpy as np

from rlkit.utils.buffer import ReplayBuffer
from torch_utils.pytorch_utils import np_to_pytorch_batch
import torch_utils.pytorch_utils as ptu
from rlkit.logger import logger


class RewModelTrainer(nn.Module):
    def __init__(
        self, 
        model,
        replay_buffer: ReplayBuffer,
        optimizer = torch.optim.Adam,
        creterion = nn.MSELoss,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        train_steps: int = None,
        holdout_ratio: float = 0.2,
        max_holdout: int = 5000,
        early_stopping_patience: int = 7,
        log_freq: int = 20,
        save_path: str = './logs/reward_model',
        save_name: str = 'model',
        debug: bool = False,
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.buffer = replay_buffer
        self.loss_fn = creterion()
        self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.holdout_ratio = holdout_ratio
        self.max_holdout = max_holdout
        self.early_stopping_patience = early_stopping_patience
        self.log_freq = log_freq
        self.best_model_params = None
        self.normalize_obs = kwargs.get('normalize_obs', False)
        self.params_dict = dict(
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            holdout_ratio=holdout_ratio,
            max_holdout=max_holdout,
            early_stopping_patience=early_stopping_patience,
            normalize_obs=self.normalize_obs,
        )
        self.save_file = os.path.join(save_path, f'{save_name}.pkl')
        self.is_loaded = False
        self.debug = debug
        
    def train(self):
        if self.is_loaded:
            logger.log(f'Reward model already loaded, skipping training')
            return
        data = self.buffer.get_all()
        data = np_to_pytorch_batch(data)
        obs = data["observations"]
        act = data["actions"]
        rew = data["rewards"]
        num_ho = min(self.max_holdout, int(len(obs) * self.holdout_ratio))
        ho_idx = np.random.choice(len(obs), num_ho, replace=False)
        ho = np.zeros(len(obs), dtype=bool)
        ho[ho_idx] = True
        ho_obs = obs[ho]
        ho_act = act[ho]
        ho_rew = rew[ho]
        tr_obs = obs[~ho]
        tr_act = act[~ho]
        tr_rew = rew[~ho]
        best_loss = float("inf")
        patience = 0
        train_iter = range(self.train_steps) if self.train_steps else count()
        total_epoch = 0
        for i in train_iter:
            total_epoch += 1
            # shuffle
            idx = np.random.permutation(len(tr_obs))
            tr_obs = tr_obs[idx]
            tr_act = tr_act[idx]
            tr_rew = tr_rew[idx]
            st = 0
            while st < len(tr_obs):
                ed = min(st + self.batch_size, len(tr_obs))
                obs = tr_obs[st:ed]
                act = tr_act[st:ed]
                rew = tr_rew[st:ed]
            
                self.optimizer.zero_grad()
                pred_rew = self.model(obs, act)
                loss = self.loss_fn(pred_rew, rew)
                loss.backward()
                self.optimizer.step()
                st = ed
            
            with torch.inference_mode():
                pred_rew = self.model(ho_obs, ho_act)
                pred_loss = self.loss_fn(pred_rew, ho_rew)
                if pred_loss < best_loss:
                    best_loss = pred_loss
                    patience = 0
                    self.best_model_params = self.model.state_dict()
                else:
                    patience += 1
                    if patience > self.early_stopping_patience:
                        break
            if i % self.log_freq == 0:
                logger.log(f'Reward model training | Iter {i}, Loss: {loss.item()}, Holdout Loss: {pred_loss.item()}')
            if self.debug:
                logger.log(f'Reward model training breaks for DEBUG')
                break
                
        self.params_dict['loss'] = best_loss.item()
        for k in self.best_model_params.keys():
            self.best_model_params[k] = self.best_model_params[k].cpu()
        self.params_dict['model_params'] = self.best_model_params
        with open(self.save_file, 'wb') as f:
            pickle.dump(self.params_dict, f)
        logger.log(f'Epoch {total_epoch} | Reward model saved to {self.save_file}')
                    
    def forward(self, obs, act):
        return self.model(obs, act)
    
    def load_model(self, path):
        logger.log(f'Loading reward model from {path}')
        with open(path, 'rb') as f:
            self.params_dict = pickle.load(f)
        self.model.load_state_dict(self.params_dict['model_params'])
        self.normalize_obs = self.params_dict.get('normalize_obs', self.normalize_obs)
        del self.params_dict['model_params']
        logger.log(f'Reward model loaded')
        self.is_loaded = True
        
    def get_snapshot(self):
        return dict(reward_model=self.model)

    @property
    def networks(self):
        return [self.model]
    
    def to(self, device):
        self.model.to(device)