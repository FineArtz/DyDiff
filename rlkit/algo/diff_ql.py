from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy

import torch_utils.pytorch_utils as ptu
from torch_utils.helpers import EMA
from rlkit.utils.eval_util import create_stats_ordered_dict


class Diffusion_QL:
    def __init__(
        self,
        policy: nn.Module,
        qf1: nn.Module,
        qf2: nn.Module,
        discount=0.99,
        tau=0.005,
        max_q_backup=False,
        eta=1.0,
        ema_decay=0.995,
        step_start_ema=1000,
        update_ema_every=5,
        lr=3e-4,
        lr_decay=False,
        optimizer_class=optim.Adam,
        lr_maxt=1000,
        grad_norm=1.0,
        use_pred_start=True,
        **kwargs
    ):
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = qf1.copy()
        self.target_qf2 = qf2.copy()
        self.ema_model = copy.deepcopy(self.policy)
        
        self.policy_optimizer = optimizer_class(policy.parameters(), lr=lr)
        self.qf1_optimizer = optimizer_class(qf1.parameters(), lr=lr)
        self.qf2_optimizer = optimizer_class(qf2.parameters(), lr=lr)
        self.lr_decay = lr_decay
        self.grad_norm = grad_norm
        if lr_decay:
            self.policy_lr_scheduler = CosineAnnealingLR(self.policy_optimizer, T_max=lr_maxt, eta_min=0.)
            self.qf1_lr_scheduler = CosineAnnealingLR(self.qf1_optimizer, T_max=lr_maxt, eta_min=0.)
            self.qf2_lr_scheduler = CosineAnnealingLR(self.qf2_optimizer, T_max=lr_maxt, eta_min=0.)

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.update_ema_every = update_ema_every

        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.max_q_backup = max_q_backup
        self.use_pred_start = use_pred_start
        self.eval_statistics = None

    def train_step(self, batch):
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]
        batch_size = obs.shape[0]

        """ Q Training """
        current_q1 = self.qf1(obs, actions)
        current_q2 = self.qf2(obs, actions)

        with torch.inference_mode():
            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_obs, repeats=10, dim=0)
                next_action_rpt, _, _, _ = self.ema_model(next_state_rpt)
                target_q1 = self.target_qf1(next_state_rpt, next_action_rpt)
                target_q2 = self.target_qf2(next_state_rpt, next_action_rpt)
                target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action, _, _, _ = self.ema_model(next_obs)
                target_q1 = self.target_qf1(next_obs, next_action)
                target_q2 = self.target_qf2(next_obs, next_action)
                target_q = torch.min(target_q1, target_q2)

            target_q = rewards + (1.0 - terminals) * self.discount * target_q

        qf1_loss = torch.mean((current_q1 - target_q) ** 2)
        qf2_loss = torch.mean((current_q2 - target_q) ** 2)
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        if self.grad_norm > 0:
            nn.utils.clip_grad_norm_(self.qf1.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.qf1_optimizer.step()
        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        if self.grad_norm > 0:
            nn.utils.clip_grad_norm_(self.qf2.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.qf2_optimizer.step()

        """ Policy Training """
        if self.use_pred_start:
            bc_loss, new_action = self.policy.loss_and_xrecon(actions, obs)
        else:
            bc_loss = self.policy.loss(actions, obs)
            new_action, _, _, _ = self.policy(obs)

        q1_new_action = self.qf1(obs, new_action)
        q2_new_action = self.qf2(obs, new_action)
        if np.random.uniform() > 0.5:
            q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
        else:
            q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
        policy_loss = bc_loss + self.eta * q_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        if self.grad_norm > 0: 
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.policy_optimizer.step()

        """ Step Target network """
        if self.step % self.update_ema_every == 0:
            self.step_ema()
        self._update_target_network()

        self.step += 1

        """ Log """
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
            self.eval_statistics['QF1 Loss'] = qf1_loss.item()
            self.eval_statistics['QF2 Loss'] = qf2_loss.item()
            self.eval_statistics['Policy Loss'] = policy_loss.item()
            self.eval_statistics['BC Loss'] = bc_loss.item()
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Q1 Predictions', 
                    ptu.get_numpy(current_q1)
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Q2 Predictions', 
                    ptu.get_numpy(current_q2)
                )
            )
            
    @property
    def networks(self):
        return [
            self.policy,
            self.ema_model,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]
    
    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.policy)
    
    def _update_target_network(self):
        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.tau)
        ptu.soft_update_from_to(self.qf2, self.target_qf2, self.tau)

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            ema_model=self.ema_model,
            policy_optimizer=self.policy_optimizer,
            qf1_optimizer=self.qf1_optimizer,
            qf2_optimizer=self.qf2_optimizer,
            step=self.step,
        )
        
    def load_snapshot(self, snapshot):
        self.policy = snapshot['policy']
        self.qf1 = snapshot['qf1']
        self.qf2 = snapshot['qf2']
        self.target_qf1 = snapshot['target_qf1']
        self.target_qf2 = snapshot['target_qf2']
        self.ema_model = snapshot['ema_model']
        self.policy_optimizer = snapshot['policy_optimizer']
        self.qf1_optimizer = snapshot['qf1_optimizer']
        self.qf2_optimizer = snapshot['qf2_optimizer']
        self.step = snapshot['step']
        
    def get_eval_statistics(self):
        return self.eval_statistics
    
    def end_epoch(self):
        self.eval_statistics = None
        if self.lr_decay:
            self.policy_lr_scheduler.step()
            self.qf1_lr_scheduler.step()
            self.qf2_lr_scheduler.step()

    def to(self, device):
        for net in self.networks:
            net.to(device)