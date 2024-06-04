from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torch_utils.pytorch_utils as ptu
from rlkit.utils.eval_util import create_stats_ordered_dict


class TD3_BC:
    def __init__(
        self,
        policy: nn.Module,
        qf1: nn.Module,
        qf2: nn.Module,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        alpha=2.5,
        policy_lr=3e-4,
        qf_lr=3e-4,
        optimizer_class=optim.Adam,
        beta_1=0.9,
        **kwargs
    ):
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.target_policy = policy.copy()
        self.target_qf1 = qf1.copy()
        self.target_qf2 = qf2.copy()

        self.eval_statistics = None

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(), lr=policy_lr, betas=(beta_1, 0.999)
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(), lr=qf_lr, betas=(beta_1, 0.999)
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(), lr=qf_lr, betas=(beta_1, 0.999)
        )

        self.total_it = 0

    def train_step(self, batch):
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        """
        QF Loss
        """
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)

        with torch.inference_mode():
            next_policy_outputs = self.target_policy(next_obs, return_log_prob=True)
            (
                next_actions,
                next_policy_mean,
                next_policy_log_std,
                next_log_pi,
            ) = next_policy_outputs[:4]            
            target_q1_values = self.target_qf1(next_obs, next_actions)
            target_q2_values = self.target_qf2(next_obs, next_actions)
            target_q_values = torch.min(target_q1_values, target_q2_values)
            q_target = rewards + (1.0 - terminals) * self.discount * target_q_values

        qf1_loss = 0.5 * torch.mean((q1_pred - q_target) ** 2)
        qf2_loss = 0.5 * torch.mean((q2_pred - q_target) ** 2)

        qf1_loss.backward()
        qf2_loss.backward()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        """
        Policy Loss
        """
        self.total_it += 1

        if self.total_it % self.policy_freq == 0:
            self.policy_optimizer.zero_grad()
            next_policy_outputs = self.policy(obs, return_log_prob=True)
            (
                policy_actions,
                policy_mean,
                policy_log_std,
                log_pi,
            ) = next_policy_outputs[:4]
            q_values = self.qf1(obs, policy_actions)
            lmbda = self.alpha / q_values.abs().mean().detach()

            policy_loss = -lmbda * q_values.mean() + 0.5 * torch.mean((policy_actions - actions) ** 2)
            policy_loss.backward()
            self.policy_optimizer.step()

            """
            Update Target Networks
            """
            self._update_target_networks()

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
            self.eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(qf2_loss))
            if self.total_it % self.policy_freq == 0:
                self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q1 Predictions",
                    ptu.get_numpy(q1_pred),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q2 Predictions",
                    ptu.get_numpy(q2_pred),
                )
            )
            if self.total_it % self.policy_freq == 0:
                self.eval_statistics.update(
                    create_stats_ordered_dict(
                        "Policy Actions",
                        ptu.get_numpy(policy_actions),
                    )
                )

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_policy,
            self.target_qf1,
            self.target_qf2,
        ]

    def _update_target_networks(self):
        ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.tau)
        ptu.soft_update_from_to(self.qf2, self.target_qf2, self.tau)

    def get_snapshot(self):
        return dict(
            qf1=self.qf1,
            qf2=self.qf2,
            policy=self.policy,
            target_policy=self.target_policy,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            policy_optimizer=self.policy_optimizer,
            qf1_optimizer=self.qf1_optimizer,
            qf2_optimizer=self.qf2_optimizer,
        )

    def load_snapshot(self, snapshot):
        self.qf1 = snapshot["qf1"]
        self.qf2 = snapshot["qf2"]
        self.policy = snapshot["policy"]
        self.target_policy = snapshot["target_policy"]
        self.target_qf1 = snapshot["target_qf1"]
        self.target_qf2 = snapshot["target_qf2"]
        self.policy_optimizer = snapshot["policy_optimizer"]
        self.qf1_optimizer = snapshot["qf1_optimizer"]
        self.qf2_optimizer = snapshot["qf2_optimizer"]

    def get_eval_statistics(self):
        return self.eval_statistics

    def end_epoch(self):
        self.eval_statistics = None

    def to(self, device):
        for net in self.networks:
            net.to(device)
