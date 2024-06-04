from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch_utils.pytorch_utils as ptu
from rlkit.utils.eval_util import create_stats_ordered_dict

LOG_ALPHA_MIN = -10.0
LOG_ALPHA_MAX = 0.5

class CQL:
    def __init__(
        self,
        policy: nn.Module,
        qf1: nn.Module,
        qf2: nn.Module,
        env,
        reward_scale=1.0,
        discount=0.99,
        policy_lr=1e-4,
        qf_lr=3e-4,
        soft_target_tau=5e-3,
        optimizer_class=optim.Adam,
        target_entropy=None,
        # CQL params
        temp=1.0,
        cql_weight=1.0,
        num_random=10,
        importance_sampling=True,
        max_backup=False,
        lagrange=False,
        target_action_gap=0.0,
        grad_norm=0.0,
        debug=False,
        **kwargs
    ) -> None:
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.reward_scale = reward_scale
        self.discount = discount
        self.soft_target_tau = soft_target_tau
        
        self.target_entropy = target_entropy
        if target_entropy is None:
            self.target_entropy = -np.prod(env.action_space.shape).item()
        self.log_alpha = ptu.zeros(1, requires_grad=True)

        self.target_qf1 = qf1.copy()
        self.target_qf2 = qf2.copy()

        self.eval_statistics = None

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.alpha_optimizer = optimizer_class(
            [self.log_alpha], lr=policy_lr
        )

        # cql
        self.temp = temp
        self.cql_weight = cql_weight
        self.num_random = num_random
        self.importance_sampling = importance_sampling
        self.max_backup = max_backup
        self.lagrange = lagrange
        self.target_action_gap = target_action_gap
        self.grad_norm = grad_norm
        if self.lagrange:
            self.log_alpha_prime = ptu.zeros(1, requires_grad=True)
            self.alpha_prime_optimizer = optimizer_class(
                [self.log_alpha_prime], lr=qf_lr
            )
        
        self.debug = debug

    def _get_tensor_values(self, obs, actions, network):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int (action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds
    
    def _get_policy_actions(self, obs, num_actions, network):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        new_obs_actions, _, _, new_obs_log_pi, *_ = network(
            obs_temp, return_log_prob=True,
        )
        return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)

    def train_step(self, batch):
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        rewards = batch['rewards']
        dones = batch['terminals']

        """
        Alpha loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, return_log_prob=True,
        )
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        alpha = torch.clamp(self.log_alpha, LOG_ALPHA_MIN, LOG_ALPHA_MAX).exp()

        """
        Policy loss
        """
        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        """
        QF TD loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        
        with torch.no_grad():
            if self.max_backup:
                # for sparse reward
                next_act_rep, _ = self._get_policy_actions(next_obs, num_actions=self.num_random, network=self.policy)
                target_q_values, _ = torch.max(
                    torch.min(
                        self._get_tensor_values(next_obs, next_act_rep, self.target_qf1),
                        self._get_tensor_values(next_obs, next_act_rep, self.target_qf2),
                    ),
                    dim=-1,
                )
            else:
                new_next_actions, *_ = self.policy(
                    next_obs, return_log_prob=True,
                )
                target_q_values = torch.min(
                    self.target_qf1(next_obs, new_next_actions),
                    self.target_qf2(next_obs, new_next_actions),
                )
            q_target = self.reward_scale * rewards + (1. - dones) * self.discount * target_q_values
        
        qf1_td_loss = F.mse_loss(q1_pred, q_target)
        qf2_td_loss = F.mse_loss(q2_pred, q_target)
        
        """
        QF CQL loss
        """
        random_actions_tensor = torch.rand(q2_pred.shape[0] * self.num_random, actions.shape[-1]).uniform_(-1, 1).to(ptu.device)
        curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, num_actions=self.num_random, network=self.policy)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs, num_actions=self.num_random, network=self.policy)
        q1_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf1)
        q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf2)
        q1_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf1)
        q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf2)
        q1_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf1)
        q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf2)
    
        cat_q1 = torch.cat(
            [q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1
        )
        cat_q2 = torch.cat(
            [q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1
        )
        if self.importance_sampling:
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            cat_q1 = torch.cat(
                [q1_rand - random_density, q1_next_actions - new_log_pis.detach(), q1_curr_actions - curr_log_pis.detach()], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density, q2_next_actions - new_log_pis.detach(), q2_curr_actions - curr_log_pis.detach()], 1
            )
            
        min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1,).mean() * self.temp - q1_pred.mean()
        min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1,).mean() * self.temp - q2_pred.mean() 
        
        if self.lagrange:
            alpha_prime = self.log_alpha_prime.exp()
            min_qf1_loss = alpha_prime * self.cql_weight * (min_qf1_loss - self.target_action_gap)
            min_qf2_loss = alpha_prime * self.cql_weight * (min_qf2_loss - self.target_action_gap)
            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = -(min_qf1_loss + min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            min_qf1_loss = min_qf1_loss * self.cql_weight
            min_qf2_loss = min_qf2_loss * self.cql_weight
        
        qf1_loss = qf1_td_loss + min_qf1_loss
        qf2_loss = qf2_td_loss + min_qf2_loss
        
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        if self.grad_norm > 0:
            nn.utils.clip_grad_norm_(self.qf1.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.qf1_optimizer.step()
        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        if self.grad_norm > 0:
            nn.utils.clip_grad_norm_(self.qf2.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.qf2_optimizer.step()
        
        """
        Update networks
        """
        self._update_target_network()
        
        """
        Save statistic for eval
        """
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
            self.eval_statistics["Reward Scale"] = self.reward_scale
            self.eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics["QF1 TD Loss"] = np.mean(ptu.get_numpy(qf1_td_loss))
            self.eval_statistics["QF2 TD Loss"] = np.mean(ptu.get_numpy(qf2_td_loss))
            self.eval_statistics["QF1 CQL Loss"] = np.mean(ptu.get_numpy(min_qf1_loss))
            self.eval_statistics["QF2 CQL Loss"] = np.mean(ptu.get_numpy(min_qf2_loss))
            self.eval_statistics["Alpha"] = alpha.item()
            self.eval_statistics["Alpha Loss"] = np.mean(ptu.get_numpy(alpha_loss))
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
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q1 in-distribution",
                    ptu.get_numpy(q1_curr_actions),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q2 in-distribution",
                    ptu.get_numpy(q2_curr_actions),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q1 random",
                    ptu.get_numpy(q1_rand),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q2 random",
                    ptu.get_numpy(q2_rand),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q1 next_actions",
                    ptu.get_numpy(q1_next_actions),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q2 next_actions",
                    ptu.get_numpy(q2_next_actions),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Log Pis",
                    ptu.get_numpy(log_pi),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy mu",
                    ptu.get_numpy(policy_mean),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy log std",
                    ptu.get_numpy(policy_log_std),
                )
            )
            if self.lagrange:
                self.eval_statistics["Alpha Prime"] = alpha_prime.item()
                self.eval_statistics["Alpha Prime Loss"] = np.mean(ptu.get_numpy(alpha_prime_loss))
            
    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]
    
    def _update_target_network(self):
        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
        ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)
        
    def get_snapshot(self):
        snapshot = dict(
            qf1=self.qf1,
            qf2=self.qf2,
            policy=self.policy,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            log_alpha=self.log_alpha,
            policy_optimizer=self.policy_optimizer,
            qf1_optimizer=self.qf1_optimizer,
            qf2_optimizer=self.qf2_optimizer,
            alpha_optimizer=self.alpha_optimizer,
        )
        if self.lagrange:
            snapshot["log_alpha_prime"] = self.log_alpha_prime
            snapshot["alpha_prime_optimizer"] = self.alpha_prime_optimizer
        return snapshot

    def load_snapshot(self, snapshot):
        self.qf1 = snapshot["qf1"]
        self.qf2 = snapshot["qf2"]
        self.policy = snapshot["policy"]
        self.target_qf1 = snapshot["target_qf1"]
        self.target_qf2 = snapshot["target_qf2"]
        self.log_alpha = snapshot["log_alpha"]
        self.policy_optimizer = snapshot["policy_optimizer"]
        self.qf1_optimizer = snapshot["qf1_optimizer"]
        self.qf2_optimizer = snapshot["qf2_optimizer"]
        self.alpha_optimizer = snapshot["alpha_optimizer"]
        if self.lagrange:
            self.log_alpha_prime = snapshot["log_alpha_prime"]
            self.alpha_prime_optimizer = snapshot["alpha_prime_optimizer"]
        
    def get_eval_statistics(self):
        return self.eval_statistics
    
    def end_epoch(self):
        self.eval_statistics = None

    def to(self, device):
        self.log_alpha = self.log_alpha.to(device)
        for net in self.networks:
            net.to(device)
