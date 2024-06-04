from typing import Callable, Dict, List
from collections import OrderedDict
import os
import gtimer as gt
import numpy as np
import time
from torch import nn as nn
import torch

from rlkit.logger import logger
from rlkit.fake_env import TrajFakeEnv
from rlkit.policy import MakeDeterministic
from rlkit.utils.normalizer import ModelEnvNormalizer
from rlkit.utils.path_sampler import PathSampler, VecPathSampler
from rlkit.envs.vecenvs import BaseVectorEnv
from rlkit.utils.buffer import ReplayBuffer, EnvReplayBuffer
import rlkit.utils.eval_util as eval_util
import torch_utils.pytorch_utils as ptu


class OfflineTrainer:
    def __init__(
        self,
        env,
        model,
        reward_model,           # can be None
        algo,                   # model-free algorithm to train policy
        is_terminal: Callable,  # for fake_env
        exploration_policy,
        # model params
        traj_sampler: Callable,
        replay_buffer: ReplayBuffer = None,
        replay_buffer_size: int = 1000000,
        model_replay_buffer: ReplayBuffer = None,
        model_replay_buffer_size: int = 1000000,
        deterministic: bool = False,
        rollout_batch_size: int = int(1e5),
        each_rollout_size: int = 8, # the number of sample for each rollout to avoid OOM
        real_ratio: float = 0.8,
        include_action: bool = True,
        rollout_length: int = 5,
        generate_length: int = 5,
        rollout_freq: int = 10,
        # training params
        num_epochs: int = 501,
        batch_size: int = 256, 
        num_train_steps_per_train_call: int = 1000,
        num_steps_per_eval: int = 5000,
        max_path_length: int = 1000,
        freq_saving: int = 10,
        
        save_best: bool = False,
        save_replay_buffer: bool = False,
        save_model_replay_buffer: bool = False,
        save_epoch: bool = False,
        best_key: str = "AverageReturn",
        debug: bool = False,
        sampler_kwargs: Dict = {},
        keep_condition: Dict = {},
        normalizer: ModelEnvNormalizer = None,
        checkpt_per_epoch: int = 100,
    ):  
        self.env = env
        self.model = model
        self.reward_model = reward_model
        self.algo = algo
        self.is_terminal = is_terminal
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.normalizer = normalizer
        self.checkpt_per_epoch = checkpt_per_epoch
        
        self.fake_env = TrajFakeEnv(
            model=model, 
            sampler=traj_sampler, 
            observation_dim=self.obs_dim,
            action_dim=self.act_dim,
            include_action=include_action, 
            is_terminal=is_terminal,
            normalizer=normalizer
        )
        if replay_buffer is None:
            replay_buffer = EnvReplayBuffer(
                replay_buffer_size, self.env, random_seed=np.random.randint(10000)
            )
        self.replay_buffer = replay_buffer
        self.model_replay_buffer_size = model_replay_buffer_size
        if model_replay_buffer is None:
            assert max_path_length < model_replay_buffer_size
            model_replay_buffer = EnvReplayBuffer(
                model_replay_buffer_size, self.env, random_seed=np.random.randint(10000)
            )
        else:
            assert max_path_length < model_replay_buffer._max_replay_buffer_size
        self.model_replay_buffer = model_replay_buffer

        self.deterministic = deterministic
        self.rollout_batch_size = rollout_batch_size
        self.each_rollout_size = each_rollout_size
        self.real_ratio = real_ratio
        self.rollout_length = rollout_length
        self.generate_length = generate_length
        if self.rollout_length is None or self.rollout_length < 0 or self.rollout_length > self.generate_length:
            self.rollout_length = self.generate_length
        self.rollout_freq = rollout_freq
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_train_steps_per_train_call = num_train_steps_per_train_call
        self.freq_saving = freq_saving
        self._n_train_steps_total = 0
        
        self.sampler_kwargs = sampler_kwargs
        self.sampler_kwargs["is_normalized"] = True
        
        eval_policy = MakeDeterministic(exploration_policy)
        if isinstance(env, BaseVectorEnv):
            self.eval_sampler = VecPathSampler(
                env,
                eval_policy,
                num_steps_per_eval,
                max_path_length,
                no_terminal=False
            )
        else:
            self.eval_sampler = PathSampler(
                env,
                eval_policy,
                num_steps_per_eval,
                max_path_length,
                no_terminal=False
            )
    
        self.save_replay_buffer = save_replay_buffer
        self.save_model_replay_buffer = save_model_replay_buffer
        self.save_best = save_best
        self.save_epoch = save_epoch
        self.best_key = best_key
        self.best_statistic_so_far = -np.inf
        self.eval_statistics = None
        self.keep_condition = keep_condition
        self.debug = debug
        
    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)
            
    def train(self, start_epoch=0):
        self.pretrain()
        if start_epoch == 0:
            params = self.get_epoch_snapshot(-1)
            logger.save_itr_params(-1, params)
        self.training_mode(False)
        gt.reset()
        gt.set_def_unique(False)
        self.start_training(start_epoch=start_epoch)
        
    def pretrain(self):
        logger.log(f"Model pretraining")
        # nothing to do
        gt.stamp("model_train")

    def start_training(self, start_epoch: int = 0) -> None:
        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            if epoch % self.rollout_freq == 0:
                gt.subdivide("model_rollout", save_itrs=True)
                self._rollout_stat = self._rollout_model(self.deterministic)
                gt.end_subdivision()
            else:
                self._rollout_stat = None
            
            self._epoch_start_time = time.time()
            logger.push_prefix("Iteration #%d | " % epoch)
            
            self.training_mode(True)
            self._do_training()
            self._n_train_steps_total += 1
            self.training_mode(False)
            gt.stamp("train")

            self._try_to_eval(epoch)
            gt.stamp("eval")
            self._end_epoch()
            
            if self.debug:
                print("Trainer | The main loop breaks for debug")
                break

    def get_batch(self) -> torch.Tensor:
        if int(self.batch_size * (1 - self.real_ratio)) > self.model_replay_buffer.num_steps_can_sample():
            real_ratio = 1.0
        else:
            real_ratio = self.real_ratio
        real_batch_size = int(self.batch_size * real_ratio)
        model_batch_size = self.batch_size - real_batch_size

        real_batch = self.replay_buffer.random_batch(real_batch_size)
        if model_batch_size > 0:
            model_batch = self.model_replay_buffer.random_batch(model_batch_size)
            batch = {
                k: np.concatenate([real_batch[k], model_batch[k]], axis=0)
                for k in real_batch.keys()
            }
            batch = ptu.np_to_pytorch_batch(batch)
        else:
            batch = ptu.np_to_pytorch_batch(real_batch)
        return batch

    def _do_training(self) -> None:
        for _ in range(self.num_train_steps_per_train_call):
            self.algo.train_step(self.get_batch())
            if self.debug:
                print("Trainer | The agent training loop breaks for debug")
                break
        
    def sample_data_by_reward_softmax(self, data, reward, percentage, temp=0.05, eps=1e-6):
        reward = np.float128(reward)
        if temp <= 0:
            reward_exp = np.exp(reward - reward.mean())
            probs = reward_exp / np.sum(reward_exp)
        else:
            normalized_reward = (reward - reward.mean()) / (reward.std() + eps)
            reward_exp = np.exp(normalized_reward / temp)
            probs = reward_exp / np.sum(reward_exp) * (1 - eps) + eps / reward_exp.shape[0]
        
        batch_size = data.shape[0]
        sampled = np.random.choice(batch_size, size=int(percentage * batch_size), replace=False, p=np.float64(probs))
        indices = np.zeros(batch_size, dtype=bool)
        indices[sampled] = True
        return indices

    def filter_idx(self, traj: np.array, results, **kwargs) -> np.array:
        if self.keep_condition:
            threshold = self.keep_condition["threshold"]
            if self.keep_condition["filter_type"] == "hardmax":
                idx = (results >= np.quantile(results, 1 - threshold))
            elif self.keep_condition["filter_type"] == "softmax":
                idx = self.sample_data_by_reward_softmax(traj, results, threshold, **kwargs)
            else:
                raise NotImplementedError
            return idx
        else:
            return np.arange(traj.shape[0])

    def _rollout_model(self, deterministic: bool = False, **kwargs) -> Dict[str, float]:
        if self.real_ratio == 1.0:
            return {"mean_rollout_length": 0}
        batch = self.replay_buffer.random_batch(self.rollout_batch_size)
        obs = batch["observations"]
        
        self.sampler_kwargs["policy"] = self.fake_env.sampler._get_wrapped_policy(self.algo.policy)
        st = 0
        traj = None
        if self.debug:
            print("===== BEGIN ROLLOUT =====")
        # seperate rollout batch into smaller batches to avoid OOM
        while st < self.rollout_batch_size:
            ed = min(st + self.each_rollout_size, self.rollout_batch_size)
            batch_obs = obs[st: ed]
            batch_traj = self.fake_env.step(obs=batch_obs, sample_len=self.generate_length, 
                                            **self.sampler_kwargs, **kwargs)
            if traj is None:
                traj = batch_traj
            else:
                traj = np.concatenate([traj, batch_traj], axis=0)
            st = ed
        gt.stamp('rollout')
        traj = traj[:, :self.rollout_length + 1, :] # +1 for next_obs

        # compute rewards with reward model
        with torch.inference_mode():
            obs_t = ptu.from_numpy(traj[:, :, :self.obs_dim]).reshape(-1, self.obs_dim)
            act_t = ptu.from_numpy(traj[:, :, self.obs_dim:]).reshape(-1, self.act_dim)
            rew = self.reward_model(obs_t, act_t)
            rew = ptu.get_numpy(rew).reshape(traj.shape[0], traj.shape[1])
            rew = np.sum(rew, axis=1)
        gt.stamp('rollout_eval')

        idx = self.filter_idx(traj, rew, temp=self.keep_condition.get("softmax_temp", 0.05))
        traj = traj[idx]
        obs = obs[idx]
        if traj.size > 0:
            is_term = np.zeros(traj.shape[0], dtype=bool)
            total_steps = 0
            for i in range(traj.shape[1] - 1): # x.shape is (batch, s+a)
                next_obs, act = traj[:, i + 1, :self.obs_dim], traj[:, i, self.obs_dim:]
                with torch.inference_mode():
                    obs_t = ptu.from_numpy(obs)
                    act_t = ptu.from_numpy(act)
                    rew = self.reward_model(obs_t, act_t)
                    rew = ptu.get_numpy(rew)
                raw_obs = self.env._apply_unnormalize_obs(obs)
                raw_next_obs = self.env._apply_unnormalize_obs(next_obs)
                cur_term = self.is_terminal(raw_obs, act, raw_next_obs).reshape(-1)
                samples = {
                    "observations": obs[~is_term],
                    "actions": act[~is_term],
                    "next_observations": next_obs[~is_term],
                    "rewards": rew[~is_term],
                    "terminals": cur_term[~is_term],
                }
                self.model_replay_buffer.add_path(samples)
                total_steps += np.sum(cur_term[~is_term]) * (i + 1)
                is_term = np.logical_or(is_term, cur_term)
                if np.all(is_term):
                    logger.log(
                        f"Model Rollout | Breaking early at {i}: all episodes terminate"
                    )
                    break
                obs = next_obs
            total_steps += np.sum(~is_term) * (traj.shape[1] - 1)
            
            mean_len = total_steps / np.sum(idx)
        else: 
            total_steps = 0
            mean_len = 0
        logger.log(
            f"Model Rollout | Added: {total_steps:.1e} | Model pool: {self.model_replay_buffer._size:.1e} (max {self.model_replay_buffer._max_replay_buffer_size:.1e}) | Mean length: {mean_len} | Train rep: {self.num_train_steps_per_train_call}"
        )
        gt.stamp('rollout_add')
        stats_dict = {
            "mean_rollout_length": mean_len
        }
        return stats_dict
    
    def _end_epoch(self):
        print("Trainer end epoch")
        self.algo.end_epoch()
        self.eval_statistics = None
        logger.log("Epoch Duration: {0}".format(time.time() - self._epoch_start_time))
        logger.pop_prefix()

    @property
    def networks(self) -> List[nn.Module]:
        return self.algo.networks

    def to(self, device):
        self.model.to(device)
        self.reward_model.to(device)
        self.algo.to(device)
        
    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(epoch=epoch)
        data_to_save.update(self.reward_model.get_snapshot())
        data_to_save.update(self.algo.get_snapshot())
        return data_to_save
    
    def get_extra_data_to_save(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            n_train_steps_total=self._n_train_steps_total,
            best_statistic_so_far=self.best_statistic_so_far,
        )
        if self.save_replay_buffer:
            data_to_save["replay_buffer"] = self.replay_buffer
        if self.save_model_replay_buffer is not None:
            data_to_save["model_replay_buffer"] = self.model_replay_buffer
        return data_to_save
        
    def _try_to_eval(self, epoch):
        if (self.freq_saving <= 0) and (epoch < self.num_epochs-1):
            pass
        else:
            if (int(epoch) % self.freq_saving == 0) or (epoch + 1 >= self.num_epochs):
                # if epoch + 1 >= self.num_epochs:
                # epoch = 'final'
                logger.save_extra_data(self.get_extra_data_to_save(epoch))
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)

        self.evaluate(epoch)

        logger.record_tabular(
            "Number of train calls total",
            self._n_train_steps_total,
        )
        
        times = gt.get_times()
        times_itrs = times.stamps.itrs
        train_time = times_itrs["train"][-1]
        if "sample" in times_itrs:
            sample_time = times_itrs["sample"][-1]
        else:
            sample_time = 0
        
        if "eval" in times_itrs:
            eval_time = times_itrs["eval"][-1] if epoch > 0 else 0
        else:
            eval_time = 0
        epoch_time = train_time + sample_time + eval_time
        total_time = gt.get_times().total

        logger.record_tabular("Train Time (s)", train_time)
        logger.record_tabular("(Previous) Eval Time (s)", eval_time)
        logger.record_tabular("Sample Time (s)", sample_time)
        logger.record_tabular("Epoch Time (s)", epoch_time)
        logger.record_tabular("Total Train Time (s)", total_time)
        for k, sub_times in times.subdvsn.items():
            for sub_time in sub_times:
                sub_itrs = sub_time.stamps.itrs
                for k2, v in sub_itrs.items():
                    logger.record_tabular(f"{sub_time.name}/{k2} (s)", v[-1])

        logger.record_tabular("Epoch", epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
    
    def evaluate(self, epoch: int) -> None:
        """
        Remove exploration paths for offline training
        Evaluate the policy, e.g. save/print progress.
        :param epoch:
        :return:
        """
        if hasattr(self.model, "get_eval_statistics"):
            self.eval_statistics = self.model.get_eval_statistics()
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
        self.eval_statistics.update(self.algo.get_eval_statistics())
        if self._rollout_stat:
            self.eval_statistics.update(self._rollout_stat)
        
        statistics = OrderedDict()
        try:
            statistics.update(self.eval_statistics)
            self.eval_statistics = None
        except Exception as e:
            print("No Stats to Eval", str(e))

        logger.log("Collecting samples for evaluation")
        test_paths = self.eval_sampler.obtain_samples()

        statistics.update(
            eval_util.get_generic_path_information(
                test_paths,
                stat_prefix="Test",
            )
        )

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(test_paths)
        if hasattr(self.env, "log_statistics"):
            statistics.update(self.env.log_statistics(test_paths))

        average_returns = eval_util.get_average_returns(test_paths)
        statistics["AverageReturn"] = average_returns
        for key, value in statistics.items():
            try:
                logger.record_tabular(key, np.mean(value))
            except Exception:
                print(f"Log error with key: {key}, value: {value}")

        best_statistic = statistics[self.best_key]
        data_to_save = {"epoch": epoch, "statistics": statistics}
        data_to_save.update(self.get_epoch_snapshot(epoch))
        if self.save_epoch:
            logger.save_extra_data(data_to_save, "epoch{}.pkl".format(epoch))
            print("\n\nSAVED MODEL AT EPOCH {}\n\n".format(epoch))
        data_to_save = {"epoch": epoch, "statistics": statistics}
        data_to_save.update(self.get_epoch_snapshot(epoch))
        if best_statistic > self.best_statistic_so_far:
            self.best_statistic_so_far = best_statistic
            if self.save_best:
                logger.save_extra_data(data_to_save, "best.pkl")
                print("\n\nSAVED BEST\n\n")
        logger.save_extra_data(data_to_save, "snapshot-{}.pkl".format(epoch))
        if os.path.exists(os.path.join(logger.get_snapshot_dir(), f"snapshot-{epoch - 10}.pkl")):
            if (epoch - 10) % 50 != 0:
                os.remove(os.path.join(logger.get_snapshot_dir(), f"snapshot-{epoch - 10}.pkl"))

    def load_snapshot(self, model_data):
        self.algo.load_snapshot(model_data)
        self.algo.to(ptu.device)
        self.eval_sampler.policy = MakeDeterministic(self.algo.policy)

    def set_steps(self, extra_data):
        self._n_train_steps_total = extra_data["n_train_steps_total"]
        self.model_replay_buffer_size = self.model_replay_buffer._max_replay_buffer_size
        self.best_statistic_so_far = extra_data["best_statistic_so_far"]
