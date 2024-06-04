import os
from typing import List
from collections import OrderedDict
import gtimer as gt
import numpy as np
import time
from torch import nn as nn
import torch

from rlkit.logger import logger
from rlkit.policy import MakeDeterministic
from rlkit.utils.path_sampler import PathSampler, VecPathSampler
from rlkit.envs.vecenvs import BaseVectorEnv
from rlkit.utils.buffer import ReplayBuffer, EnvReplayBuffer
import rlkit.utils.eval_util as eval_util
import torch_utils.pytorch_utils as ptu


class BaseOfflineTrainer:
    def __init__(
        self,
        env,
        algo,  # model-free algorithm to train policy
        exploration_policy,
        # model params
        replay_buffer: ReplayBuffer = None,
        replay_buffer_size: int = 1000000,
        # training params
        num_epochs: int = 501,
        batch_size: int = 256, 
        num_train_steps_per_train_call: int = 1000,
        num_steps_per_eval: int = 5000,
        max_path_length: int = 1000,
        freq_saving: int = 10,
        
        save_best: bool = False,
        save_replay_buffer: bool = False,
        save_epoch: bool = False,
        best_key: str = "AverageReturn",
        debug: bool = False,
    ):  
        self.env = env
        self.algo = algo
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
    
        if replay_buffer is None:
            replay_buffer = EnvReplayBuffer(
                replay_buffer_size, self.env, random_seed=np.random.randint(10000)
            )
        self.replay_buffer = replay_buffer
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_train_steps_per_train_call = num_train_steps_per_train_call
        self.freq_saving = freq_saving
        self._n_train_steps_total = 0
        
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
        self.save_best = save_best
        self.save_epoch = save_epoch
        self.best_key = best_key
        self.best_statistic_so_far = -np.inf
        self.eval_statistics = None
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
        pass

    def start_training(self, start_epoch: int = 0) -> None:
        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):            
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
        batch = self.replay_buffer.random_batch(self.batch_size)
        batch = ptu.np_to_pytorch_batch(batch)
        return batch

    def _do_training(self) -> None:
        for _ in range(self.num_train_steps_per_train_call):
            self.algo.train_step(self.get_batch())
            if self.debug:
                print("Trainer | The agent training loop breaks for debug")
                break
    
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
        self.algo.to(device)
        
    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(epoch=epoch)
        data_to_save.update(self.algo.get_snapshot())
        return data_to_save
    
    def get_extra_data_to_save(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            n_train_steps_total=self._n_train_steps_total,
        )
        if self.save_replay_buffer:
            data_to_save["replay_buffer"] = self.replay_buffer
        return data_to_save
        
    def _try_to_eval(self, epoch):
        # save if it's time to save
        if (self.freq_saving <= 0) and (epoch < self.num_epochs-1):
            pass
        else:
            if (int(epoch) % self.freq_saving == 0) or (epoch + 1 >= self.num_epochs):
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
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
        self.eval_statistics.update(self.algo.get_eval_statistics())
        
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
