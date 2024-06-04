from itertools import starmap
import numpy as np
import pickle

from rlkit.utils.eval_util import get_dim


class ReplayBuffer:
    """
    THE MAX LENGTH OF AN EPISODE SHOULD BE STRICTLY SMALLER THAN THE max_replay_buffer_size
    OTHERWISE THERE IS A BUG IN TERMINATE_EPISODE

    # It's a bit memory inefficient to save the observations twice,
    # but it makes the code *much* easier since you no longer have to
    # worry about termination conditions.
    """

    def __init__(
        self, max_replay_buffer_size, observation_dim, action_dim, random_seed=1995
    ):
        self._np_rand_state = np.random.RandomState(random_seed)

        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size

        obs_dtype = np.uint8 if type(observation_dim) == tuple else np.float64

        if isinstance(observation_dim, tuple):
            dims = [d for d in observation_dim]
            dims = [max_replay_buffer_size] + dims
            dims = tuple(dims)
            self._observations = np.zeros(dims, dtype=obs_dtype)
            self._next_obs = np.zeros(dims, dtype=obs_dtype)
        elif isinstance(observation_dim, dict):
            # assuming that this is a one-level dictionary
            self._observations = {}
            self._next_obs = {}

            for key, dims in observation_dim.items():
                if isinstance(dims, tuple):
                    dims = tuple([max_replay_buffer_size] + list(dims))
                else:
                    dims = (max_replay_buffer_size, dims)
                self._observations[key] = np.zeros(dims, dtype=obs_dtype)
                self._next_obs[key] = np.zeros(dims, dtype=obs_dtype)
        else:
            # else observation_dim is an integer
            self._observations = np.zeros((max_replay_buffer_size, observation_dim))
            self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))

        self._actions = np.zeros((max_replay_buffer_size, action_dim))

        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype="uint8")
        self._timeouts = np.zeros((max_replay_buffer_size, 1), dtype="uint8")
        # absorbing[0] is if obs was absorbing, absorbing[1] is if next_obs was absorbing
        self._absorbing = np.zeros((max_replay_buffer_size, 2))
        self._top = 0
        self._size = 0
        self._trajs = 0

        # keeping track of trajectory boundaries
        # assumption is trajectory lengths are AT MOST the length of the entire replay buffer
        self._cur_start = 0
        self._traj_endpoints = {}  # start->end means [start, end)

        self._obs_mean = None
        self._obs_std = None
        self._act_mean = None
        self._act_std = None
        self._is_normalized = False

    def _np_randint(self, *args, **kwargs):
        rets = self._np_rand_state.randint(*args, **kwargs)
        return rets

    def _np_choice(self, *args, **kwargs):
        rets = self._np_rand_state.choice(*args, **kwargs)
        return rets

    def add_sample(
        self,
        observation,
        action,
        reward,
        terminal,
        next_observation,
        timeout=False,
        **kwargs
    ):
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._timeouts[self._top] = timeout
        if "absorbing" in kwargs:
            self._absorbing[self._top] = kwargs["absorbing"]

        if terminal:
            next_start = (self._top + 1) % self._max_replay_buffer_size
            self._traj_endpoints[self._cur_start] = next_start
            self._cur_start = next_start

        if isinstance(self._observations, dict):
            for key, obs in observation.items():
                self._observations[key][self._top] = obs
            for key, obs in next_observation.items():
                self._next_obs[key][self._top] = obs
        else:
            self._observations[self._top] = observation
            self._next_obs[self._top] = next_observation
        self._advance()

    def save_data(self, save_name):
        save_dict = {
            "observations": self._observations[: self._top],
            "actions": self._actions[: self._top],
            "next_observations": self._next_obs[: self._top],
            "terminals": self._terminals[: self._top],
            "timeouts": self._timeouts[: self._top],
            "rewards": self._rewards[: self._top],
            "agent_infos": [None] * len(self._observations[: self._top]),
            "env_infos": [None] * len(self._observations[: self._top]),
        }

        with open(save_name, "wb") as f:
            pickle.dump(save_dict, f)

    def terminate_episode(self):
        if self._cur_start != self._top:
            # if they are equal it means that the previous state was terminal
            # and was handled so there is no need to handle it again
            # THERE WILL BE A BUG HERE IS max_replay_buffer_size
            # IS NOT STRICTLY LARGER THAN MAX EPISODE LENGTH
            self._traj_endpoints[self._cur_start] = self._top
            self._cur_start = self._top

    def add_path(self, path, absorbing=False, env=None):
        if not absorbing:
            for (
                ob,
                action,
                reward,
                next_ob,
                terminal,
                # agent_info,
                # env_info
            ) in zip(
                path["observations"],
                path["actions"],
                path["rewards"],
                path["next_observations"],
                path["terminals"],
                # path["agent_infos"],
                # path["env_infos"],
            ):
                self.add_sample(
                    observation=ob,
                    action=action,
                    reward=reward,
                    terminal=terminal,
                    next_observation=next_ob,
                    # agent_info=agent_info,
                    # env_info=env_info,
                )
        else:
            for (
                ob,
                action,
                reward,
                next_ob,
                terminal,
                # agent_info,
                # env_info
            ) in zip(
                path["observations"],
                path["actions"],
                path["rewards"],
                path["next_observations"],
                path["terminals"],
                # path["agent_infos"],
                # path["env_infos"],
            ):
                self.add_sample(
                    observation=ob,
                    action=action,
                    reward=reward,
                    terminal=np.array([False]),
                    next_observation=next_ob,
                    absorbing=np.array([0.0, 0.0]),
                    # agent_info=agent_info,
                    # env_info=env_info,
                )
                if terminal[0]:
                    print("add terminal")
                    self.add_sample(
                        observation=next_ob,
                        # action=action,
                        action=env.action_space.sample(),
                        reward=reward,
                        terminal=np.array([False]),
                        next_observation=np.zeros_like(next_ob),  # next_ob,
                        absorbing=np.array([0.0, 1.0]),
                        # agent_info=agent_info,
                        # env_info=env_info,
                    )
                    self.add_sample(
                        observation=np.zeros_like(next_ob),  # next_ob,
                        # action=action,
                        action=env.action_space.sample(),
                        reward=reward,
                        terminal=np.array([False]),
                        next_observation=np.zeros_like(next_ob),  # next_ob,
                        absorbing=np.array([1.0, 1.0]),
                        # agent_info=agent_info,
                        # env_info=env_info,
                    )

        self.terminate_episode()
        self._trajs += 1

    def get_traj_num(self):
        return self._trajs

    def get_all(self, keys=None, multi_step=False, step_num=1, **kwargs):
        indices = range(self._size)

        return self._get_batch_using_indices(
            indices, keys=keys, multi_step=multi_step, step_num=step_num, **kwargs
        )

    def _advance(self):
        if self._top in self._traj_endpoints:
            # this means that the step in the replay buffer
            # that we just overwrote was the start of a some
            # trajectory, so now the full trajectory is no longer
            # there and we should remove it
            del self._traj_endpoints[self._top]
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(
        self, batch_size, keys=None, multi_step=False, step_num=1, **kwargs
    ):
        indices = self._np_randint(0, self._size, batch_size)
        if multi_step:
            indices = self._np_randint(0, self._size - step_num, batch_size)
            # candidates = list(range(0, self._size))
            # for value in list(self._traj_endpoints.values()):
            #     for step in np.arange(0, step_num):
            #         candidates.remove((value-step-1)%self._size) # endpoints are like [:endpoints] that may do not live in candidates
            # indices = np.random.choice(candidates, batch_size)

        return self._get_batch_using_indices(
            indices, keys=keys, multi_step=multi_step, step_num=step_num, **kwargs
        )

    def _get_batch_using_indices(
        self, indices, keys=None, multi_step=False, step_num=1
    ):
        if keys is None:
            keys = set(
                [
                    "observations",
                    "actions",
                    "rewards",
                    "terminals",
                    "next_observations",
                    "absorbing",
                ]
            )
        if isinstance(self._observations, dict):
            obs_to_return = {}
            next_obs_to_return = {}
            for k in self._observations:
                if "observations" in keys:
                    obs_to_return[k] = self._observations[k][indices]
                if "next_observations" in keys:
                    next_obs_to_return[k] = self._next_obs[k][indices]
        else:
            obs_to_return = self._observations[indices]
            next_obs_to_return = self._next_obs[indices]

        ret_dict = {}
        if "observations" in keys:
            ret_dict["observations"] = obs_to_return
        if "actions" in keys:
            ret_dict["actions"] = self._actions[indices]
        if "rewards" in keys:
            ret_dict["rewards"] = self._rewards[indices]
        if "terminals" in keys:
            ret_dict["terminals"] = self._terminals[indices]
        if "next_observations" in keys:
            ret_dict["next_observations"] = next_obs_to_return
        if "absorbing" in keys:
            ret_dict["absorbing"] = self._absorbing[indices]

        if multi_step:
            next_next_obs_return = [None] * step_num
            for i in np.arange(1, step_num + 1):
                if isinstance(self._observations, dict):
                    next_next_obs_return[i - 1] = {}
                    for k in self._observations:
                        next_next_obs_return[i - 1][k] = self._next_obs[k][
                            (indices + i) % self._max_replay_buffer_size
                        ]
                else:
                    next_next_obs_return[i - 1] = self._next_obs[
                        (indices + i) % self._max_replay_buffer_size
                    ]

                for j, indice in enumerate(indices):
                    source_list = list(range(indice + 1, indice + i + 1))
                    target_list = list(self._traj_endpoints.values())
                    res = set(source_list) & set(target_list)
                    if (
                        len(res) > 0
                    ):  # there is a number in range(indice, indice+i+1) are a traj endpoint, this should be the last state of the traj
                        next_next_obs_return[i - 1][j] = self._next_obs[
                            list(res)[0] - 1
                        ]

                ret_dict["next{}_observations".format(i)] = next_next_obs_return[i - 1]

        # print(step_num, ret_dict.keys())
        return ret_dict

    def _get_segment(self, start, end, keys=None):
        if start < end or end == 0:
            if end == 0:
                end = self._max_replay_buffer_size
            return self._get_batch_using_indices(range(start, end), keys=keys)

        inds = list(range(start, self._max_replay_buffer_size)) + list(range(0, end))
        return self._get_batch_using_indices(inds, keys=keys)

    def _get_samples_from_traj(self, start, end, samples_per_traj, keys=None):
        # subsample a trajectory
        if start < end or end == 0:
            if end == 0:
                end = self._max_replay_buffer_size
            inds = range(start, end)
        else:
            inds = list(range(start, self._max_replay_buffer_size)) + list(
                range(0, end)
            )
        inds = self._np_choice(
            inds, size=samples_per_traj, replace=len(inds) < samples_per_traj
        )
        return self._get_batch_using_indices(inds, keys=keys)

    def sample_trajs(self, num_trajs, keys=None, samples_per_traj=None):
        # samples_per_traj of None mean use all of the samples
        keys_list = list(self._traj_endpoints.keys())
        starts = self._np_choice(
            keys_list, size=num_trajs, replace=len(keys_list) < num_trajs
        )
        ends = map(lambda k: self._traj_endpoints[k], starts)

        if samples_per_traj is None:
            return list(
                starmap(lambda s, e: self._get_segment(s, e, keys), zip(starts, ends))
            )
        else:
            return list(
                starmap(
                    lambda s, e: self._get_samples_from_traj(
                        s, e, samples_per_traj, keys
                    ),
                    zip(starts, ends),
                )
            )

    def num_steps_can_sample(self):
        return self._size

    def sample_all_trajs(
        self,
        keys=None,
        samples_per_traj=None,
    ):
        # samples_per_traj of None mean use all of the samples
        starts = list(self._traj_endpoints.keys())
        ends = map(lambda k: self._traj_endpoints[k], starts)

        if samples_per_traj is None:
            return list(
                starmap(lambda s, e: self._get_segment(s, e, keys), zip(starts, ends))
            )
        else:
            return list(
                starmap(
                    lambda s, e: self._get_samples_from_traj(
                        s, e, samples_per_traj, keys
                    ),
                    zip(starts, ends),
                )
            )

    def clear(self):
        if isinstance(self._observation_dim, tuple):
            dims = [d for d in self._observation_dim]
            dims = [self._max_replay_buffer_size] + dims
            dims = tuple(dims)
            self._observations = np.zeros(dims)
            self._next_obs = np.zeros(dims)
        elif isinstance(self._observation_dim, dict):
            # assuming that this is a one-level dictionary
            self._observations = {}
            self._next_obs = {}

            for key, dims in self._observation_dim.items():
                if isinstance(dims, tuple):
                    dims = tuple([self._max_replay_buffer_size] + list(dims))
                else:
                    dims = (self._max_replay_buffer_size, dims)
                self._observations[key] = np.zeros(dims)
                self._next_obs[key] = np.zeros(dims)
        else:
            # else observation_dim is an integer
            self._observations = np.zeros(
                (self._max_replay_buffer_size, self._observation_dim)
            )
            self._next_obs = np.zeros(
                (self._max_replay_buffer_size, self._observation_dim)
            )

        self._actions = np.zeros((self._max_replay_buffer_size, self._action_dim))

        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((self._max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((self._max_replay_buffer_size, 1), dtype="uint8")
        self._timeouts = np.zeros((self._max_replay_buffer_size, 1), dtype="uint8")
        # absorbing[0] is if obs was absorbing, absorbing[1] is if next_obs was absorbing
        self._absorbing = np.zeros((self._max_replay_buffer_size, 2))
        self._top = 0
        self._size = 0
        self._trajs = 0

        # keeping track of trajectory boundaries
        # assumption is trajectory lengths are AT MOST the length of the entire replay buffer
        self._cur_start = 0
        self._traj_endpoints = {}  # start->end means [start, end)

        self._is_normalized = False

    def calc_statistics(self, return_obs: bool = True, return_act: bool = True):
        if self._obs_mean is None:
            assert self._obs_std is None
            self._obs_mean = np.mean(self._observations[: self._size], axis=0)
            self._obs_std = np.std(self._observations[: self._size], axis=0)
        if self._act_mean is None:
            assert self._act_std is None
            self._act_mean = np.mean(self._actions[: self._size], axis=0)
            self._act_std = np.std(self._actions[: self._size], axis=0)
        return {
            "size": self._size,
            "max_size": self._max_replay_buffer_size,
            "trajs": self.get_traj_num(),
            "obs_mean": self._obs_mean if return_obs else None,
            "obs_std": self._obs_std if return_obs else None,
            "act_mean": self._act_mean if return_act else None,
            "act_std": self._act_std if return_act else None,
        }
    
    def normalize(self, normalize_obs: bool = True, normalize_act: bool = False):
        if self._is_normalized:
            return
        if normalize_obs:
            if self._obs_mean is None:
                self.calc_statistics()
            self._observations = (self._observations - self._obs_mean) / (
                self._obs_std + 1e-6
            )
            self._next_obs = (self._next_obs - self._obs_mean) / (
                self._obs_std + 1e-6
            )
        if normalize_act:
            if self._act_mean is None:
                self.calc_statistics()
            self._actions = (self._actions - self._act_mean) / (self._act_std + 1e-6)
        self._is_normalized = True


class EnvReplayBuffer(ReplayBuffer):
    def __init__(self, max_replay_buffer_size, env, random_seed=1995):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            random_seed=random_seed,
        )

    def add_sample(
        self, observation, action, reward, terminal, next_observation, **kwargs
    ):
        # if isinstance(self._action_space, Discrete):
        #     new_action = np.zeros(self._action_dim)
        #     new_action[action] = 1
        # else:
        #     new_action = action
        super(EnvReplayBuffer, self).add_sample(
            observation, action, reward, terminal, next_observation, **kwargs
        )

    # def get_snapshot(self):
    #     return {
    #         "observations": self._observations,
    #         "actions": self._actions,
    #         "rewards": self._rewards,
    #         "terminals": self._terminals,
    #         "next_observations": self._next_obs,
    #         "timeouts": self._timeouts,
    #         "absorbing": self._absorbing,
    #         "top": self._top,
    #         "size": self._size,
    #         "trajs": self._trajs,
    #         "cur_start": self._cur_start,
    #         "traj_endpoints": self._traj_endpoints,
    #         "obs_mean": self._obs_mean,
    #         "obs_std": self._obs_std,
    #         "act_mean": self._act_mean,
    #         "act_std": self._act_std,
    #         "is_normalized": self._is_normalized,
    #         "ob_space": self._ob_space,
    #         "action_space": self._action_space,
    #     }

    # def load_snapshot(self, snapshot):
    #     self._observations = snapshot["observations"]
    #     self._actions = snapshot["actions"]
    #     self._rewards = snapshot["rewards"]
    #     self._terminals = snapshot["terminals"]
    #     self._next_obs = snapshot["next_observations"]
    #     self._timeouts = snapshot["timeouts"]
    #     self._absorbing = snapshot["absorbing"]
    #     self._top = snapshot["top"]
    #     self._size = snapshot["size"]
    #     self._trajs = snapshot["trajs"]
    #     self._cur_start = snapshot["cur_start"]
    #     self._traj_endpoints = snapshot["traj_endpoints"]
    #     self._obs_mean = snapshot["obs_mean"]
    #     self._obs_std = snapshot["obs_std"]
    #     self._act_mean = snapshot["act_mean"]
    #     self._act_std = snapshot["act_std"]
    #     self._is_normalized = snapshot["is_normalized"]
    #     self._ob_space = snapshot["ob_space"]
    #     self._action_space = snapshot["action_space"]

