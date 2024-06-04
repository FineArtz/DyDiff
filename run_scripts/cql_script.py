import yaml
import argparse
import os, sys, inspect
import gym
import pickle
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

from rlkit.envs.envs import get_env
from rlkit.envs.wrappers import NormalizedBoxEnv, ProxyEnv
from rlkit.logger.logger import load_from_file
from rlkit.logger.setup import setup_logger, set_seed
from rlkit.utils.buffer import EnvReplayBuffer
from rlkit.utils.process import process_d4rl_dataset
from training.networks import FlattenMLP
from rlkit.policy import TanhGaussianPolicy
from rlkit.algo.cql import CQL
from rlkit.base_trainer import BaseOfflineTrainer
import torch_utils.pytorch_utils as ptu


def experiment(variant):
    env_specs = variant["env_specs"]
    env = get_env(env_specs)
    env.seed(env_specs["eval_env_seed"])

    print("\n\nEnv: {}".format(env_specs["env_name"]))
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(env.observation_space))
    print("Act Space: {}\n\n".format(env.action_space))

    obs_space = env.observation_space
    act_space = env.action_space
    assert not isinstance(obs_space, gym.spaces.Dict)
    assert len(obs_space.shape) == 1
    assert len(act_space.shape) == 1

    env_wrapper = ProxyEnv  # Identical wrapper
    wrapper_kwargs = {}

    if isinstance(act_space, gym.spaces.Box):
        env_wrapper = NormalizedBoxEnv

    env = env_wrapper(env, **wrapper_kwargs)

    obs_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]
    
    debug: bool = variant.get("debug", False)
    if debug:
        print("===== DEBUG MODE =====\n")
    
    # Buffer
    with open('demos_listing.yaml', 'r') as f:
        listings = yaml.load(f.read(), Loader=yaml.FullLoader)
    demos_path = listings[variant['dataset_name']]['file_paths'][0]
    if not os.path.exists(demos_path):
        # get task name, e.g. 'hopper-medium-v0'
        task_name = demos_path.split('/')[-1][:-4]
        print(f'Downloading and processing {task_name}...')
        process_d4rl_dataset(task_name, demos_path)
    print("demos_path", demos_path)
    with open(demos_path, 'rb') as f:
        traj_list, dataset = pickle.load(f)
    print(f"dataset size: {len(dataset['observations'])}")
    replay_buffer = EnvReplayBuffer(
        max_replay_buffer_size=len(dataset['observations']),
        env=env,
        random_seed=env_specs['eval_env_seed']
    )
    observations = dataset["observations"]
    next_observations = dataset["next_observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    dones = dataset["terminals"]
    for obs, action, reward, done, next_obs in zip(
        observations, actions, rewards, dones, next_observations
    ):
        replay_buffer.add_sample(obs, action, reward, done, next_obs)
    print("---- finish loading dataset ----\n")
    
    if normalize_obs := env_specs.get("normalize_obs", False):
        obs_all = replay_buffer.get_all()['observations']
        env.estimate_obs_stats(obs_all)
        wrapper_kwargs = {
            "obs_mean": env._obs_mean,
            "obs_std": env._obs_std,
        }
        replay_buffer.normalize(normalize_obs=True, normalize_act=False)
        assert np.all(env._obs_mean == replay_buffer._obs_mean)
        assert np.all(env._obs_std == replay_buffer._obs_std)

    # CQL params
    cql_params = variant["cql_params"]
    cql_net_size = cql_params["net_size"]
    cql_num_hidden = cql_params["num_hidden_layers"]
    qf1 = FlattenMLP(
        input_dim=obs_dim + action_dim,
        output_dim=1,
        hidden_dims=cql_num_hidden * [cql_net_size],
    )
    qf2 = FlattenMLP(
        input_dim=obs_dim + action_dim,
        output_dim=1,
        hidden_dims=cql_num_hidden * [cql_net_size],
    )
    policy = TanhGaussianPolicy(
        state_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=cql_num_hidden * [cql_net_size],
    )
    agent_trainer = CQL(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        env=env,
        debug=debug,
        **cql_params,
    )

    # trainer params
    trainer_params = variant["rl_trainer_params"]
        
    algorithm = BaseOfflineTrainer(
        env=env,
        replay_buffer=replay_buffer,
        algo=agent_trainer,
        exploration_policy=policy,
        debug=debug,
        **trainer_params,
    )

    epoch = 0
    if "load_params" in variant:
        algorithm, epoch = load_from_file(algorithm, **variant["load_params"])

    if ptu.gpu_enabled():
        algorithm.to(ptu.device)

    print("Start from epoch", epoch)
    algorithm.train(start_epoch=epoch)

    return 1

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()
    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.safe_load(spec_string)

    # make all seeds the same.
    exp_specs["env_specs"]["eval_env_seed"] = exp_specs["seed"]

    if exp_specs["using_gpus"] > 0 and args.gpu >= 0:
        print("\n\nUSING GPU\n\n")
        ptu.set_gpu_mode(True, args.gpu)
    else:
        print("\n\nUSING CPU\n\n")
        ptu.set_gpu_mode(False)
    exp_id = exp_specs["exp_id"]
    exp_prefix = exp_specs["exp_name"]
    seed = exp_specs["seed"]
    set_seed(seed)

    log_dir = None
    if "load_params" in exp_specs:
        load_path = exp_specs["load_params"]["load_path"]
        if (load_path is not None) and (len(load_path) > 0):
            log_dir = load_path

    setup_logger(
        exp_prefix=exp_prefix,
        exp_id=exp_id,
        variant=exp_specs,
        seed=seed,
        log_dir=log_dir,
    )

    experiment(exp_specs)
