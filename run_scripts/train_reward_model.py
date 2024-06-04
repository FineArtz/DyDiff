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
from rlkit.logger.setup import setup_logger, set_seed
from rlkit.rew_model_trainer import RewModelTrainer
from rlkit.utils.buffer import EnvReplayBuffer
from rlkit.utils.process import process_d4rl_dataset
from training.networks import FlattenMLP
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

    # model params
    model_params = variant["model_params"]
    net_size = model_params["rew_net_size"]
    num_hidden = model_params["rew_num_hidden_layers"]
    rew_model = FlattenMLP(
        input_dim=obs_dim + action_dim,
        output_dim=1,
        hidden_dims=num_hidden * [net_size],
    )
    save_path = model_params.get("save_path", "./logs/reward_model")
    save_path = os.path.join(save_path, variant['dataset_name'])
    directories = save_path.split('/')
    current_path = ''
    for directory in directories:
        current_path = os.path.join(current_path, directory)
        if not os.path.exists(current_path):
            os.mkdir(current_path)
    model_params["save_path"] = save_path
    model_name = f"rew_model"
    if normalize_obs:
        model_name += "_scaled"
    model_name += f"_seed{env_specs['eval_env_seed']}"
    rew_model_trainer = RewModelTrainer(
        model=rew_model,
        replay_buffer=replay_buffer,
        debug=debug,
        save_name=model_name,
        normalize_obs=normalize_obs,
        **model_params,
    )

    if ptu.gpu_enabled():
        rew_model_trainer.to(ptu.device)

    rew_model_trainer.train()
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
    exp_specs["env_specs"]["eval_env_seed"] = exp_specs["env_specs"][
        "training_env_seed"
    ] = exp_specs["seed"]

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
