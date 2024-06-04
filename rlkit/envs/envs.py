import importlib
import gym

try:
    import dmc2gym
except Exception:
    pass

from rlkit.envs.envs_dict import envs_dict
from rlkit.envs.envpool import EnvpoolEnv
from rlkit.envs.wrappers import ProxyEnv
from rlkit.envs.vecenvs import SubprocVectorEnv, DummyVectorEnv

env_overwrite = {}


def load(name):
    # taken from OpenAI gym registration.py
    print(name)
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def get_env(env_specs):
    """
    env_specs:
        env_name: 'halfcheetah'
        env_kwargs: {} # kwargs to pass to the env constructor call
    """
    domain = env_specs["env_name"]

    if domain == "dmc":
        env_class = dmc2gym.make
    elif "maze" in domain:
        env_class = None
    else:
        env_class = load(envs_dict[domain])

    if "antmaze" in domain:
        env = gym.make(domain, **env_specs["env_kwargs"]).unwrapped.wrapped_env
    elif "maze2d" in domain:
        env = gym.make(domain, **env_specs["env_kwargs"]).unwrapped
    else:
        # Equal to gym.make()
        env = env_class(**env_specs["env_kwargs"])

    print(domain, domain in env_overwrite)
    if domain in env_overwrite:
        print(
            "[ environments/utils ] WARNING: Using overwritten {} environment".format(
                domain
            )
        )
        env = env_overwrite[domain]()

    return env


def get_envs(env_specs, env_wrapper=None, wrapper_kwargs={}, **kwargs):
    """
    env_specs:
        env_name: 'halfcheetah'
        env_kwargs: {} # kwargs to pass to the env constructor call
    """
    if "use_envpool" in env_specs and env_specs["use_envpool"]:
        envs = EnvpoolEnv(env_specs)
        print(
            "[ environments/utils ] WARNING: Using envpool {} environment".format(
                env_specs["envpool_name"]
            )
        )
    else:
        domain = env_specs["env_name"]

        if env_wrapper is None:
            env_wrapper = ProxyEnv

        if domain == "dmc":
            env_class = dmc2gym.make
        elif "maze" in domain:
            env_class = None
        else:
            env_class = load(envs_dict[domain])

        if "env_num" not in env_specs.keys() or env_specs["env_num"] <= 1:
            if "antmaze" in domain:
                env = gym.make(domain, **env_specs["env_kwargs"]).unwrapped.wrapped_env
                envs = env_wrapper(env, **wrapper_kwargs)
            elif "maze2d" in domain:
                env = gym.make(domain, **env_specs["env_kwargs"]).unwrapped
                envs = env_wrapper(env, **wrapper_kwargs)
            else:
                envs = env_wrapper(env_class(**env_specs["env_kwargs"]), **wrapper_kwargs)

            if domain in env_overwrite:
                print(
                    "[ environments/utils ] WARNING: Using overwritten {} environment".format(
                        domain
                    )
                )
                envs = env_wrapper(
                    env_overwrite[domain](**env_specs["env_kwargs"]), **wrapper_kwargs
                )

            print("\n WARNING: Single environment detected, wrap to DummyVectorEnv.")
            envs = DummyVectorEnv([lambda: envs], **kwargs)

        else:
            if "antmaze" in domain:
                envs = SubprocVectorEnv(
                    [
                        lambda: env_wrapper(
                            gym.make(domain, **env_specs["env_kwargs"]).unwrapped.wrapped_env,
                            **wrapper_kwargs
                        )
                        for _ in range(env_specs["env_num"])
                    ],
                    **kwargs
                )
            elif "maze2d" in domain:
                envs = SubprocVectorEnv(
                    [
                        lambda: env_wrapper(gym.make(domain, **env_specs["env_kwargs"]).unwrapped, **wrapper_kwargs)
                        for _ in range(env_specs["env_num"])
                    ],
                    **kwargs
                )
            else:
                envs = SubprocVectorEnv(
                    [
                        lambda: env_wrapper(
                            env_class(**env_specs["env_kwargs"]), **wrapper_kwargs
                        )
                        for _ in range(env_specs["env_num"])
                    ],
                    **kwargs
                )

            if domain in env_overwrite:
                envs = SubprocVectorEnv(
                    [
                        lambda: env_wrapper(env_overwrite[domain](), **wrapper_kwargs)
                        for _ in range(env_specs["env_num"])
                    ],
                    **kwargs
                )

    return envs