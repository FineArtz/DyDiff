"""
Common evaluation utilities.
"""

from collections import OrderedDict
from numbers import Number
import numpy as np
from gym.spaces import Box, Discrete, Tuple, Dict


def get_generic_path_information(paths, stat_prefix=""):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    statistics = OrderedDict()
    returns = [sum(path["rewards"]) for path in paths]
    # rewards = np.vstack([path["rewards"] for path in paths])
    rewards = np.concatenate([path["rewards"] for path in paths])
    statistics.update(
        create_stats_ordered_dict(
            "Rewards", rewards, stat_prefix=stat_prefix, always_show_all_stats=True
        )
    )
    statistics.update(
        create_stats_ordered_dict(
            "Returns", returns, stat_prefix=stat_prefix, always_show_all_stats=True
        )
    )
    if "is_success" in paths[0]["env_infos"][0].keys():
        acc_sum = [
            (np.sum([x["is_success"] for x in path["env_infos"]]) > 0).astype(float)
            for path in paths
        ]
        acc = np.sum(acc_sum) * 1.0 / len(paths)
        statistics.update(
            create_stats_ordered_dict(
                "Success Num",
                np.sum(acc_sum),
                stat_prefix=stat_prefix,
                always_show_all_stats=True,
            )
        )
        statistics.update(
            create_stats_ordered_dict(
                "Traj Num",
                len(paths),
                stat_prefix=stat_prefix,
                always_show_all_stats=True,
            )
        )
        statistics.update(
            create_stats_ordered_dict(
                "Success Rate", acc, stat_prefix=stat_prefix, always_show_all_stats=True
            )
        )
    actions = [path["actions"] for path in paths]
    statistics.update(
        create_stats_ordered_dict(
            "Actions", actions, stat_prefix=stat_prefix, always_show_all_stats=True
        )
    )
    statistics.update(
        create_stats_ordered_dict(
            "Ep. Len.",
            np.array([len(path["terminals"]) for path in paths]),
            stat_prefix=stat_prefix,
            always_show_all_stats=True,
        )
    )
    statistics["Num Paths"] = len(paths)

    return statistics


def get_average_returns(paths, std=False):
    returns = [sum(path["rewards"]) for path in paths]
    if std:
        return np.mean(returns), np.std(returns)

    return np.mean(returns)


def create_stats_ordered_dict(
    name,
    data,
    stat_prefix=None,
    always_show_all_stats=False,
    exclude_max_min=False,
):
    # print('\n<<<< STAT FOR {} {} >>>>'.format(stat_prefix, name))
    if stat_prefix is not None:
        name = "{} {}".format(stat_prefix, name)
    if isinstance(data, Number):
        # print('was a Number')
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        # print('was a tuple')
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        # print('was a list')
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if isinstance(data, np.ndarray) and data.size == 1 and not always_show_all_stats:
        # print('was a numpy array of data.size==1')
        return OrderedDict({name: float(data)})

    # print('was a numpy array NOT of data.size==1')
    stats = OrderedDict(
        [
            (name + " Mean", np.mean(data)),
            (name + " Std", np.std(data)),
        ]
    )
    if not exclude_max_min:
        stats[name + " Max"] = np.max(data)
        stats[name + " Min"] = np.min(data)
    return stats


def get_dim(space):
    if isinstance(space, Box):
        if len(space.low.shape) > 1:
            return space.low.shape
        return space.low.size
    elif isinstance(space, Discrete):
        return 1  # space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif isinstance(space, Dict):
        return {k: get_dim(v) for k, v in space.spaces.items()}
    elif hasattr(space, "flat_dim"):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))