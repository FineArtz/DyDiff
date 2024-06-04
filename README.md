## Long-Horizon Rollout via Dynamics Diffusion for Offline Reinforcement Learning (Dydiff)

Code to reproduce the experiments in Long-Horizon Rollout via Dynamics Diffusion for Offline Reinforcement Learning.

**Long-Horizon Rollout via Dynamics Diffusion for Offline Reinforcement Learning**<br>
https://arxiv.org/abs/2405.19189

Abstract: *With the great success of diffusion models (DMs) in generating realistic synthetic vision data, many researchers have investigated their potential in decision-making and control. Most of these works utilized DMs to sample directly from the trajectory space, where DMs can be viewed as a combination of dynamics models and policies. In this work, we explore how to decouple DMs' ability as dynamics models in fully offline settings, allowing the learning policy to roll out trajectories. As DMs learn the data distribution from the dataset, their intrinsic policy is actually the behavior policy induced from the dataset, which results in a mismatch between the behavior policy and the learning policy. We propose Dynamics Diffusion, short as Dydiff, which can inject information from the learning policy to DMs iteratively. Dydiff ensures long-horizon rollout accuracy while maintaining policy consistency and can be easily deployed on model-free algorithms. We provide theoretical analysis to show the advantage of DMs on long-horizon rollout over models and demonstrate the effectiveness of Dydiff in the context of offline reinforcement learning, where the rollout dataset is provided but no online environment for interaction.*

## Requirements

* 64-bit Python 3.9 and PyTorch 1.12.1 (or later). See https://pytorch.org for PyTorch install instructions.
* Python libraries: See [requirements.txt](./requirements.txt) for exact library dependencies or create a conda environment using [environment.yml](./environment.yml)

## Usage

### Train diffusion models

You can use the [train.py](./train.py) to train a diffusion model on a dataset from [D4RL](https://github.com/Farama-Foundation/D4RL).
- Train a diffusion model on hopper-medium-expert-v2 dataset:

```shell
python train.py --outdir=training-runs  -m hopper-medium-expert-v2  --max_length_of_trajectory=100  --rl_model_type=UNet_RL --batch=8 --duration=4 --tick=1 --device='cuda:0' --seed=2001 --name=multienv --use_cond=True --include_action=True --cond_on_action True 
```

### Train single-step dynamic models

- Train a single-step dynamic model on hopper-medium-expert-v2 dataset
``` shell
python run_experiment.py -e exp_specs/dyn_model/dyn_model_hopper_me.yaml
```

### Train reward models

- Train a reward model on hopper-medium-expert-v2 dataset
``` shell
python run_experiment.py -e exp_specs/rew_model/rew_model_hopper_me.yaml
```

### Train policy
Before training a policy, please make the pretrained diffusion model, the single-step model and the reward model ready and fill paths of the corresponding pretrained models in the config files.

- Train a TD3BC policy with the softmax filter on hopper-medium-expert-v2 dataset.
```shell
python run_experiment.py -e exp_specs/dydiff_td3bc/softmax/dydiff_td3bc_hopper_me_softmax.yaml
```
- Train a Diffusion Q-Learning policy with the hardmax filter on hopper-medium-expert-v2 dataset.

```shell
python run_experiment.py -e exp_specs/dydiff_diffql/hardmax/dydiff_diffql_hopper_me_hardmax.yaml
```

## Citation

```
@misc{zhao2024diffusionbased,
      title={Long-Horizon Rollout via Dynamics Diffusion for Offline Reinforcement Learning}, 
      author={Hanye Zhao and Xiaoshen Han and Zhengbang Zhu and Minghuan Liu and Yong Yu and Weinan Zhang},
      year={2024},
      eprint={2405.19189},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```