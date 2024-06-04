## Diffusion-based Dynamics Models for Long-Horizon Rollout in Offline Reinforcement Learning (Dydiff)

Code to reproduce the experiments in Diffusion-based Dynamics Models for Long-Horizon Rollout in Offline Reinforcement Learning.

**Diffusion-based Dynamics Models for Long-Horizon Rollout in Offline Reinforcement Learning**<br>
<!-- Tero Karras, Miika Aittala, Timo Aila, Samuli Laine -->
<!-- <br>https://arxiv.org/abs/2206.00364<br> -->

Abstract: *With the great success of diffusion models (DMs) in generating realistic synthetic vision data, many researchers have investigated their potential in decision-making and control. Most of these works utilized DMs to sample directly from the trajectory space, where DMs can be viewed as a combination of dynamics models and policies. In this work, we explore how to decouple DMs' ability as dynamics models in fully offline settings, allowing the learning policy to roll out trajectories. As DMs learn the data distribution from the dataset, their intrinsic policy is actually the behavior policy induced from the dataset, which results in a mismatch between the behavior policy and the learning policy. We propose Dynamics Diffusion, short as Dydiff, which can inject information from the learning policy to DMs iteratively. Dydiff ensures long-horizon rollout accuracy while maintaining policy consistency and can be easily deployed on model-free algorithms. We provide theoretical analysis to show the advantage of DMs on long-horizon rollout over models and demonstrate the effectiveness of Dydiff in the context of offline reinforcement learning, where the rollout dataset is provided but no online environment for interaction.*

<!-- For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/) -->

## Requirements

* 64-bit Python 3.9 and PyTorch 1.12.1 (or later). See https://pytorch.org for PyTorch install instructions.
* Python libraries: See [requirements.txt](./requirements.txt) for exact library dependencies or create a conda environment using [environment.yml](./environment.yml)

## Usage

### Train diffusion models

You can use the [train.py](./train.py) to train a diffusion model on a dataset from [D4RL](https://github.com/Farama-Foundation/D4RL).
- Train a diffusion model on the dataset `hopper-medium-expert-v2`:

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
Before training a policy, please make the pretrained diffusion model, the single-step model and the reward model ready and fill paths of the corresponding pretrained models into the config files.

- Train a td3bc policy with the softmax filter on hopper-medium-expert-v2 dataset.
```shell
python run_experiment.py -e exp_specs/dydiff_td3bc/softmax/dydiff_td3bc_hopper_me_softmax.yaml
```
- Train a diffusion q-learning policy with the hardmax filter on hopper-medium-expert-v2 dataset.

```shell
python run_experiment.py -e exp_specs/dydiff_diffql/hardmax/dydiff_diffql_hopper_me_hardmax.yaml
```