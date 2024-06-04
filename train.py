# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop
from training.dataset import get_trajectories
import wandb
import random
import warnings

warnings.filterwarnings('ignore',
                        'Grad strides do not match bucket view strides')  # False warning printed by PyTorch 1.12.

transformer_config = {
    'n_layer': 4,
    'n_head': 2,
    'n_emb': 768,
    "p_drop_emb": 0.0,
    'p_drop_attn': 0.002,
    'causal_attn': True,
    'time_as_cond': True,
    'obs_as_cond': False,
    'n_cond_layers': 0,
}

# sampler_kwargs = {
#     "S_churn": 90,
#     "S_max": 13.1,
#     "S_min": 0.026,
#     "S_noise": 1.006,
#     "num_steps": 384,
# }

sampler_kwargs = {
    "S_churn": 60,
    "S_max": 52.24,
    "S_min": 0.3699,
    "S_noise": 1.002,
    "num_steps": 34,
}

# ----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------

@click.command()
# Main options.
@click.option('--outdir', help='Where to save the results', metavar='DIR', type=str, required=True, default="./log")
@click.option('--data', help='Path to the dataset', metavar='ZIP|DIR', type=str)
@click.option('--precond', help='Preconditioning & loss function', metavar='vp|ve|edm',
              type=click.Choice(['vp', 've', 'edm']), default='edm', show_default=True)
# Hyperparameters.
@click.option('--duration', help='Training duration', metavar='MIMG', type=click.FloatRange(min=0, min_open=True),
              default=200, show_default=True)
@click.option('--batch', help='Total batch size', metavar='INT', type=click.IntRange(min=1), default=512,
              show_default=True)
@click.option('--batch-gpu', help='Limit batch size per GPU', metavar='INT', type=click.IntRange(min=1))
@click.option('--cbase', help='Channel multiplier  [default: varies]', metavar='INT', type=int)
@click.option('--cres', help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--lr', help='Learning rate', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=10e-4,
              show_default=True)
@click.option('--ema', help='EMA half-life', metavar='MIMG', type=click.FloatRange(min=0), default=0.5,
              show_default=True)
@click.option('--dropout', help='Dropout probability', metavar='FLOAT', type=click.FloatRange(min=0, max=1),
              default=0.13, show_default=True)
# Performance-related.
@click.option('--fp16', help='Enable mixed-precision training', metavar='BOOL', type=bool, default=False,
              show_default=True)
@click.option('--ls', help='Loss scaling', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=1,
              show_default=True)
@click.option('--bench', help='Enable cuDNN benchmarking', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--cache', help='Cache dataset in CPU memory', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--workers', help='DataLoader worker processes', metavar='INT', type=click.IntRange(min=1), default=1,
              show_default=True)
# I/O-related.
@click.option('--desc', help='String to include in result dir name', metavar='STR', type=str)
@click.option('--nosubdir', help='Do not create a subdirectory for results', is_flag=True)
@click.option('--tick', help='How often to print progress', metavar='KIMG', type=click.IntRange(min=1), default=50,
              show_default=True)
@click.option('--snap', help='How often to save snapshots', metavar='TICKS', type=click.IntRange(min=1), default=50,
              show_default=True)
@click.option('--dump', help='How often to dump state', metavar='TICKS', type=click.IntRange(min=1), default=500,
              show_default=True)
@click.option('--seed', help='Random seed  [default: random]', metavar='INT', type=int, default=0)
@click.option('--transfer', help='Transfer learning from network pickle', metavar='PKL|URL', type=str)
@click.option('--resume', help='Resume from previous training state', metavar='PT', type=str)
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)
# wandb
@click.option('--wandb_dir', help='Where to save the wandb results', metavar='DIR', type=str, default='.')
@click.option('--resume_id', help='Wandb id to resume from', metavar='ID', type=str, default=None)
@click.option('--name', help='Name of the wandb run', metavar='STR', type=str,
              default="action-conditon-Unet-more_evaluate")
# reinforcement learning related 
@click.option('-m', '--mujoco_name', help='The name of the mujoco env', type=str, default='hopper-medium-expert-v0')
@click.option('--max_length_of_trajectory', type=int, default=50)
@click.option('--rl_model_type', help='The model type used in RL settings.', metavar='UNet_RL|MLP_RL',
              type=click.Choice(['UNet_RL', 'MLP_RL', 'Transformer_RL']), default='UNet_RL', show_default=True)
@click.option('--use_cond', help='Use condition in RL settings', default=True, show_default=True)
@click.option('--include_action', help='Include action in RL settings', default=True, show_default=True)
@click.option('--cond_on_action', help='Condition on action in RL settings', default=False, show_default=True)

# device
@click.option('--device', help='The device to train on', type=str, default='cuda:0')
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    # check for rl
    assert opts.mujoco_name is not None, "must contain a mujoco name in rl settings"

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.rl_setting = True
    c.use_cond = opts.use_cond
    c.include_action = opts.include_action
    c.max_length_of_trajectory = opts.max_length_of_trajectory
    c.device = opts.device
    c.sampler_kwargs = sampler_kwargs
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.MujocoDataset', name=opts.mujoco_name,
                                               use_labels=0, use_cond=opts.use_cond)

    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9, 0.999], eps=1e-8)

    # Validate dataset options. 
    try:
        env, trajectories, lengths, indexer, datas = get_trajectories(c.dataset_kwargs["name"],
                                                                        max_length_of_trajectory=opts.max_length_of_trajectory,
                                                                        include_action=c.include_action)
        c.dataset_kwargs.observation_dim = env.observation_space.shape[0]
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs,
                                                            trajectories=trajectories,
                                                            lengths=lengths, indexer=indexer,
                                                            dim=trajectories[0].shape[1],
                                                            max_length_of_trajectory=opts.max_length_of_trajectory,
                                                            datas=datas)  # subclass of training.dataset.Dataset
        dataset_name = dataset_obj.name
        c.dataset_kwargs.max_size = len(dataset_obj)  # be explicit about dataset size
        del dataset_obj  # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')
    # Network architecture.
    c.network_kwargs.update(model_type=opts['rl_model_type'], embedding_type='positional')
    print(f"Using {opts['rl_model_type']}...")
    c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1, 1])
    if opts["rl_model_type"] == "Transformer_RL":
        for key, value in transformer_config.items():
            c.network_kwargs[key] = value

    # Preconditioning & loss function.
    if opts.precond == 'vp':
        c.network_kwargs.class_name = 'training.networks.VPPrecond'
        c.loss_kwargs.class_name = 'training.loss.VPLoss'
    elif opts.precond == 've':
        c.network_kwargs.class_name = 'training.networks.VEPrecond'
        c.loss_kwargs.class_name = 'training.loss.VELoss'
    elif opts.precond == 'edm':
        assert opts.precond == 'edm'
        c.network_kwargs.use_cond = opts.use_cond
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.EDMLoss'
        c.loss_kwargs.use_cond = opts.use_cond
        c.loss_kwargs.observation_dim = c.dataset_kwargs.observation_dim
        c.loss_kwargs.rl_setting = True
        c.loss_kwargs.cond_on_action = opts.cond_on_action
    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=c.device)
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}-cond_on_aciton-{opts.cond_on_action}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}-{random.randint(10000000, 99999999)}')
        assert not os.path.exists(c.run_dir)

    # Initialize wandb
    if dist.get_rank() == 0:
        # wandb.init(
        #     entity=os.environ['WANDB_ENTITY'], project=os.environ['WANDB_PROJECT'], config=c,
        #     dir=opts.wandb_dir, id=opts.resume_id, resume=opts.resume_id is not None,
        # )
        wandb.init(
            name=opts.name if "name" in opts else None,
            project="dydiff", config=c,
            dir=opts.wandb_dir, id=opts.resume_id, resume=opts.resume_id is not None,
        )
    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(f'Training on {opts.device}...')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Mujoco Dataset Name:            {c.dataset_kwargs.name}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    net = training_loop.training_loop(**c)
    return net


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------
