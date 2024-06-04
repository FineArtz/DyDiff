# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import wandb
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from training.dataset import get_trajectories
from generate_rl import mujoco_get_videos, condition_guidance_sampler
import einops
from rlkit.envs.wrappers import get_set_state


def getEnvNextState(trajectories, env, observation_dim, max_batch=16) -> torch.Tensor:
    batch_size = min(trajectories.shape[0], max_batch)
    trajectories = trajectories[:batch_size]
    if isinstance(trajectories, torch.Tensor):
        trajectories = trajectories.cpu().numpy()
    truth_states = np.zeros([batch_size, trajectories.shape[1], observation_dim])
    rewards = np.zeros([batch_size, trajectories.shape[1]])
    states = trajectories[:, :, :observation_dim]
    actions = trajectories[:, :, observation_dim:]
    for i, trajectory in enumerate(trajectories):
        env.reset()
        for j in range(trajectory.shape[0]):
            env.set_state(states[i][j])
            next_state, reward, done, info = env.step(actions[i][j])
            truth_states[i][j] = next_state
            rewards[i][j] = reward
    rewards = rewards.sum(axis=1)
    results = {'states': truth_states, 'rewards': rewards, "rewards_min": rewards.min(), "rewards_max": rewards.max(),
               "rewards_mean": rewards.mean()}
    return results


def evaluate(net: torch.nn.Module, dataset_iterator, device, env=None, use_cond=False, observation_dim=None,
             evalute_batch_size=16, cond_on_action=False, **sampler_kwargs):
    # assert use_cond, "use_cond should be True now"
    assert observation_dim is not None, "observation_dim should be None now. And must contain actions"
    net.eval()
    if observation_dim is None:
        observation_dim = net.dim
    with torch.inference_mode():
        if use_cond:
            (trajectories, cond), labels = next(dataset_iterator)
            cond[0] = cond[0][:evalute_batch_size]
            sampler_kwargs["conditions"] = cond
            sampler_kwargs["observation_dim"] = observation_dim
        else:
            trajectories, _ = next(dataset_iterator)
        if cond_on_action:
            sampler_kwargs["cond_actions"] = trajectories[:evalute_batch_size, :, observation_dim:].detach()
       
        trajectories = trajectories[:evalute_batch_size].to(device).to(torch.float32)
        noises = torch.rand_like(trajectories).to(device)
        sampler = condition_guidance_sampler
        generated_trajectories: torch.Tensor = sampler(net, noises, **sampler_kwargs)
        log = {}
        if generated_trajectories.shape[-1] > observation_dim: 
            # mse
            predicted_observaitons = net.unnormalize(generated_trajectories.cpu().numpy())[:, 1:, :observation_dim]
            results = getEnvNextState(net.unnormalize(generated_trajectories.cpu().numpy()), env, observation_dim)
            states = results['states'][:, :-1, :]
            mse = ((predicted_observaitons - states) ** 2).mean()
            one_step_mse = ((predicted_observaitons[:, :1, :] - states[:, :1, :]) ** 2).mean()

            log = {'loss/mse': mse, 'loss/one_step_mse': one_step_mse,
            "evaluates/rewards_min": results['rewards_min'], "evaluates/rewards_max": results['rewards_max'],
            "evaluates/rewards_mean": results['rewards_mean']}
    if env is not None:
        generated_trajectories = generated_trajectories.cpu().numpy()
        trajectories = trajectories.cpu().numpy()
        generated_trajectories: np.ndarray = net.unnormalize(generated_trajectories)
        trajectories = net.unnormalize(trajectories)
        videos = mujoco_get_videos(env, generated_trajectories).astype(np.uint8)
        origin_videos = mujoco_get_videos(env, trajectories).astype(np.uint8)
    net.train()
    video = einops.rearrange(videos[0], 't h w c -> t c h w')
    origin_video = einops.rearrange(origin_videos[0], 't h w c -> t c h w')
    generate_video = einops.rearrange(videos[-1], 't h w c -> t c h w')
    video_log = {
            'Video/video': wandb.Video(video), 'Video/origin_video': wandb.Video(origin_video),
            'Video/generate_video': wandb.Video(generate_video),
    }
    for k, v in video_log.items():
        log[k] = v
    return log


def training_loop(
        run_dir='.',  # Output directory.
        dataset_kwargs={},  # Options for training set.
        data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
        network_kwargs={},  # Options for model and preconditioning.
        loss_kwargs={},  # Options for loss function.
        optimizer_kwargs={},  # Options for optimizer.
        augment_kwargs=None,  # Options for augmentation pipeline, None = disable.
        seed=0,  # Global random seed.
        batch_size=512,  # Total batch size for one training iteration.
        batch_gpu=None,  # Limit batch size per GPU, None = no limit.
        total_kimg=200000,  # Training duration, measured in thousands of training images.
        ema_halflife_kimg=500,  # Half-life of the exponential moving average (EMA) of model weights.
        ema_rampup_ratio=0.05,  # EMA ramp-up coefficient, None = no rampup.
        lr_rampup_kimg=10000,  # Learning rate ramp-up duration.
        loss_scaling=1,  # Loss scaling factor for reducing FP16 under/overflows.
        kimg_per_tick=50,  # Interval of progress prints.
        snapshot_ticks=50,  # How often to save network snapshots, None = disable.
        state_dump_ticks=500,  # How often to dump training state, None = disable.
        resume_pkl=None,  # Start from the given network snapshot, None = random initialization.
        resume_state_dump=None,  # Start from the given training state, None = reset training state.
        resume_kimg=0,  # Start from the given training progress.
        cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
        device=torch.device('cuda'),
        rl_setting=False,  # RL setting
        max_length_of_trajectory=50,
        use_cond=False,  # Use conditioning in RL setting
        include_action=False,  # Include action in RL setting
        sampler_kwargs={},  # kwargs used when sampling
        start_to_save_snapshot=1000, #start to save snapshot
        **_kwargs):  # Ignore unrecognized keyword args.
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    if rl_setting:
        env, trajectories, lengths, indexer, datas = get_trajectories(dataset_kwargs["name"],
                                                                      max_length_of_trajectory=max_length_of_trajectory,
                                                                      include_action=include_action)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs, trajectories=trajectories, lengths=lengths,
                                                          indexer=indexer, dim=trajectories[0].shape[1],
                                                          max_length_of_trajectory=max_length_of_trajectory,
                                                          datas=datas)  # subclass of training.dataset.Dataset
        dist.print0(f'Dataset mean: {dataset_obj.mean}.       Dataset std: {dataset_obj.std}')
        dist.print0(f'the length of the trajectory: {max_length_of_trajectory}')
        env.set_state = get_set_state(env)
    else:
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(),
                                           num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(
        torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu,
                                    **data_loader_kwargs))
    # Construct network.
    dist.print0('Constructing network...')
    if rl_setting:
        interface_kwargs = dict(data_dim=dataset_obj.dim, label_dim=dataset_obj.label_dim,
                                max_length_of_trajectory=max_length_of_trajectory, rl_setting=rl_setting)
    else:
        interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels,
                                label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)  # subclass of torch.nn.Module
    if rl_setting:
        net.dataset_mean = dataset_obj.mean
        net.dataset_std = dataset_obj.std
        net.normalize = dataset_obj.normalize
        net.unnormalize = dataset_obj.unnormalize
        net.dataset = dataset_obj
        net.use_cond = use_cond
    net.train().requires_grad_(True).to(device)
    if dist.get_rank() == 0:
        wandb.log({'num_parameters': sum(p.numel() for p in net.parameters() if p.requires_grad)})
        with torch.inference_mode():
            if rl_setting:
                images = torch.zeros([batch_gpu, dataset_obj.max_length_of_trajectory, dataset_obj.dim], device=device)
            else:
                images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution],
                                     device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    if rl_setting and use_cond:
        loss_kwargs["observation_dim"] = env.observation_space.shape[0]
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)  # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(),
                                                    **optimizer_kwargs)  # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(
        **augment_kwargs) if augment_kwargs is not None else None  # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()  # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()  # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data  # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data  # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None

    while True:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                if rl_setting and use_cond:
                    images, cond = images[0].to(device).to(torch.float32), images[1]
                else:
                    images = images.to(device).to(torch.float32)
                if not rl_setting:
                    images = images / 127.5 - 1
                labels = labels.to(device)
                if rl_setting and use_cond:
                    loss = loss_fn(net=ddp, images=images, cond=cond, labels=labels, augment_pipe=augment_pipe)
                else:
                    loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe)
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2 ** 30):<6.2f}"]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2 ** 30):<6.2f}"]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2 ** 30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot and evaluate.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0) and (cur_tick >= start_to_save_snapshot):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value  # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg // 1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data  # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (
                done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()),
                       os.path.join(run_dir, f'training-state-{cur_nimg // 1000:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
                wandb_dict = evaluate(net, dataset_iterator, device, env=env, sigma_max=80, use_cond=use_cond,
                                observation_dim=env.observation_space.shape[0],
                                cond_on_action=loss_kwargs.get("cond_on_action", False),
                                **sampler_kwargs)
            else:
                wandb_dict = {}
            for field, values in training_stats.default_collector.as_dict().items():
                wandb_dict[field] = values["mean"]
                if values["num"] > 1:
                    wandb_dict[field + "_std"] = values["std"]
            wandb.log(wandb_dict, step=cur_nimg)
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(
                json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

    return net

# ----------------------------------------------------------------------------
