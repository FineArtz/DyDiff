# Modified from generate.py

import numpy as np
import torch
import cv2
from torch_utils import mujoco_utils
import torch_utils.pytorch_utils as ptu
from rlkit.dynamics_model_trainer import DynamicsTrainer

TORCH_EPS = 1e-4


def mujoco_get_frame(env, state: np.ndarray, width=640, height=480):
    state = np.concatenate([np.array([0]), state])
    env.reset()
    qpos_shape = env.sim.data.qpos.shape[0]
    qvel_shape = env.sim.data.qvel.shape[0]
    qpos = state[:qpos_shape]
    qvel = state[qpos_shape:qpos_shape + qvel_shape]
    env.sim.data.qpos[:] = qpos
    env.sim.data.qvel[:] = qvel
    env.sim.forward()
    frame = env.render(mode='rgb_array', width=width, height=height)
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def mujoco_get_videos(env, states: np.ndarray, width=640, height=480):
    videos = np.zeros((states.shape[0], states.shape[1], height, width, 3))
    for b in range(videos.shape[0]):
        for idx in range(states.shape[1]):
            frame = mujoco_get_frame(env, states[b][idx])
            videos[b, idx] = frame
    return videos


def condition_guidance_sampler(
        net, latents, class_labels=None, randn_like=torch.randn_like, conditions=None,
        observation_dim=None, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, cond_actions=None, **kwargs
):
    if observation_dim is None:
        observation_dim = latents.shape[-1]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    if conditions is not None:
        mujoco_utils.apply_conditioning(x_next, conditions, observation_dim)
    if cond_actions is not None:
        mujoco_utils.apply_action_condition(x_next, cond_actions, observation_dim)
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        
        denoised = net(x_hat, t_hat.repeat(x_hat.shape[0]), class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        if conditions is not None:
            mujoco_utils.apply_conditioning(x_next, conditions, observation_dim)
        if cond_actions is not None:
            mujoco_utils.apply_action_condition(x_next, cond_actions, observation_dim)
            
        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next.repeat(x_next.shape[0]), class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        if conditions is not None:
            mujoco_utils.apply_conditioning(x_next, conditions, observation_dim)
        if cond_actions is not None:
            mujoco_utils.apply_action_condition(x_next, cond_actions, observation_dim)
    return x_next


@torch.inference_mode()
def dynamics_recursive_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dynamics_model: DynamicsTrainer=None, observation_dim=None,
    conditions=None, policy=None, noised=False, iterative_times=5, **kwargs
):
    # get the init traj with dynamic and env auto-regressively
    state = conditions[0]
    state = state.to(ptu.device)
    trajs = torch.empty_like(latents)
    trajs[:, 0, :observation_dim] = state
    for i in range(latents.shape[1] - 1):
        action, mean, log_std, log_prob = policy(state, deterministic=True)
        trajs[:, i, observation_dim:] = action
        state = dynamics_model(state, action, deterministic=True)
        trajs[:, i + 1, :observation_dim] = state
    action, _, _, _ = policy(state, deterministic=True)
    trajs[:, -1, observation_dim:] = action
    cond_actions = trajs[:, :, observation_dim:]
    
    for i in range(iterative_times):
        new_trajs = condition_guidance_sampler(
            net, latents, class_labels, randn_like, conditions, observation_dim, num_steps, sigma_min, sigma_max, rho,
            S_churn, S_min, S_max, S_noise, cond_actions=cond_actions, policy=policy, noised=noised, **kwargs
        )
        assert (torch.abs(new_trajs[:, 0, :observation_dim] - trajs[:, 0, :observation_dim]) < TORCH_EPS).all()
        trajs[:, :, :observation_dim] = (new_trajs[:, :, :observation_dim] + trajs[:, :, :observation_dim]) / 2
        cond_actions, mean, log_std, log_prob = policy(trajs[:, :, :observation_dim], deterministic=True)
        trajs[:, :, observation_dim:] = cond_actions
        latents = randn_like(latents)
    return trajs