import torch
import torch.nn as nn
import torch.nn.utils as ut
import numpy as np
import os
from model.vae import VAE
from model.controller import Controller
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, TransformObservation

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def current_directory_path(filename):
    return os.path.join(CURRENT_DIRECTORY, filename)


def drive_n_times(n, mean):
    LEARNING_RATE = 0.0001
    PARAMETER_SD = 0.05
    rewards = []
    noises = []
    for sample in range(n):
        noise = torch.randn_like(ut.parameters_to_vector(c.parameters()))
        noises.append(noise)
        sampled_c_parameters = mean + PARAMETER_SD * noise
        ut.vector_to_parameters(sampled_c_parameters, c.parameters())
        reward = start_driving(env, vae, c)
        rewards.append(reward)
        print(f"Drove {sample+1}/{n} times. Reward: {reward}")

    update = calculate_update(rewards, noises)
    mean = mean + (LEARNING_RATE / (PARAMETER_SD * n)) * update
    average_reward = sum(rewards) / len(rewards)
    return mean, average_reward


def start_driving(env, vae, c):
    observation, info = env.reset()
    episode_over = False
    reward_accum = 0
    device = next(c.parameters()).device
    while not episode_over:
        observation = hwc_to_chw(observation).to(device)
        z = vae.encode(observation)
        action = generate_action(z, c)
        observation, reward, terminated, truncated, info = env.step(action)
        reward_accum += reward
        episode_over = terminated or truncated
    return reward_accum


def preview_model(vae, c):
    env = load_gym("human")
    c.load_state_dict(
        torch.load(current_directory_path("c_model.bin"), map_location=device)
    )
    start_driving(env, vae, c)


def load_vae(device) -> VAE:
    vae = VAE().to(device)
    vae.load_state_dict(
        torch.load(current_directory_path("vae_model.bin"), map_location=device)
    )
    return vae


def load_gym(render_mode):
    env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=True)
    env = TransformObservation(env, lambda obs: obs[:80, 0:, 0:], env.observation_space)
    env = ResizeObservation(env, (64, 64))
    return env


def hwc_to_chw(x):
    return torch.as_tensor(x).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def generate_action(z, c):
    steer, accelerate, brake = c.forward(z)
    return np.array(
        [
            steer[0].detach().cpu().item(),
            accelerate[0].detach().cpu().item(),
            brake[0].detach().cpu().item(),
        ],
        dtype=np.float32,
    )


def calculate_update(rewards, noises):
    rewards_mean = np.mean(rewards)
    rewards_std = np.std(rewards) + 1e-8
    length = len(rewards)
    sum = 0
    for i in range(length):
        standardized_reward = (rewards[i] - rewards_mean) / rewards_std
        sum += standardized_reward * noises[i]
    return sum


if __name__ == "__main__":
    EPOCH = 200
    NO_PARAMETERS_SAMPLES = 16

    env = load_gym("rgb_array")
    device = torch.device("cuda")
    vae = load_vae(device)
    c = Controller().to(device)
    if os.path.exists(current_directory_path("c_model.bin")):
        preview_model(vae, c)
        quit()
    mean = ut.parameters_to_vector(c.parameters()).clone()

    print("Started training...")
    max_reward = float("-inf")
    for epoch in range(EPOCH):
        mean, average_reward = drive_n_times(NO_PARAMETERS_SAMPLES, mean)
        if average_reward > max_reward:
            max_reward = average_reward
        ut.vector_to_parameters(mean, c.parameters())
        torch.save(c.state_dict(), current_directory_path("c_model.bin"))
        print("**************************************************************\n")
        print(
            f"EPOCH: {epoch+1}/{EPOCH}. Average Reward: {average_reward}. Maximum Reward: {max_reward}"
        )
        print("\n**************************************************************")
