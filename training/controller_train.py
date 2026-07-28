import torch
import torch.nn as nn
import torch.nn.utils as ut
import numpy as np
import os
from ..model.vae import VAE
from ..model.controller import Controller
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, TransformObservation


def drive_n_times(n, mean):
    LEARNING_RATE = 0.01
    PARAMETER_SD = 0.5
    rewards = []
    noises = []
    for sample in range(n):
        noise = torch.randn_like(ut.parameters_to_vector(c.parameters()))
        noises.append(noise)
        sampled_c_parameters = mean + PARAMETER_SD * noise
        ut.vector_to_parameters(sampled_c_parameters, c.parameters())
        reward = start_driving(env, vae, c)
        rewards.append(reward)
        print(f"Drove {sample}/{n} times. Reward: {reward}")

    update = calculate_update(rewards, noises)
    mean = mean + LEARNING_RATE / (PARAMETER_SD * n) * update
    average_reward = sum(rewards) / len(rewards)
    return mean, average_reward


def start_driving(env, vae, c):
    observation, info = env.reset()
    episode_over = False
    reward_accum = 0
    while not episode_over:
        hwc_to_chw(observation)
        z = vae.encode(observation)
        action = generate_action(c)
        observation, reward, terminated, truncated, info = env.step(action)
        reward_accum += reward
    return reward_accum


def preview_model(vae, c):
    env = load_gym("human")
    c.load_state_dict(torch.load("c_model.bin", map_location=device))
    start_driving(env, vae, c)


def load_vae(device) -> VAE:
    vae = VAE().to(device)
    vae.load_state_dict(torch.load("vae_model.bin", map_location=device))
    return vae


def load_gym(render_mode):
    env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=True)
    env = TransformObservation(env, lambda obs: obs[:80, 0:, 0:], env.observation_space)
    env = ResizeObservation(env, (64, 64))
    return env


def hwc_to_chw(x):
    return x.permute(2, 0, 1).unsqueeze(0)


def generate_action(c):
    steer, accelerate, brake = c.forward(z)
    return np.array([steer, accelerate, brake])


def calculate_update(rewards, noises):
    length = len(rewards)
    sum = 0
    for i in range(length):
        sum += rewards[i] * noises[i]
    return sum


if __name__ == "__main__":
    EPOCH = 200
    NO_PARAMETERS_SAMPLES = 8

    env = load_gym("rgb_array")
    device = torch.device("cuda")
    vae = load_vae(device)
    c = Controller().to(device)
    if os.path.exists("c_model.bin"):
        preview_model(vae, c)
        quit()
    mean = torch.randn_like(parameters_to_vector(c.parameters()))

    print("Started training...")
    max_reward = float("-inf")
    for epoch in range(EPOCH):
        mean, average_reward = drive_n_times(NO_PARAMETERS_SAMPLES, mean)
        max_reward = average_reward if average_reward > max_reward else max_reward
        ut.vector_to_parameters(mean, c.parameters())
        torch.save(c.state_dict(), "c_model.bin")
        print(
            f"EPOCH: {epoch}/{EPOCH}. Average Reward: {average_reward}. Maximum Reward: {max_reward}"
        )
