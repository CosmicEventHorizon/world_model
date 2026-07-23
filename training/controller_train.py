import torch
import torch.nn as nn
import numpy as np
import os
from ..model.vae import VAE
from ..model.controller import Controller
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, TransformObservation


def load_vae(device) -> VAE:
    vae = VAE().to(device)
    vae.load_state_dict(torch.load("vae_model.bin", map_location=device))
    return vae


def load_gym():
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
    env = TransformObservation(env, lambda obs: obs[:80, 0:, 0:], env.observation_space)
    env = ResizeObservation(env, (64, 64))
    return env


def hwc_to_chw(x):
    return x.permute(2, 0, 1).unsqueeze(0)


def chw_to_hwc(x):
    return x.squeeze(0).permute(1, 2, 0)


if __name__ == "__main__":
    env = load_gym()

    device = torch.device("cuda")
    vae = load_vae(device)
    c = Controller().to(device)
    optim = torch.optim.Adam(c.parameters(), lr=1e-4)

    episode_over = False
    reward_accum = 0
    no_train_frames = 10000

    print("Started training...")
    for frame in range(no_train_frames):
        optim.zero_grad()
        observation, info = env.reset()
        z = vae.encode(observation)
        steer, accelerate, brake = c.forward(z)
        action = np.array([steer, accelerate, brake])
        observation, reward, terminated, truncated, info = env.step(action)
        loss = reward
        loss.backward()
        optim.step()
        torch.save(c.state_dict(), "c_model.bin")
        original = (chw_to_hwc(x) * 255).to(torch.uint8).cpu().numpy()
        reconstruction = (chw_to_hwc(x_hat) * 255).to(torch.uint8).cpu().numpy()
        Image.fromarray(original).save("original.png")
        Image.fromarray(reconstruction).save("reconstruction.png")
