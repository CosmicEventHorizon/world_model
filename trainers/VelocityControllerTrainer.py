import torch
import torch.nn as nn
import torch.nn.utils as ut
import numpy as np
import os
from models.VAE import VAE
from models.VelocityController import VelocityController
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, TransformObservation


class VelocityControllerTrainer:

    @staticmethod
    def current_directory_path(filename):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    @classmethod
    def drive_n_times(cls, n, mean, vae, c):
        LEARNING_RATE = 0.005
        PARAMETER_SD = 0.05
        positive_rewards = []
        negative_rewards = []
        noises = []
        for sample in range(n):
            rollout_seed = np.random.randint(0, 2**31)
            noise = torch.randn_like(ut.parameters_to_vector(c.parameters()))
            noises.append(noise)
            positive_sampled_c_parameters = mean + PARAMETER_SD * noise
            ut.vector_to_parameters(positive_sampled_c_parameters, c.parameters())
            positive_reward = cls.start_driving(vae, c, seed=rollout_seed)
            positive_rewards.append(positive_reward)

            negative_sampled_c_parameters = mean - PARAMETER_SD * noise
            ut.vector_to_parameters(negative_sampled_c_parameters, c.parameters())
            negative_reward = cls.start_driving(vae, c, seed=rollout_seed)
            negative_rewards.append(negative_reward)

            print(
                f"Drove {sample+1}/{n} times. Positive Reward: {positive_reward}. Negative Reward: {negative_reward}"
            )

        update = cls.calculate_update(positive_rewards, negative_rewards, noises)
        mean = mean + (LEARNING_RATE / (PARAMETER_SD * n)) * update
        rewards = positive_rewards + negative_rewards
        average_reward = sum(rewards) / len(rewards)
        return mean, average_reward

    @classmethod
    def start_driving(cls, vae, c, render_mode="rgb_array", seed=None):
        env = cls.load_gym(render_mode)
        observation, info = env.reset(seed=seed)
        episode_over = False
        reward_accum = 0
        device = next(c.parameters()).device
        while not episode_over:
            observation = cls.hwc_to_chw(observation).to(device)
            z = vae.encode(observation)
            velocity = cls.get_velocity(env, device)
            action = cls.generate_action(z, velocity, c)
            observation, reward, terminated, truncated, info = env.step(action)
            reward_accum += reward
            episode_over = terminated or truncated
        env.close()
        return reward_accum

    @classmethod
    def preview_model(cls, vae, c):
        cls.start_driving(vae, c, "human")

    @classmethod
    def load_vae(cls, device) -> VAE:
        vae = VAE().to(device)
        vae.load_state_dict(
            torch.load(cls.current_directory_path("vae_model.bin"), map_location=device)
        )
        return vae

    @staticmethod
    def load_gym(render_mode):
        env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=True)
        env = TransformObservation(
            env, lambda obs: obs[:80, 0:, 0:], env.observation_space
        )
        env = ResizeObservation(env, (64, 64))
        return env

    @staticmethod
    def hwc_to_chw(x):
        return torch.as_tensor(x).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    @staticmethod
    def get_velocity(env, device):
        linear_velocity = env.unwrapped.car.hull.linearVelocity
        angular_velocity = env.unwrapped.car.hull.angularVelocity
        return torch.tensor(
            [linear_velocity.x, linear_velocity.y, angular_velocity],
            dtype=torch.float32,
            device=device,
        )

    @staticmethod
    def generate_action(z, velocity, c):
        z_with_velocity = torch.cat((z, velocity))
        steer, accelerate, brake = c.forward(z_with_velocity)
        return np.array(
            [
                steer[0].detach().cpu().item(),
                accelerate[0].detach().cpu().item(),
                brake[0].detach().cpu().item(),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def calculate_update(positive_rewards, negative_rewards, noises):
        length = len(positive_rewards)
        sum = 0
        for i in range(length):
            reward_difference = positive_rewards[i] - negative_rewards[i]
            sum += reward_difference * noises[i]
        return sum

    @classmethod
    def train(cls, NO_EPOCHS):
        NO_PARAMETERS_SAMPLES = 16

        device = torch.device("cuda")
        vae = cls.load_vae(device)
        c = VelocityController().to(device)
        if os.path.exists(cls.current_directory_path("v_c_model.bin")):
            answer = input(
                "v_c_model.bin exists. Continue training from saved parameters? [y/n]: "
            )
            if answer.lower() == "y":
                c.load_state_dict(
                    torch.load(
                        cls.current_directory_path("v_c_model.bin"), map_location=device
                    )
                )
            else:
                c.load_state_dict(
                    torch.load(
                        cls.current_directory_path("v_c_model.bin"), map_location=device
                    )
                )
                cls.preview_model(vae, c)
                quit()

        mean = ut.parameters_to_vector(c.parameters()).clone()

        print("Started training...")
        max_reward = cls.start_driving(vae, c)
        for epoch in range(NO_EPOCHS):
            mean, average_reward = cls.drive_n_times(
                NO_PARAMETERS_SAMPLES, mean, vae, c
            )

            ut.vector_to_parameters(mean, c.parameters())
            mean_reward = cls.start_driving(vae, c)

            if mean_reward > max_reward:
                max_reward = mean_reward
                torch.save(c.state_dict(), cls.current_directory_path("v_c_model.bin"))

            print("**************************************************************\n")
            print(
                f"EPOCH: {epoch+1}/{NO_EPOCHS}. Average Reward: {average_reward}. Mean Reward: {mean_reward}. Maximum Reward: {max_reward}"
            )
            print("\n**************************************************************")
