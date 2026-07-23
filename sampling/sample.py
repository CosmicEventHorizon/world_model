import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, TransformObservation
import numpy as np
from CarActions import CarActions as ca
import os

NUMBER_OF_SAMPLES = 100000


def collectSamples(requested_no_samples: int):
    if os.path.isdir("data"):
        print("Data already exists!")
        return
    os.makedirs("data")
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
    env = TransformObservation(env, lambda obs: obs[:80, 0:, 0:], env.observation_space)
    env = ResizeObservation(env, (64, 64))
    actions, no_samples = ca.generate(requested_no_samples)
    actions_counter = 0
    data_counter = 0
    while actions_counter < no_samples:
        x = np.zeros(shape=(no_samples, 64, 64, 3))
        a = np.zeros(shape=(no_samples, 3))
        observation, info = env.reset()
        episode_over = False
        x[0] = observation
        sample_counter = 0
        while (
            not episode_over
            and actions_counter < no_samples
            and sample_counter < no_samples
        ):
            next_action = actions[actions_counter]
            a[sample_counter] = next_action
            print(f"Taking action {actions_counter}/{no_samples}")
            observation, reward, terminated, truncated, info = env.step(next_action)
            actions_counter += 1
            sample_counter += 1
            x[sample_counter] = observation
            episode_over = terminated
        x = x[: sample_counter + 1]
        a = a[: sample_counter + 1]
        filename = f"data/data{data_counter}"
        saveFile(filename, x, a)
        data_counter += 1
    env.close()


def saveFile(filename: str, observations: np.ndarray, actions: np.ndarray):
    np.savez(filename, x=observations, a=actions)
    saved_file = np.load(filename + ".npz")
    x = saved_file["x"]
    a = saved_file["a"]
    obs_shape = "Observations shape: " + str(x.shape)
    act_shape = "Actions shape: " + str(a.shape)
    last_act = "Last action check (should be 0,0,0): " + str(a[a.shape[0] - 1])
    last_obs = "Last obs check: " + str(x[x.shape[0] - 1])
    data_dim = obs_shape + "\n" + act_shape + "\n" + last_act + "\n" + last_obs
    with open(filename + ".txt", "w") as f:
        f.write(data_dim)
    print("Episode finished! Saved file under " + filename)


if __name__ == "__main__":
    collectSamples(NUMBER_OF_SAMPLES)
