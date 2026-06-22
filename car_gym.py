import gymnasium as gym
from gymnasium.wrappers import ResizeObservation
import numpy as np
from sampling.CarActions import CarActions as ca


def main_program():
    env = gym.make("CarRacing-v3", render_mode="human", continuous=True)
    env = ResizeObservation(env, (64,64))
    observation, info = env.reset()
    no_samples = 5000
    x = np.zeros(shape=(no_samples,64,64,3))
    a = np.zeros(shape=(no_samples,3))
    x[0] = observation
    episode_over = False
    counter = 1
    while not episode_over and counter < no_samples:
        action = ca.generate()
        observation, reward, terminated, truncated, info = env.step(action)
        x[counter] = observation
        a[counter] = action
        counter = counter + 1
        episode_over = terminated
    x = x[:counter]
    a = a[:counter]
    np.savez("data", x=x, a=a)
    print(f"Episode finished! Saved file")
    env.close()



if __name__ == "__main__":
    main_program()
    saved_file = np.load("data.npz")
    print(saved_file['x'].shape)
    print(saved_file['a'].shape)
