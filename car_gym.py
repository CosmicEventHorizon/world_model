import gymnasium as gym
from gymnasium.wrappers import ResizeObservation
import numpy as np
from sampling.CarActions import CarActions as ca
import cv2

def main_program():
    env = gym.make("CarRacing-v3", render_mode="human", continuous=True)
    env = ResizeObservation(env, (64,64))
    observation, info = env.reset()
    no_samples = 5000
    x = np.zeros(shape=(no_samples,64,64,3))
    a = np.zeros(shape=(no_samples,3))
    episode_over = False
    counter = 0
    x[0] = observation
    while not episode_over and counter < no_samples:
        action = ca.generate()
        observation, reward, terminated, truncated, info = env.step(action)
        a[counter] = action
        counter = counter + 1
        x[counter] = observation
        episode_over = terminated
    x = x[:counter]
    a = a[:counter]
    np.savez("data/data", x=x, a=a)
    print(f"Episode finished! Saved file")
    env.close()



if __name__ == "__main__":
    main_program()
    saved_file = np.load("data/data.npz")
    x = saved_file['x']
    a = saved_file['a']
    print(x.shape)
    print(a.shape)