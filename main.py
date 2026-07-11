import torch
import numpy as np
from model.vae import VAE


if __name__ == "__main__":
    saved_file = np.load("data/data0.npz")
    x = saved_file["x"]
    a = torch.from_numpy(x)
    print(a.shape)
    print(a[0].shape)
    vae = VAE()
    vae.train(a[0])
