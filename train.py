import torch
import torch.nn as nn
import numpy as np
from model.vae import VAE


def kl_loss_fn(z_mean, z_logvar):
    loss = 0
    no_elements = z_mean.numel()
    for i in range(no_elements):
        loss += -0.5 * (z_logvar[i] + 1 - z_mean[i] ** 2 - torch.exp(z_logvar[i]))
    return loss / no_elements


def hwc_to_chw(x):
    return x.permute(2, 0, 1).unsqueeze(0)


if __name__ == "__main__":
    saved_file = np.load("data/data0.npz")
    f = saved_file["x"]
    x_all = torch.from_numpy(f).float()
    vae = VAE()
    optim = torch.optim.SGD(vae.parameters(), lr=1e-2, momentum=0.9)
    for image_index in range(x_all.shape[0]):
        print(image_index)
        optim.zero_grad()
        x = hwc_to_chw(x_all[image_index])
        (x_hat, z_mean, z_logvar) = vae(x)
        l2_loss = nn.MSELoss()
        loss = l2_loss(x, x_hat) + kl_loss_fn(z_mean, z_logvar)
        loss.backward()
        optim.step()
    torch.save(vae.state_dict(), "model.bin")
