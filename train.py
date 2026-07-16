import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from model.vae import VAE


def kl_loss_fn(z_mean, z_logvar):
    loss = 0
    no_elements = z_mean.numel()
    for i in range(no_elements):
        loss += -0.5 * (z_logvar[i] + 1 - z_mean[i] ** 2 - torch.exp(z_logvar[i]))
    return loss / no_elements


def hwc_to_chw(x):
    return x.permute(2, 0, 1).unsqueeze(0)


def chw_to_hwc(x):
    return x.squeeze(0).permute(1, 2, 0)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    vae = VAE().to(device)
    l2_loss = nn.MSELoss()
    if os.path.exists("model.bin"):
        vae.load_state_dict(torch.load("model.bin", map_location=device))
    else:
        optim = torch.optim.Adam(vae.parameters(), lr=1e-3)
        print("Started training...")
        for data_index in range(141):
            saved_file = np.load(f"data/data{data_index}.npz")
            f = saved_file["x"]
            x_all = torch.from_numpy(f).float() / 255.0
            loss_accum = 0.0
            no_images = x_all.shape[0]
            for image_index in range(no_images):
                optim.zero_grad()
                x = hwc_to_chw(x_all[image_index]).to(device)
                (x_hat, z_mean, z_logvar) = vae(x)
                loss = l2_loss(x, x_hat) + kl_loss_fn(z_mean, z_logvar)
                loss_accum += loss.item()
                loss.backward()
                optim.step()
            loss_accum = loss_accum / no_images
            print(
                f"Current iteration: data {data_index} with average loss {loss_accum}"
            )
            torch.save(vae.state_dict(), "model.bin")

    saved_file = np.load("data/data142.npz")
    x_all = torch.from_numpy(saved_file["x"]).float() / 255.0
    x = hwc_to_chw(x_all[100]).to(device)
    with torch.no_grad():
        x_hat, z_mean, z_logvar = vae(x)

    original = (chw_to_hwc(x).detach().cpu().clamp(0, 1) * 255).byte().numpy()
    reconstruction = (chw_to_hwc(x_hat).detach().cpu().clamp(0, 1) * 255).byte().numpy()
    comparison = np.concatenate((original, reconstruction), axis=1)
    Image.fromarray(comparison).show()
