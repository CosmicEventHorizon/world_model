import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from model.vae import VAE


def kl_loss_fn(z_mean, z_logvar):
    loss = 0
    free_bits = 0.5
    no_elements = z_mean.numel()
    for i in range(no_elements):
        kl_i = -0.5 * (z_logvar[i] + 1 - z_mean[i] ** 2 - torch.exp(z_logvar[i]))
        if kl_i < free_bits:
            loss += free_bits
        else:
            loss += kl_i
    return loss


def l2_norm_loss_fn(x, x_hat):
    return torch.sum((x - x_hat) ** 2)


def hwc_to_chw(x):
    return x.permute(2, 0, 1).unsqueeze(0)


def chw_to_hwc(x):
    return x.squeeze(0).permute(1, 2, 0)


if __name__ == "__main__":
    NO_EPOCH = 10
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    vae = VAE().to(device)
    if os.path.exists("model.bin"):
        vae.load_state_dict(torch.load("model.bin", map_location=device))
    else:
        optim = torch.optim.Adam(vae.parameters(), lr=1e-4)
        print("Started training...")
        for epoch in range(NO_EPOCH):
            for data_index in np.random.permutation(range(132)):
                saved_file = np.load(f"data/data{data_index}.npz")
                f = saved_file["x"]
                x_all = torch.from_numpy(f).float() / 255.0
                l2_loss_accum = 0.0
                kl_loss_accum = 0.0
                loss_accum = 0.0
                no_images = x_all.shape[0]
                for image_index in np.random.permutation(range(no_images)):
                    optim.zero_grad()
                    x = hwc_to_chw(x_all[image_index]).to(device)
                    (x_hat, z_mean, z_logvar) = vae(x, True)
                    l2_loss = l2_norm_loss_fn(x, x_hat)
                    l2_loss_accum += l2_loss.item()
                    kl_loss = kl_loss_fn(z_mean, z_logvar)
                    kl_loss_accum += kl_loss
                    loss = l2_loss + kl_loss
                    loss_accum += loss.item()
                    loss.backward()
                    optim.step()
                l2_loss_accum = l2_loss_accum / no_images
                kl_loss_accum = kl_loss_accum / no_images
                loss_accum = loss_accum / no_images
                print(
                    f"Current iteration: EPOCH {epoch}, data {data_index} with average:\n L2 loss: {l2_loss_accum} \n KL loss: {kl_loss_accum} \n Total loss: {loss_accum}"
                )
                torch.save(vae.state_dict(), "model.bin")

    saved_file = np.load("data/data133.npz")
    x_all = torch.from_numpy(saved_file["x"]).float() / 255.0
    x = hwc_to_chw(x_all[100]).to(device)
    x_hat, z_mean, z_logvar = vae(x, False)
    original = (chw_to_hwc(x) * 255).to(torch.uint8).cpu().numpy()
    reconstruction = (chw_to_hwc(x_hat) * 255).to(torch.uint8).cpu().numpy()
    Image.fromarray(original).save("original.png")
    Image.fromarray(reconstruction).save("reconstruction.png")
