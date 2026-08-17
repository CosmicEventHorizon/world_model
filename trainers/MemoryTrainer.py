import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from models.VAE import VAE
from models.Memory import Memory


class MemoryTrainer:

    @staticmethod
    def current_directory_path(filename):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    @staticmethod
    def kl_loss_standard_error_fn(z_next_mean, z_next_logvar):
        loss = 0
        free_bits = 0.5
        no_elements = z_next_mean.numel()
        for i in range(no_elements):
            kl_i = -0.5 * (
                z_next_logvar[i] + 1 - z_next_mean[i] ** 2 - torch.exp(z_next_logvar[i])
            )
            if kl_i < free_bits:
                loss += free_bits
            else:
                loss += kl_i
        return loss

    @staticmethod
    def kl_loss_fn(q_mean, q_logvar, p_mean, p_logvar):
        loss = 0
        no_elements = q_mean.numel()
        for i in range(no_elements):
            q_var = torch.exp(q_logvar[i])
            p_var = torch.exp(p_logvar[i])
            kl_i = 0.5 * (
                (q_var / p_var)
                + ((q_mean[i] - p_mean[i]) ** 2 / p_var)
                - 1
                + p_logvar[i]
                - q_logvar[i]
            )
            loss += kl_i
        return loss

    @staticmethod
    def hwc_to_chw(x):
        return torch.as_tensor(x).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    @staticmethod
    def l2_norm_loss_fn(z_next, z_next_hat):
        return torch.sum((z_next - z_next_hat) ** 2)

    @staticmethod
    def load_samples(data_index):
        saved_file = np.load(f"data/data{data_index}.npz")
        x_raw = saved_file["x"]
        a_raw = saved_file["a"]
        a_all = torch.from_numpy(a_raw)
        x_all = torch.from_numpy(x_raw)
        return (x_all, a_all)

    @classmethod
    def train(cls, NO_EPOCHS):
        device = torch.device("cuda")
        memory = Memory().to(device)
        vae = VAE().to(device)
        vae.load_state_dict(
            torch.load(cls.current_directory_path("vae_model.bin"), map_location=device)
        )
        optim = torch.optim.Adam(memory.parameters(), lr=1e-4)
        print("Started training...")
        for epoch in range(NO_EPOCHS):
            for data_index in np.random.permutation(range(132)):
                optim.zero_grad()
                x_all, a_all = cls.load_samples(data_index)
                l2_loss_accum = 0.0
                kl_loss_accum = 0.0
                loss_accum = 0.0
                no_images = x_all.shape[0]
                x = cls.hwc_to_chw(x_all[0]).to(device)
                h_t = torch.as_tensor(np.zeros(shape=(128))).float().to(device)
                _, z_t, _ = vae.forward(x, False)
                for image_index in range(no_images - 1):
                    a_t = torch.as_tensor(a_all[image_index]).float().to(device)
                    h_next, z_next_mean_hat, z_next_logvar_hat = memory.forward(
                        h_t, z_t, a_t
                    )
                    x = cls.hwc_to_chw(x_all[image_index + 1]).to(device)
                    _, z_next_mean, z_next_logvar = vae.forward(x, False)
                    loss = cls.kl_loss_fn(
                        z_next_mean, z_next_logvar, z_next_mean_hat, z_next_logvar_hat
                    )
                    loss_accum += loss
                    h_t = h_next
                    z_t = z_next_mean_hat
                loss_accum.backward()
                optim.step()
                print(
                    f"Current iteration: EPOCH {epoch}, data {data_index} with average:\n  KL loss: {loss_accum / (no_images-1)} \n"
                )
                torch.save(
                    memory.state_dict(), cls.current_directory_path("m_model.bin")
                )
