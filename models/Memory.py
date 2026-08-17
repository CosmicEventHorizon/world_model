import torch
import torch.nn as nn

"""
h_t n_dim: 128
z_t n_dim: 32
a_t n_dim: 3
"""


class Memory(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(163, 326), nn.LeakyReLU(), nn.Linear(326, 128), nn.LeakyReLU()
        )
        self.z_next_mean = nn.Linear(128, 32)
        self.z_next_logvar = nn.Linear(128, 32)

    def forward(self, h_t, z_t, a_t):
        x_t = torch.cat((h_t, z_t, a_t))
        h_next = self.network(x_t)
        z_next_mean = self.z_next_mean(h_next)
        z_next_logvar = self.z_next_logvar(h_next)
        return (h_next, z_next_mean, z_next_logvar)
