import torch
import torch.nn as nn


class Memory(nn.Module):
    def __init__():
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(160, 320), nn.Sigmoid(), nn.Linear(320, 128), nn.Sigmoid()
        )
        self.last_layer = nn.Sequential(nn.Linear(128, 32), nn.Sigmoid())

    def forward(self, h_t_minus_1, z_t):
        x_t = torch.cat(h_t_minus_1, z_t)
        h_t = self.network(x_t)
        z_hat_t_plus_1 = self.last_layer(h_t)
        return (z_hat_t_plus_1, h_t)
