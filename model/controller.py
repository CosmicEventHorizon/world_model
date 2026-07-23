import torch
import torch.nn as nn


class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.fully_connected = nn.Sequential(
            nn.Linear(32, 64), nn.LeakyReLU(), nn.Linear(64, 32), nn.LeakyReLU()
        )
        self.fc_steer = nn.Sequential(nn.Linear(32, 1), nn.Tanh())
        self.fc_accelerate = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.fc_brake = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.fully_connected(x)
        steer = self.fc_steer(x)
        accelerate = self.fc_accelerate(x)
        brake = self.fc_brake(x)
        return steer, accelerate, brake
