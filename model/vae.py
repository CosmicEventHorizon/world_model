import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super.__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, stride=2),
            nn.RELU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.fully_connected_mean = nn.Sequential(
            nn.Linear(1024, 2048), nn.Linear(2048, 1024)
        )

        self.fully_connected_variance = nn.Sequential(
            nn.Linear(1024, 2048), nn.Linear(2048, 1024)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x)
        self.z_mean = self.fully_connected(x)
        self.z_variance = self.fully_connected_mean(x)


if __name__ == "__main__":
    print(torch.cuda.is_available())
