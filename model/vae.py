import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2),
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
        z_mean = self.fully_connected_mean(x)
        z_variance = self.fully_connected_variance(x)
        return (z_mean, z_variance)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=128, kernel_size=5, stride=2
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=5, stride=2
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=6, stride=2
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2),
        )

    def forward(self, x):
        return self.deconv(x)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def train(self, x):
        (z_mean, z_variance) = self.encoder(x)
        z_sample = torch.normal(z_mean, z_variance * torch.eye(1024, 1))
        x_hat = self.decoder(z_sample)
        kl_loss = nn.KLDivLoss()
        loss = (
            torch.sub(x, x_hat).norm + kl_loss(z_sample, torch.normal(torch.zeros()))
        ).sum()
        loss.backward()
        optim = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        optim.step()
        torch.save(self.state_dict(), "model.bin")
