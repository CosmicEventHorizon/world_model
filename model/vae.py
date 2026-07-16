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
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        )

        self.fully_connected_variance = nn.Sequential(
            nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 1024)
        )

    def forward(self, x):
        x_conv = self.conv(x)
        x_conv = torch.flatten(x_conv)
        z_mean = self.fully_connected_mean(x_conv)
        z_logvar = self.fully_connected_variance(x_conv)
        return (z_mean, z_logvar)


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
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.deconv(x)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        (z_mean, z_logvar) = self.encoder(x)
        z_sample = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)
        x_hat = self.decoder(z_sample.view(1, 1024, 1, 1))
        return (x_hat, z_mean, z_logvar)
