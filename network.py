import torch
import torch.nn as nn
from torchsummary import summary


class CNNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=4,
        stride=2,
        padding=0,
        dropout=0,
        use_norm=True,
        inst_norm=True,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential()

        self.net.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
        )
        self.net.append(nn.Mish())
        if use_norm:
            if inst_norm:
                self.net.append(nn.InstanceNorm2d(out_channels))
            else:
                self.net.append(nn.BatchNorm2d(out_channels))
        if dropout > 0:
            self.net.append(nn.Dropout2d(dropout))

    def forward(self, X):
        return self.net(X)


class ResidualBlock(nn.Module):
    def __init__(
        self, channels, kernel=2, padding=1, use_bias=True, dropout=0, inst_norm=True
    ):
        super().__init__()

        self.net = nn.Sequential()

        self.net.append(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel,
                padding=padding,
                bias=use_bias,
                padding_mode="reflect",
            )
        )
        if inst_norm:
            self.net.append(nn.InstanceNorm2d(channels))
        else:
            self.net.append(nn.BatchNorm2d(channels))
        self.net.append(nn.ReLU())
        if dropout > 0:
            self.net.append(nn.Dropout2d(dropout))
        self.net.append(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel,
                padding=0,
                bias=use_bias,
                padding_mode="reflect",
            )
        )
        if inst_norm:
            self.net.append(nn.InstanceNorm2d(channels))
        else:
            self.net.append(nn.BatchNorm2d(channels))

    def forward(self, x):
        return x + self.net(x)


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels=3,
        depth=4,
        res_layers=1,
        kernel_size=4,
        classes=1,
        base_channels=16,
        dropout=0,
        inst_norm=True,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential()

        self.net.append(
            CNNLayer(
                in_channels,
                base_channels,
                kernel=7,
                stride=1,
                use_norm=False,
                inst_norm=inst_norm,
            )
        )
        for i in range(depth):
            self.net.append(
                CNNLayer(
                    base_channels * 2**i,
                    base_channels * 2 ** (i + 1),
                    kernel=kernel_size,
                    inst_norm=inst_norm,
                    dropout=dropout,
                )
            )
        for _ in range(res_layers):
            self.net.append(
                ResidualBlock(
                    base_channels * 2 ** (i + 1),
                    kernel=kernel_size,
                    padding=kernel_size - 1,
                    dropout=dropout,
                    inst_norm=inst_norm,
                )
            )
        self.net.append(
            CNNLayer(
                base_channels * 2 ** (i + 1),
                1,
                kernel=kernel_size,
                stride=1,
                inst_norm=inst_norm,
            )
        )
        self.net.append(nn.AdaptiveAvgPool2d(3))
        self.net.append(nn.Flatten())
        self.net.append(nn.Linear(9, classes))

    def forward(self, X):
        return self.net(X)


if __name__ == "__main__":
    model = Classifier(base_channels=16, depth=3, res_layers=2, kernel_size=5)
    summary(model, torch.ones(1, 3, 600, 600), device="cpu", depth=8)

    print(model(torch.ones(4, 3, 600, 600)))
