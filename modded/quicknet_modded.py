import torch
import torch.nn as nn


class tell_me_shape(nn.Module):
    def forward(self, x):
        print(x.shape)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, num_1x1, num_3x3_red, num_3x3, num_5x5_red, num_5x5, num_pool_proj):
        super(InceptionBlock, self).__init__()
        self.one_by_one = ConvBlock(in_channels, num_1x1, kernel_size=1)
        self.three_by_three_red = ConvBlock(in_channels, num_3x3_red, kernel_size=1)
        self.three_by_three = ConvBlock(num_3x3_red, num_3x3, kernel_size=3, padding=1)
        self.five_by_five_red = ConvBlock(in_channels, num_5x5_red, kernel_size=1)
        self.five_by_five = ConvBlock(num_5x5_red, num_5x5, kernel_size=5, padding=2)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_proj = ConvBlock(in_channels, num_pool_proj, kernel_size=1)

    def forward(self, x):
        x1 = self.one_by_one(x)

        x2 = self.three_by_three_red(x)
        x2 = self.three_by_three(x2)

        x3 = self.five_by_five_red(x)
        x3 = self.five_by_five(x3)

        x4 = self.max_pool(x)
        x4 = self.pool_proj(x4)

        x = torch.cat([x1, x2, x3, x4], 1)
        return x


class Architecture:
    backend = nn.Sequential(
        ConvBlock(3, 64, kernel_size=7, stride=2, padding=3),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Dropout(0.4),
        # 2 inception blocks with output depth 256
        InceptionBlock(128, 64, 96, 128, 16, 32, 32),
        InceptionBlock(256, 32, 112, 128, 32, 64, 32),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Dropout(0.4),
        # 4 inception blocks with output depth 512
        InceptionBlock(256, 128, 128, 224, 32, 96, 64),
        InceptionBlock(512, 96, 144, 256, 32, 96, 64),
        InceptionBlock(512, 64, 144, 288, 64, 96, 64),
        InceptionBlock(512, 32, 144, 320, 64, 96, 64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Dropout(0.4),
        # 2 inception blocks with output depth 1024
        InceptionBlock(512, 384, 160, 384, 96, 128, 128),
        InceptionBlock(1024, 256, 192, 384, 112, 256, 128),
        nn.AvgPool2d(kernel_size=7, stride=1),
    )
    frontend = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(1024, 100),
    )


class QuickNet(nn.Module):

    def __init__(self):
        super(QuickNet, self).__init__()
        self.backend = Architecture.backend
        self.frontend = Architecture.frontend

    def forward(self, x):
        encoded = self.backend(x)
        flat = encoded.reshape(x.shape[0], -1)
        return self.frontend(flat)
