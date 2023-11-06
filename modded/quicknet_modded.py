import torch
import torch.nn as nn


class tell_me_shape(nn.Module):
    def forward(self, x):
        print(x.shape)

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


# class simple_resnext_block(nn.Module):
#     def __init__(self, in_channels, cardinality, bwidth, stride=1):
#         super(simple_resnext_block, self).__init__()
#         self.expansion = 2
#         out_channels = cardinality * bwidth
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, groups=cardinality, stride=stride, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
#         self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         identity = x
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#
#         print(identity.shape)
#         print(x.shape)
#         x += identity
#         x = self.relu(x)
#         return x

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
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
    )
    frontend = nn.Sequential(
        InceptionBlock(512, 112, 144, 288, 32, 64, 64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Dropout(0.4),
        InceptionBlock(528, 256, 160, 320, 32, 128, 128),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Dropout(0.4),
        InceptionBlock(832, 384, 192, 384, 48, 128, 128),
        nn.AvgPool2d(kernel_size=7, stride=1),
        Flatten(),
        nn.Dropout(0.4),
        nn.Linear(1024, 100),
    )


class QuickNetMod(nn.Module):

    def __init__(self, backend, frontend):
        super(QuickNetMod, self).__init__()
        self.backend = backend
        self.frontend = frontend
        # freeze encoder weights
        for param in self.backend.parameters():
            param.requires_grad = False

    def forward(self, x):
        encoded = self.backend(x)
        return self.frontend(encoded)
