import torch
import torch.nn as nn
import torch.nn.functional as F


class tell_me_shape(nn.Module):
    def forward(self, x):
        print(x.shape)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


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
        Flatten(),
        nn.Dropout(0.5),
        nn.Linear(8192, 2048),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(2048, 100),
    )


class QuickNet(nn.Module):

    def __init__(self, backend, frontend=None):
        super(QuickNet, self).__init__()
        self.backend = backend
        self.frontend = frontend
        # freeze encoder weights
        for param in self.backend.parameters():
            param.requires_grad = False
        #   if no decoder loaded, then initialize with random weights
        if self.frontend is None:
            self.frontend = Architecture.frontend
            self.init_decoder_weights(mean=0.0, std=0.01)

    def init_decoder_weights(self, mean, std):
        for param in self.frontend.parameters():
            nn.init.normal_(param, mean=mean, std=std)

    def forward(self, x):
        encoded = self.backend(x)
        return self.frontend(encoded)
