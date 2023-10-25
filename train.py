from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

import quicknet


def train_transform():
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    return transforms.Compose(transform_list)


if __name__ == '__main__':

    batch_size = 4
    train_set = CIFAR100(root='./data', train=True, download=True, transform=train_transform())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    encoder_path = Path('./encoder.pth')
    encoder = classifier.Architecture.backend
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))

