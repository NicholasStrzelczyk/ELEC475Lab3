import argparse
import time
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch import nn
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

import quicknet


def train_transform():
    transform_list = [
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    return transforms.Compose(transform_list)


def adjust_learning_rate(optimizer, iteration_count):
    lr = args.learn / (1.0 + args.gamma * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    device = torch.device('cpu')

    # ----- set up argument parser for command line inputs ----- #
    parser = argparse.ArgumentParser()
    parser.add_argument('-learn', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-gamma', type=float, default=5e-5, help='Gamma value')
    parser.add_argument('-e', type=int, default=20, help='Number of epochs')
    parser.add_argument('-b', type=int, default=20, help='Batch size')
    parser.add_argument('-cuda', type=str, default='n', help='[y/N]')

    # ----- get args from the arg parser ----- #
    args = parser.parse_args()
    learn = float(args.learn)
    gamma = float(args.gamma)
    n_epochs = int(args.e)
    batch_size = int(args.b)
    use_cuda = str(args.cuda).lower()

    train_set = CIFAR100(root='./data', train=True, download=True, transform=train_transform())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    data_length = len(train_set)
    n_batches = len(train_loader)

    encoder_path = Path('./encoder.pth')
    encoder = quicknet.Architecture.backend
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    decoder = quicknet.Architecture.frontend
    decoder_path = Path('./decoder.pth')
    result_path = Path('./loss_plot.png')

    # ----- initialize model and training parameters ----- #
    model = quicknet.QuickNet(encoder, decoder)
    model.train()
    model.to(device=torch.device('cpu'))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(quicknet.Architecture.frontend.parameters(), lr=1e-4)
    print('model loaded OK!')

    print("CudaIsAvailable: {}, UseCuda: {}".format(torch.cuda.is_available(), use_cuda))
    if torch.cuda.is_available() and use_cuda == 'y':
        print('using cuda ...')
        model.cuda()
        device = torch.device('cuda')
    else:
        print('using cpu ...')

    # ----- begin training the model ----- #
    loss_train = []
    print("{} training...".format(datetime.now()))
    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if (i+1) % 1000 == 0:
                print("completed {}/{} batches".format(i+1, n_batches))

        adjust_learning_rate(optimizer=optimizer, iteration_count=epoch+1)
        loss_train.append(epoch_loss/data_length)
        print("{} Epoch {}, loss {}".format(datetime.now(), epoch+1, epoch_loss/data_length))

    end_time = time.time()

    # ----- print final training statistics ----- #
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    total_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    print("Total training time: {}".format(total_time))
    print("Final loss value: {}".format(loss_train[-1]))

    # save the decoder and plot file
    torch.save(model.decoder.state_dict(), decoder_path)
    plt.figure(figsize=(12, 7))
    plt.clf()
    plt.plot(loss_train, label='loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc=1)
    plt.savefig(result_path)
    plt.show()
