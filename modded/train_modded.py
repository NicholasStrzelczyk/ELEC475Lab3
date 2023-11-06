import argparse
import time
import torch
import torchvision.transforms as transforms
from torchsummary import torchsummary
from matplotlib import pyplot as plt
from torch import nn
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

from torchvision.transforms import InterpolationMode

import quicknet_modded


def train_transform():
    transform_list = [
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    return transforms.Compose(transform_list)


def find_top_k(output, target, k):
    with torch.no_grad():
        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return res[0]


if __name__ == '__main__':

    device = torch.device('cpu')

    # ----- set up argument parser for command line inputs ----- #
    parser = argparse.ArgumentParser()
    parser.add_argument('-encoder_file', type=str, help='file name for saved model')
    parser.add_argument('-decoder_file', type=str, help='file name for saved model')
    parser.add_argument('-plot_file', type=str, help='file name for saved plot')
    parser.add_argument('-learn', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-e', type=int, default=20, help='Number of epochs')
    parser.add_argument('-b', type=int, default=20, help='Batch size')
    parser.add_argument('-cuda', type=str, default='n', help='[y/N]')

    # ----- get args from the arg parser ----- #
    args = parser.parse_args()
    learn = float(args.learn)
    n_epochs = int(args.e)
    batch_size = int(args.b)
    use_cuda = str(args.cuda).lower()
    encoder_file = Path(args.encoder_file)
    decoder_file = Path(args.decoder_file)
    plot_file = Path(args.plot_file)

    train_set = CIFAR100(root='../data', train=True, download=True, transform=train_transform())
    valid_set = CIFAR100(root='../data', train=False, download=True, transform=train_transform())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    data_length = len(train_set)
    n_batches = len(train_loader)

    print("CudaIsAvailable: {}, UseCuda: {}".format(torch.cuda.is_available(), use_cuda))
    if torch.cuda.is_available() and use_cuda == 'y':
        print('using cuda ...')
        device = torch.device('cuda')
    else:
        print('using cpu ...')

    # ----- initialize model and training parameters ----- #
    encoder = quicknet_modded.Architecture.backend
    decoder = quicknet_modded.Architecture.frontend
    encoder.to(device=device)
    decoder.to(device=device)
    model = quicknet_modded.QuickNetMod(encoder, decoder)
    model.to(device=device)
    print('model loaded OK!')

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=5, verbose=True)

    # ----- begin training the model ----- #
    loss_train = []
    acc_train = []
    acc_valid = []
    torchsummary.summary(model, input_size=(3, 224, 224))
    model.train()
    print("{} training...".format(datetime.now()))
    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step(epoch_loss)
        loss_train.append(epoch_loss / data_length)
        acc_train.append(100 * correct_train / total_train)
        print("{} Epoch {}, loss {:.7f}, train accuracy {:.2f}%".format(
            datetime.now(),
            epoch + 1,
            epoch_loss / data_length,
            100 * correct_train / total_train))

        # Validation
        with torch.no_grad():
            correct_valid = 0
            total_valid = 0
            top1s = 0
            top5s = 0
            for images, labels in valid_loader:
                images = images.to(device=device)
                labels = labels.to(device=device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()
                top1s += find_top_k(outputs, labels, 1)
                top5s += find_top_k(outputs, labels, 5)
                del images, labels, outputs

        acc_valid.append(100 * correct_valid / total_valid)
        print('Validation accuracy: {}%, Top 1 error rate: {:.2f}%, Top 5 error rate: {:.2f}%'.format(
            100 * correct_valid / total_valid,
            100 * (1 - (top1s / total_valid)),
            100 * (1 - (top5s / total_valid))))

    end_time = time.time()

    # # ----- print final training statistics ----- #
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    total_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    print("Total training time: {}, ".format(total_time))
    print("Final loss value: {}".format(loss_train[-1]))
    print("Best validation accuracy: {}% on epoch {}".format(max(acc_valid), acc_valid.index(max(acc_valid)) + 1))

    # save the model weights
    torch.save(model.backend.state_dict(), encoder_file)
    torch.save(model.frontend.state_dict(), decoder_file)

    # save loss plot and accuracy plot
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(loss_train, label='loss')
    plt.title("Training Loss Graph")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.subplot(1, 2, 2)
    plt.plot(acc_train, label='training acc')
    plt.plot(acc_valid, label='validation acc')
    plt.title("Training and Validation Accuracy Graph")
    plt.xlabel("epoch")
    plt.ylabel("accuracy (%)")
    plt.legend(loc=1)
    plt.savefig(plot_file)
    plt.show()
