import argparse
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

import quicknet_vanilla


def find_top_k(output, target, k):
    with torch.no_grad():
        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return res[0]


def test_transform():
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    return transforms.Compose(transform_list)


if __name__ == '__main__':

    device = 'cpu'

    # ----- set up argument parser for command line inputs ----- #
    parser = argparse.ArgumentParser()
    parser.add_argument('-encoder_file', type=str, help='encoder weight file')
    parser.add_argument('-decoder_file', type=str, help='decoder weight file')
    parser.add_argument('-b', type=int, default=20, help='Batch size')
    parser.add_argument('-cuda', type=str, help='[y/N]')

    # ----- get args from the arg parser ----- #
    args = parser.parse_args()
    encoder_file = Path(args.encoder_file)
    decoder_file = Path(args.decoder_file)
    batch_size = int(args.b)
    use_cuda = str(args.cuda).lower()

    test_set = CIFAR100(root='../data', train=False, download=True, transform=test_transform())
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # ----- initialize model and training parameters ----- #
    encoder = quicknet_vanilla.Architecture.backend
    decoder = quicknet_vanilla.Architecture.frontend
    encoder.load_state_dict(torch.load(encoder_file, map_location='cpu'))
    decoder.load_state_dict(torch.load(decoder_file, map_location='cpu'))
    model = quicknet_vanilla.QuickNet(encoder, decoder)
    model.eval()
    model.to(device=torch.device('cpu'))
    print('model loaded OK!')

    print("CudaIsAvailable: {}, UseCuda: {}".format(torch.cuda.is_available(), use_cuda))
    if torch.cuda.is_available() and use_cuda == 'y':
        print('using cuda ...')
        model.cuda()
        device = torch.device('cuda')
    else:
        print('using cpu ...')

    # ----- begin testing the model ----- #
    with torch.no_grad():
        correct = 0
        total = 0
        top1s = 0
        top5s = 0

        for images, labels in test_loader:
            images = images.to(device=device)
            labels = labels.to(device=device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            top1s += find_top_k(outputs, labels, 1)
            top5s += find_top_k(outputs, labels, 5)
            del images, labels, outputs

        print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))
        print('Top 1 error rate: {:.2f} %'.format(100 * (1 - (top1s / total))))
        print('Top 5 error rate: {:.2f} %'.format(100 * (1 - (top5s / total))))
