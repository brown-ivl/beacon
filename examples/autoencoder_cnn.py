import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

import sys
import argparse

from beacon.models import CAE, SegNet

# Make both input and target be the same
class MNISTSpecialDataset(MNIST):
    def __getitem__(self, idx):
        Image, Label = super().__getitem__(idx)
        return Image, Image


def infer(Args, TestData, Net, TestDevice):
    TestNet = Net.to(TestDevice)
    nSamples = min(Args.infer_samples, len(TestData))
    print('[ INFO ]: Testing on', nSamples, 'samples')

    for i in range(nSamples):
        Image, _ = TestData[i]
        Image = Image.to(TestDevice)
        PredImage = TestNet(Image.unsqueeze_(0)).detach()
        plt.subplot(2, 1, 1)
        plt.imshow(Image.cpu().numpy().squeeze(), cmap='gray')
        plt.subplot(2, 1, 2)
        plt.imshow(PredImage.cpu().numpy().squeeze(), cmap='gray')
        plt.pause(1)


Parser = argparse.ArgumentParser(description='Sample code that uses the beacon framework for training a simple '
                                             'autoencoder on MNIST.')
Parser.add_argument('--arch', help='Architecture to use.', choices=['SimpleCNN', 'SegNet'], default='SimpleCNN')
InputGroup = Parser.add_mutually_exclusive_group()
InputGroup.add_argument('--mode', help='Operation mode.', choices=['train', 'infer'])
InputGroup.add_argument('--infer-samples', help='Number of samples to use during testing.', default=30, type=int)

MNISTTrans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
SegNetTrans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,)), transforms.Resize((64, 64))])

if __name__ == '__main__':
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    if Args.arch == 'SegNet':
        SampleNet = SegNet.SegNet(n_classes=1, in_channels=1, pretrained=False)
        Trans = SegNetTrans
    else:
        SampleNet = CAE.SimpleCAE()
        Trans = MNISTTrans

    if Args.mode == 'train':
        TrainDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        TrainData = MNISTSpecialDataset(root=SampleNet.Config.Args.input_dir, train=True, download=True, transform=Trans)
        print('[ INFO ]: Data has', len(TrainData), 'samples.')
        TrainDataLoader = torch.utils.data.DataLoader(TrainData, batch_size=SampleNet.Config.Args.batch_size, shuffle=True, num_workers=1)

        # Train
        SampleNet.fit(TrainDataLoader, Objective=nn.MSELoss(), TrainDevice=TrainDevice)
    elif Args.mode == 'infer':
        SampleNet.loadCheckpoint()

        TestDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        TestData = MNISTSpecialDataset(root=SampleNet.Config.Args.input_dir, train=False, download=True, transform=Trans)
        print('[ INFO ]: Data has', len(TestData), 'samples.')

        infer(Args, TestData, SampleNet, TestDevice)
