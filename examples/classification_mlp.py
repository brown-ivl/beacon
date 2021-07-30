import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

import sys
import argparse

from beacon.models import ClassificationNet

def infer(Args, TestData, Net, TestDevice):
    TestNet = Net.to(TestDevice)
    nSamples = min(Args.test_samples, len(TestData))
    print('[ INFO ]: Testing on', nSamples, 'samples')

    for i in range(nSamples):
        Image, GTLabel = TestData[i]
        # print(Image.size())
        Image = Image.to(TestDevice)
        PredLabel = TestNet(Image.unsqueeze_(0)).to('cpu').argmax().item()
        plt.imshow(Image.to('cpu').numpy().squeeze(), cmap='gray')
        plt.xlabel('GT: {}; Pred: {}'.format(GTLabel, PredLabel))
        plt.pause(1)


Parser = argparse.ArgumentParser(description='Sample code that uses the beacon framework for training a simple MNIST '
                                             'classification task.')
InputGroup = Parser.add_mutually_exclusive_group()
InputGroup.add_argument('--mode', help='Operation mode.', choices=['train', 'infer'])
InputGroup.add_argument('--test-samples', help='Number of samples to use during testing.', default=30, type=int)

MNISTClassTrans = transforms.Compose([
                                        # transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (1.0,))
                                     ])
MNISTCAETrans = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (1.0,))
                                     ])

if __name__ == '__main__':
    Args, _ = Parser.parse_known_args()
    if len(sys.argv) <= 1:
        Parser.print_help()
        exit()

    if Args.mode == 'train':
        SampleNet = ClassificationNet.SimpleClassNet()
        print(SampleNet)

        TrainDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        TrainData = MNIST(root=SampleNet.Config.Args.input_dir, train=True, download=True, transform=MNISTClassTrans)
        print('[ INFO ]: Data has', len(TrainData), 'samples.')
        TrainDataLoader = torch.utils.data.DataLoader(TrainData, batch_size=SampleNet.Config.Args.batch_size, shuffle=True, num_workers=4)

        # Train
        SampleNet.fit(TrainDataLoader, Objective=nn.NLLLoss(), TrainDevice=TrainDevice)
    elif Args.mode == 'infer':
        SampleNet = ClassificationNet.SimpleClassNet()
        SampleNet.loadCheckpoint()
        print(SampleNet)

        TestDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        TestData = MNIST(root=SampleNet.Config.Args.input_dir, train=False, download=True, transform=MNISTClassTrans)
        print('[ INFO ]: Data has', len(TestData), 'samples.')

        infer(Args, TestData, SampleNet, TestDevice)
