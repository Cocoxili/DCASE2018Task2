
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import pretrainedmodels


class WaveResnext(nn.Module):
    def __init__(self, modules, num_classes):
        super(WaveResnext, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=51, stride=5, padding=25)
        self.conv1_3 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=101, stride=10, padding=50)

        self.bn1_1 = nn.BatchNorm1d(32)
        self.bn1_2 = nn.BatchNorm1d(32)
        self.bn1_3 = nn.BatchNorm1d(32)

        self.conv2_1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.bn2_1 = nn.BatchNorm1d(32)
        self.bn2_2 = nn.BatchNorm1d(32)
        self.bn2_3 = nn.BatchNorm1d(32)

        self.pool2_1 = nn.MaxPool1d(kernel_size=150, stride=150)
        self.pool2_2 = nn.MaxPool1d(kernel_size=30, stride=30)
        self.pool2_3 = nn.MaxPool1d(kernel_size=15, stride=15)

        self.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.resBlocks = nn.Sequential(*modules)

        self.avgpool = nn.AvgPool2d((3, 14), stride=(1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x1 = self.relu(self.bn1_1(self.conv1_1(x)))
        x2 = self.relu(self.bn1_2(self.conv1_2(x)))
        x3 = self.relu(self.bn1_3(self.conv1_3(x)))

        x1 = self.relu(self.bn2_1(self.conv2_1(x1)))
        x2 = self.relu(self.bn2_2(self.conv2_2(x2)))
        x3 = self.relu(self.bn2_3(self.conv2_3(x3)))

        x1 = self.pool2_1(x1)
        x2 = self.pool2_2(x2)
        x3 = self.pool2_3(x3)  # (batchSize, 32L, 441L)

        x1 = torch.unsqueeze(x1, 1)
        x2 = torch.unsqueeze(x2, 1)
        x3 = torch.unsqueeze(x3, 1)  # (batchSize, 1L, 32L, 441L)

        x = torch.cat((x1, x2, x3), dim=2)  # (batchSize, 1L, 96L, 441L)
        # x = torch.cat((x1, x2, x3), dim=1)  # (batchSize, 3L, 64L, 441L)

        x = self.conv0(x)

        x = self.resBlocks(x)  # [bs, 2048, 3, 14]
        # print(x.size())
        x = self.avgpool(x)  # [bs, 2048, 1, 1]

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def waveResnext101_32x4d(pretrained='imagenet', num_classes=41):

    base = pretrainedmodels.resnext101_32x4d(pretrained=pretrained)
    modules = list(base.children())[0]
    modules = modules[1:]  # We need 1 channel input, remove first conv layer.
    model = WaveResnext(modules, num_classes)
    return model


def waveResnext101_64x4d(pretrained='imagenet', num_classes=41):

    base = pretrainedmodels.resnext101_64x4d(pretrained=pretrained)
    modules = list(base.children())[0]
    model = WaveResnext(modules, num_classes)
    return model


if __name__ == '__main__':
    model = waveResnext101_32x4d(pretrained='imagenet')
    print(model)

