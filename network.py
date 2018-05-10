import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

def num_flat_features(x):
    # (32L, 50L, 11L, 14L), 32 is batch_size
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class Wnet(nn.Module):
    def __init__(self, num_classes=41, init_weights=True):
        super(Wnet, self).__init__()

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=51, stride=5, padding=25),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=15, stride=15)
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.fc = nn.Sequential(
            nn.Linear(5120, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

        if init_weights:
            self._initialize_weights()

        # return layers

    def forward(self, x):
        # input: (batchSize, 1L, 33150L)
        x = self.conv1d(x)  # (bs, 32, 441)
        x = torch.unsqueeze(x, 1)  # (bs, 1, 32, 441)
        x = self.conv2d(x) # (bs, 256, 32, 441)
        x = x.view(-1, num_flat_features(x))
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def resnet18_m(pretrained=False, **kwargs):
    model = models.resnet18(pretrained=pretrained)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    model.avgpool = nn.AvgPool2d((2, 14), stride=(2, 14))
    model.fc = nn.Linear(512 * 1, 41)
    return model


def resnet101_m(pretrained=False, **kwargs):
    model = models.resnet101(pretrained=pretrained)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    model.avgpool = nn.AvgPool2d((2, 14), stride=(2, 14))
    model.fc = nn.Linear(512 * 4, 41)
    return model