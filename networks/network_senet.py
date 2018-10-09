import torch.nn as nn
import pretrainedmodels


def se_resnet50_(pretrained='imagenet', **kwargs):
    model = pretrainedmodels.se_resnet50(pretrained=pretrained)
    model.avg_pool = nn.AvgPool2d((2, 5), stride=(2, 5))
    model.last_linear = nn.Linear(512 * 4, 41)
    return model


def se_resnet101_(pretrained='imagenet', **kwargs):
    model = pretrainedmodels.se_resnet101(pretrained=pretrained)
    model.avg_pool = nn.AvgPool2d((2, 5), stride=(2, 5))
    model.last_linear = nn.Linear(512 * 4, 41)
    return model


def se_resnext50_32x4d_(pretrained='imagenet', **kwargs):
    model = pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained)
    model.avg_pool = nn.AvgPool2d((2, 5), stride=(2, 5))
    model.last_linear = nn.Linear(512 * 4, 41)
    return model


def se_resnext101_32x4d_(pretrained='imagenet', **kwargs):
    model = pretrainedmodels.se_resnext101_32x4d(pretrained=pretrained)
    model.avg_pool = nn.AvgPool2d((2, 5), stride=(2, 5))
    model.last_linear = nn.Linear(512 * 4, 41)
    return model
