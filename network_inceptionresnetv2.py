import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
import pretrainedmodels
from pretrainedmodels.models.inceptionresnetv2 import *

pretrained_settings = {
    'inceptionresnetv2': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        },
        'imagenet+background': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1001
        }
    }
}

def inceptionresnetv2_(num_classes=41, pretrained='imagenet'):
    r"""InceptionResNetV2 model architecture from the
    `"InceptionV4, Inception-ResNet..." <https://arxiv.org/abs/1602.07261>`_ paper.
    """
    if pretrained:
        settings = pretrained_settings['inceptionresnetv2'][pretrained]

        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = InceptionResNetV2(num_classes=1001)
        model.load_state_dict(model_zoo.load_url(settings['url']))
        
        #  self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        #  self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        #  model.last_linear = nn.Linear(1536, num_classes)
        
        #  if pretrained == 'imagenet':
            #  new_last_linear = nn.Linear(1536, 1000)
            #  new_last_linear.weight.data = model.last_linear.weight.data[1:]
            #  new_last_linear.bias.data = model.last_linear.bias.data[1:]
            #  model.last_linear = new_last_linear
        
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        
        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = InceptionResNetV2(num_classes=num_classes)
    return model

if __name__ == "__main__":
    model = inceptionresnetv2_()
    print(model)
