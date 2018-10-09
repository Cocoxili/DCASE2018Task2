import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
import pretrainedmodels
from pretrainedmodels.models.dpn import *

pretrained_settings = {
    'dpn68': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn68-66bebafa7.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [124 / 255, 117 / 255, 104 / 255],
            'std': [1 / (.0167 * 255)] * 3,
            'num_classes': 1000
        }
    },
    'dpn68b': {
        'imagenet+5k': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn68b_extra-84854c156.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [124 / 255, 117 / 255, 104 / 255],
            'std': [1 / (.0167 * 255)] * 3,
            'num_classes': 1000
        }
    },
    'dpn92': {
        # 'imagenet': {
        #     'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn68-66bebafa7.pth',
        #     'input_space': 'RGB',
        #     'input_size': [3, 224, 224],
        #     'input_range': [0, 1],
        #     'mean': [124 / 255, 117 / 255, 104 / 255],
        #     'std': [1 / (.0167 * 255)] * 3,
        #     'num_classes': 1000
        # },
        'imagenet+5k': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-b040e4a9b.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [124 / 255, 117 / 255, 104 / 255],
            'std': [1 / (.0167 * 255)] * 3,
            'num_classes': 1000
        }
    },
    'dpn98': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn98-5b90dec4d.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [124 / 255, 117 / 255, 104 / 255],
            'std': [1 / (.0167 * 255)] * 3,
            'num_classes': 1000
        }
    },
    'dpn131': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn131-71dfe43e0.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [124 / 255, 117 / 255, 104 / 255],
            'std': [1 / (.0167 * 255)] * 3,
            'num_classes': 1000
        }
    },
    'dpn107': {
        'imagenet+5k': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn107_extra-1ac7121e2.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [124 / 255, 117 / 255, 104 / 255],
            'std': [1 / (.0167 * 255)] * 3,
            'num_classes': 1000
        }
    }
}


def dpn68_(num_classes=41, pretrained='imagenet'):
    model = DPN(
        small=True, num_init_features=10, k_r=128, groups=32,
        k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64),
        num_classes=num_classes, test_time_pool=False)
    if pretrained:
        model = DPN(
            small=True, num_init_features=10, k_r=128, groups=32,
            k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64),
            num_classes=1000, test_time_pool=False)
        settings = pretrained_settings['dpn68'][pretrained]
        #  assert num_classes == settings['num_classes'], \
            #  "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
        model.classifier = nn.Conv2d(832, num_classes, kernel_size=1, bias=True)
    return model


def dpn98_(num_classes=41, pretrained='imagenet'):
    model = DPN(
        num_init_features=96, k_r=160, groups=40,
        k_sec=(3, 6, 20, 3), inc_sec=(16, 32, 32, 128),
        num_classes=num_classes, test_time_pool=False)
    if pretrained:
        model = DPN(
            num_init_features=96, k_r=160, groups=40,
            k_sec=(3, 6, 20, 3), inc_sec=(16, 32, 32, 128),
            num_classes=1000, test_time_pool=False)
        settings = pretrained_settings['dpn98'][pretrained]
        #  assert num_classes == settings['num_classes'], \
            #  "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
        
        model.classifier = nn.Conv2d(2688, num_classes, kernel_size=1, bias=True)
    return model


def dpn107_(num_classes=41, pretrained='imagenet+5k'):
    model = DPN(
        num_init_features=128, k_r=200, groups=50,
        k_sec=(4, 8, 20, 3), inc_sec=(20, 64, 64, 128),
        num_classes=num_classes, test_time_pool=False)
    if pretrained:
        model = DPN(
            num_init_features=128, k_r=200, groups=50,
            k_sec=(4, 8, 20, 3), inc_sec=(20, 64, 64, 128),
            num_classes=1000, test_time_pool=False)
        settings = pretrained_settings['dpn107'][pretrained]
        #  assert num_classes == settings['num_classes'], \
            #  "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
        model.classifier = nn.Conv2d(2688, num_classes, kernel_size=1, bias=True)
    return model

