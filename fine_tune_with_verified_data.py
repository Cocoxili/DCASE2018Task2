from core import *
from data_loader import *
from util import *

import torch.optim as optim
import torch.backends.cudnn as cudnn

import time

np.random.seed(1001)


def fine_tune():
    train = pd.read_csv('../input/train.csv')

    LABELS = list(train.label.unique())

    label_idx = {label: i for i, label in enumerate(LABELS)}
    train.set_index("fname")

    train["label_idx"] = train.label.apply(lambda x: label_idx[x])

    if DEBUG:
        train = train[:500]

    skf = StratifiedKFold(n_splits=config.n_folds)

    for foldNum, (train_split, val_split) in enumerate(skf.split(train, train.label_idx)):
        end = time.time()
        # split the dataset for cross-validation
        train_set = train.iloc[train_split]
        train_set = train_set.reset_index(drop=True)
        val_set = train.iloc[val_split]
        val_set = val_set.reset_index(drop=True)
        logging.info("Fold {0}, Train samples:{1}, val samples:{2}"
                     .format(foldNum, len(train_set), len(val_set)))

        # only train on manually_verified data
        train_set = train_set.loc[train_set['manually_verified'] == 1]
        train_set = train_set.reset_index(drop=True)
        
        # define train loader and val loader
        trainSet = Freesound_logmel(config=config, frame=train_set,
                                    transform=transforms.Compose([ToTensor()]),
                                    mode="train")
        train_loader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, num_workers=4)

        valSet = Freesound_logmel(config=config, frame=val_set,
                                  transform=transforms.Compose([ToTensor()]),
                                  mode="train")

        val_loader = DataLoader(valSet, batch_size=config.batch_size, shuffle=False, num_workers=4)


        # define loss function (criterion) and optimizer
        # criterion = nn.CrossEntropyLoss().cuda()
        train_criterion = cross_entropy_onehot
        val_criterion = nn.CrossEntropyLoss().cuda()

        cudnn.benchmark = True

        model = run_method_by_string(config.arch)(pretrained=config.pretrain)
        checkpoint = origin_model_dir + '/model_best.' + str(foldNum) + '.pth.tar'
        print("=> loading checkpoint '{}'".format(checkpoint))
        checkpoint = torch.load(checkpoint)
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()

        prec1, prec3 = val_on_fold(model, val_criterion, val_loader, config, foldNum)
        print("=> best_prec1: {:.2f}, actual_prec1: {:.2f}".format(best_prec1, prec1))

        optimizer = optim.SGD(model.parameters(), lr=config.lr,
                              momentum=config.momentum,
                              weight_decay=config.weight_decay)
        
        train_on_fold(model, train_criterion, val_criterion,
                     optimizer, train_loader, val_loader, config, foldNum)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    DEBUG = False
    #  DEBUG = True

    origin_model_dir = '../model/mixup_logmel_delta_dpn98'

    config = Config(sampling_rate=22050,
                    audio_duration=1.5,
                    batch_size=128,
                    n_folds=5,
                    data_dir="../logmel+delta_w80_s10_m64",
                    model_dir='../model/mixup_logmel_delta_dpn98_finetune',
                    prediction_dir='../prediction/mixup_logmel_delta_dpn98_finetune',
                    arch='dpn98_',
                    lr=0.0001,
                    pretrain='imagenet',
                    epochs=20)

    # create log
    logging = create_logging('../log', filemode='a')
    logging.info(os.path.abspath(__file__))
    attrs = '\n'.join('%s:%s' % item for item in vars(config).items())
    logging.info(attrs)

    fine_tune()
