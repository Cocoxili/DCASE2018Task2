from core import *

import torch.optim as optim
import torch.backends.cudnn as cudnn

import time


def main():

    make_dirs()

    train = pd.read_csv('../train.csv')

    LABELS = list(train.label.unique())
    # ['Hi-hat', 'Saxophone', 'Trumpet', 'Glockenspiel', 'Cello', 'Knock',
    # 'Gunshot_or_gunfire', 'Clarinet', 'Computer_keyboard', 'Keys_jangling',
    # 'Snare_drum', 'Writing', 'Laughter', 'Tearing', 'Fart', 'Oboe', 'Flute',
    # 'Cough', 'Telephone', 'Bark', 'Chime', 'Bass_drum', 'Bus', 'Squeak',
    # 'Scissors', 'Harmonica', 'Gong', 'Microwave_oven', 'Burping_or_eructation',
    # 'Double_bass', 'Shatter', 'Fireworks', 'Tambourine', 'Cowbell',
    # 'Electric_piano', 'Meow', 'Drawer_open_or_close', 'Applause', 'Acoustic_guitar',
    # 'Violin_or_fiddle', 'Finger_snapping']

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

        # define train loader and val loader
        trainSet = Freesound(config=config, frame=train_set,
                             transform=transforms.Compose([
                                 ToLogMel(config),
                                 ToTensor()
                             ]),
                             mode="train")
        train_loader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, num_workers=4)

        valSet = Freesound(config=config, frame=val_set,
                             transform=transforms.Compose([
                                 ToLogMel(config),
                                 ToTensor()
                             ]),
                             mode="train")
        val_loader = DataLoader(valSet, batch_size=config.batch_size, shuffle=True, num_workers=4)

        model = run_method_by_string(config.arch)(pretrained=config.pretrain)

        if config.cuda:
            model.cuda()

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda()

        optimizer = optim.SGD(model.parameters(), lr=config.lr,
                              momentum=config.momentum,
                              weight_decay=config.weight_decay)

        cudnn.benchmark = True

        train_on_fold(model, criterion, optimizer, train_loader, val_loader, foldNum)
        val_on_fold(model, criterion, val_loader, foldNum)

        time_on_fold = time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-end))
        logging.info("--------------Time on fold {}: {}--------------\n"
              .format(foldNum, time_on_fold))



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    DEBUG = False

    config = Config(arch='resnet101_m',
                    pretrain=True,
                    epochs=40)

    # create log
    logging = create_logging('../log', filemode='a')
    logging.info(os.path.abspath(__file__))
    attrs = '\n'.join('%s:%s' % item for item in vars(config).items())
    logging.info(attrs)

    main()