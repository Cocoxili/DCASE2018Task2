from core import *
from data_loader import *
from util import *

import torch.optim as optim
import torch.backends.cudnn as cudnn

import time


def main():

    train = pd.read_csv('../input/train.csv')

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

    if config.debug == True:
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
        trainSet = Freesound(config=config, frame=train_set, mode="train")
        train_loader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, num_workers=4)

        valSet = Freesound(config=config, frame=val_set, mode="train")

        val_loader = DataLoader(valSet, batch_size=config.batch_size, shuffle=False, num_workers=4)

        model = run_method_by_string(config.arch)(pretrained=config.pretrain)

        if config.cuda:
            model.cuda()

        # define loss function (criterion) and optimizer
        if config.mixup:
            train_criterion = cross_entropy_onehot
        else:
            train_criterion = nn.CrossEntropyLoss().cuda()
        val_criterion = nn.CrossEntropyLoss().cuda()

        optimizer = optim.SGD(model.parameters(), lr=config.lr,
                              momentum=config.momentum,
                              weight_decay=config.weight_decay)
        #  optimizer = optim.Adam(model.parameters(), lr=config.lr)

        cudnn.benchmark = True

        train_on_fold(model, train_criterion, val_criterion,
                      optimizer, train_loader, val_loader, config, foldNum)

        # val_on_file_wave(model, config, val_set)

        time_on_fold = time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-end))
        logging.info("--------------Time on fold {}: {}--------------\n"
              .format(foldNum, time_on_fold))

    # # train on the whole training set
    # foldNum = config.n_folds
    # end = time.time()
    # logging.info("Fold {0}, Train samples:{1}."
    #              .format(foldNum, len(train)))
    #
    # # define train loader and val loader
    # trainSet = Freesound(config=config, frame=train, mode="train")
    # train_loader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, num_workers=4)
    #
    # model = run_method_by_string(config.arch)(pretrained=config.pretrain)
    #
    # if config.cuda:
    #     model.cuda()
    #
    # # define loss function (criterion) and optimizer
    # # criterion = nn.CrossEntropyLoss().cuda()
    # train_criterion = cross_entropy_onehot
    # val_criterion = nn.CrossEntropyLoss().cuda()
    #
    #
    # optimizer = optim.SGD(model.parameters(), lr=config.lr,
    #                       momentum=config.momentum,
    #                       weight_decay=config.weight_decay)
    # # optimizer = optim.Adam(model.parameters(), lr=config.lr)
    #
    # cudnn.benchmark = True
    #
    # train_all_data(model, train_criterion,
    #                optimizer, train_loader, config, foldNum)
    #
    # time_on_fold = time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time() - end))
    # logging.info("--------------Time on fold {}: {}--------------\n"
    #              .format(foldNum, time_on_fold))


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "5"
    
    # config = Config(debug=False,
    #                 sampling_rate=22050,
    #                 audio_duration=2,
    #                 data_dir="../data-22050",
    #                 arch='waveResnet18',
    #                 lr=0.01,
    #                 pretrain=False,
    #                 epochs=50)

    config = Config(debug=False,
                    mixup=False,
                    n_folds=5,
                    sampling_rate=44100,
                    audio_duration=1.5,
                    batch_size=32,
                    data_dir="../data-44100",
                    arch='waveResnext101_32x4d',
                    model_dir='../model/waveResnext101_32x4d',
                    prediction_dir='../prediction/waveResnext101_32x4d',
                    lr=0.01,
                    pretrain='imagenet',
                    print_freq=60,
                    epochs=50)

    # config = Config(debug=False,
    #                 n_folds=5,
    #                 sampling_rate=44100,
    #                 audio_duration=1.5,
    #                 batch_size=32,
    #                 data_dir="../data-44100",
    #                 arch='waveResnet101',
    #                 model_dir='../model/MTOwaveResnet101_3c',
    #                 prediction_dir='../prediction/MTOwaveResnet101_3c',
    #                 lr=0.01,
    #                 pretrain=True,
    #                 print_freq=30,
    #                 epochs=50)

    # create log
    logging = create_logging('../log', filemode='a')
    logging.info(os.path.abspath(__file__))
    attrs = '\n'.join('%s:%s' % item for item in vars(config).items())
    logging.info(attrs)

    main()
