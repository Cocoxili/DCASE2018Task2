
from util import *
from data_loader import *
from network import *
from tqdm import tqdm
import torch.nn.functional as F

def predict_one_model(checkpoint, data_loader):

    print("=> loading checkpoint '{}'".format(checkpoint))
    checkpoint = torch.load(checkpoint)

    best_prec1 = checkpoint['best_prec1']
    model = checkpoint['model']
    model = model.cuda()

    print("=> loaded checkpoint, best_prec1: {:.2f}".format(best_prec1))

    if config.cuda is True:
        model.cuda()
    model.eval()

    prediction = torch.zeros((1, 41)).cuda()
    with torch.no_grad():
        for input in tqdm(data_loader):

            if config.cuda:
                input = input.cuda()

            # compute output
            # print("input size:", input.size())
            output = model(input)
            output = F.softmax(output, dim=1)
            # print(output.size())
            # print(output.type())
            prediction = torch.cat((prediction, output), dim=0)

    prediction = prediction[1:].cpu().numpy()
    return prediction


def predict_one_model_with_wave(checkpoint, fold):

    print("=> loading checkpoint '{}'".format(checkpoint))
    checkpoint = torch.load(checkpoint)

    best_prec1 = checkpoint['best_prec1']
    model = checkpoint['model']
    model = model.cuda()

    print("=> loaded checkpoint, best_prec1: {:.2f}".format(best_prec1))

    test_set = pd.read_csv('../sample_submission.csv')
    test_set.set_index("fname")
    frame = test_set

    win_size = config.audio_length
    stride = int(config.sampling_rate * 0.2)

    if config.cuda is True:
        model.cuda()
    model.eval()

    prediction = torch.zeros((1, 41)).cuda()

    with torch.no_grad():

        for idx in tqdm(range(frame.shape[0])):
            filename = os.path.splitext(frame["fname"][idx])[0] + '.pkl'
            file_path = os.path.join(config.data_dir, filename)
            record_data = load_data(file_path)

            if len(record_data) < win_size:
                record_data = np.pad(record_data, (0, win_size - len(record_data)), "constant")

            wins_data = []
            for j in range(0, len(record_data) - win_size + 1, stride):
                win_data = record_data[j: j + win_size]

                maxamp = np.max(np.abs(win_data))
                if maxamp < 0.005 and j > 1:
                    continue
                wins_data.append(win_data)

            # print(file_path, len(record_data)/config.sampling_rate, len(wins_data))

            if len(wins_data) == 0:
                print(file_path)

            wins_data = np.array(wins_data)

            wins_data = wins_data[:, np.newaxis, :]

            data = torch.from_numpy(wins_data).type(torch.FloatTensor)

            if config.cuda:
                data = data.cuda()

            output = model(data)
            output = torch.sum(output, dim=0, keepdim=True)

            prediction = torch.cat((prediction, output), dim=0)

    prediction = prediction[1:]
    return prediction


def predict_one_model_with_logmel(checkpoint, fold):

    print("=> loading checkpoint '{}'".format(checkpoint))
    checkpoint = torch.load(checkpoint)

    best_prec1 = checkpoint['best_prec1']
    model = checkpoint['model']
    # model = run_method_by_string(config.arch)(pretrained=config.pretrain)
    # model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()

    print("=> loaded checkpoint, best_prec1: {:.2f}".format(best_prec1))

    test_set = pd.read_csv('../sample_submission.csv')
    test_set.set_index("fname")
    frame = test_set

    input_frame_length = int(config.audio_duration * 1000 / config.frame_shift)
    stride = 20

    if config.cuda is True:
        model.cuda()

    model.eval()

    prediction = torch.zeros((1, 41)).cuda()

    with torch.no_grad():

        for idx in tqdm(range(frame.shape[0])):
            filename = os.path.splitext(frame["fname"][idx])[0] + '.pkl'
            file_path = os.path.join(config.data_dir, filename)
            logmel = load_data(file_path)

            if logmel.shape[2] < input_frame_length:
                logmel = np.pad(logmel, ((0, 0), (0, 0), (0, input_frame_length - logmel.shape[2])), "constant")

            wins_data = []
            for j in range(0, logmel.shape[2] - input_frame_length + 1, stride):
                win_data = logmel[:, :, j: j + input_frame_length]

                # maxamp = np.max(np.abs(win_data))
                # if maxamp < 0.005 and j > 1:
                #     continue
                wins_data.append(win_data)

            # print(file_path, logmel.shape[1], input_frame_length)

            if len(wins_data) == 0:
                print(file_path)

            wins_data = np.array(wins_data)
            # wins_data = wins_data[:, np.newaxis, :, :]

            data = torch.from_numpy(wins_data).type(torch.FloatTensor)

            if config.cuda:
                data = data.cuda()

            # print("input:", data.size())
            output = model(data)
            output = torch.sum(output, dim=0, keepdim=True)

            prediction = torch.cat((prediction, output), dim=0)

    prediction = prediction[1:]
    return prediction


def predict():
    """
    Save test predictions.
    """
    for i in range(config.n_folds):
        ckp = config.model_dir + '/model_best.'+ str(i) + '.pth.tar'
        prediction = predict_one_model_with_logmel(ckp, i)
        torch.save(prediction, config.prediction_dir + '/prediction_'+str(i)+'.pt')


def txt2tensor():
    filePath = '/home/zbq/work/Kaggle/freesound-audio-tagging/prediction/lj_lcnn'
    import glob
    dataDictList = []
    for files in glob.glob(os.path.join(filePath, '*.txt')):
        f = open(files, 'r')
        dataDict = {}
        for id_, line in enumerate(f.readlines()):
            line = line.split('\n')[0].strip().split()
            data = []
            for item in line:
                data.append(float(item))
            data = np.asarray(data, dtype=np.float32)
            dataDict[id_] = data
        dataDictList.append(dataDict)
    # print("dataDictList: ", len(dataDictList))

    dataNumpy = []
    for id_ in range(9400):
        dataList = []
        for dataDict in dataDictList:
            dataList.append(dataDict[id_])
        data = np.asarray(dataList, dtype=np.float32)
        data = data.mean(axis=0)
        dataNumpy.append(data)

    dataNumpy = np.asarray(dataNumpy, dtype=np.float32)
    # print("dataNumpy: ", dataNumpy.shape)
    # print("dataSum: ", dataNumpy[0, :])
    dataTensor = torch.from_numpy(dataNumpy).type(torch.FloatTensor).cuda()
    # print("dataTensor: ", dataTensor.size())
    return dataTensor


def ensemble():
    prediction_files = []
    for i in range(config.n_folds):
        pf = '../prediction/waveResnext101_32x4d/prediction_' + str(i) + '.pt'
        prediction_files.append(pf)

    for i in range(config.n_folds):
        pf = '../prediction/mixup_logmel_delta_resnext101_32x4d/prediction_' + str(i) + '.pt'
        prediction_files.append(pf)

    # pf = '../prediction/logmel+delta/test_predictions.npy'
    # prediction_files.append(pf)

    # pf = '../prediction/logmel+delta/prediction_2.pt'
    # prediction_files.append(pf)

    pred_list = []
    for pf in prediction_files:
        pred_list.append(torch.load(pf))

    # print(txt2tensor().type())
    # print(pred_list[0].type())
    # pred_list.append(txt2tensor())


    prediction = torch.zeros_like(pred_list[0])
    # prediction = np.zeros_like(pred_list[0])
    # prediction = np.ones_like(pred_list[0])
    for pred in pred_list:
        # geometric average
        # prediction = prediction * pred
        # arithmetic average
        prediction = prediction + pred

    # prediction = prediction ** (1. / len(pred_list))
    prediction = prediction / len(pred_list)

    return prediction


def make_a_submission_file(prediction):

    test_set = pd.read_csv('../sample_submission.csv')
    result_path = './sbm.csv'
    top_3 = np.array(config.labels)[np.argsort(-prediction, axis=1)[:, :3]]
    # top_3 = np.argsort(-output, axis=1)[:, :3]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test_set['label'] = predicted_labels
    test_set.set_index("fname", inplace=True)
    test_set[['label']].to_csv(result_path)
    print('Result saved as %s' % result_path)


def make_prediction_files():

    model_dir = '../model/wave1d/'
    prediction_dir = '../prediction/wave1d'
    # make train prediction
    train = pd.read_csv('../train.csv')

    LABELS = list(train.label.unique())

    label_idx = {label: i for i, label in enumerate(LABELS)}
    train.set_index("fname")

    train["label_idx"] = train.label.apply(lambda x: label_idx[x])

    skf = StratifiedKFold(n_splits=config.n_folds)

    predictions = np.zeros((1, 41))

    for foldNum, (train_split, val_split) in enumerate(skf.split(train, train.label_idx)):
        train_set = train.iloc[train_split]
        train_set = train_set.reset_index(drop=True)
        val_set = train.iloc[val_split]
        val_set = val_set.reset_index(drop=True)
        logging.info("Fold {0}, Train samples:{1}, val samples:{2}"
              .format(foldNum, len(train_set), len(val_set)))

        # define train loader and val loader
        # valSet = Freesound_logmel(config=config, frame=val_set,
        #                      transform=transforms.Compose([ToTensor()]),
        #                      mode="test")
        #
        # val_loader = DataLoader(valSet, batch_size=config.batch_size, shuffle=False, num_workers=4)

        valSet = Freesound(config=config, frame=val_set, mode="test")

        val_loader = DataLoader(valSet, batch_size=config.batch_size, shuffle=False, num_workers=4)

        train_model = os.path.join(model_dir, 'model_best.%d.pth.tar'%foldNum)

        predictions = np.concatenate((predictions, predict_one_model(train_model, val_loader)))

    predictions = predictions[1:]
    # print(predictions, np.sum(predictions, axis=1))
    np.save(os.path.join(prediction_dir, 'train_predictions.npy'), predictions)

    # make test prediction
    test_set = pd.read_csv('../sample_submission.csv')

    # testSet = Freesound_logmel(config=config, frame=test_set,
    #                     transform=transforms.Compose([ToTensor()]),
    #                     mode="test")
    # test_loader = DataLoader(testSet, batch_size=config.batch_size, shuffle=False, num_workers=4)

    testSet = Freesound(config=config, frame=test_set, mode="test")

    test_loader = DataLoader(testSet, batch_size=config.batch_size, shuffle=False, num_workers=4)

    test_model = os.path.join(model_dir, 'model_best.%d.pth.tar'%(config.n_folds+1))

    predictions = np.zeros((1, 41))
    predictions = np.concatenate((predictions, predict_one_model(test_model, test_loader)))
    predictions = predictions[1:]
    np.save(os.path.join(prediction_dir, 'test_predictions.npy'), predictions)


def test():
    file = '../prediction/wave1d/test_predictions.npy'
    file = np.load(file)
    print(file.shape)
    # index = np.zeros((file.shape[0], 1))
    index = np.array([[i] for i in range(file.shape[0])])
    # index = np.array([0,1,2,3])
    # print(index.shape)
    file = np.hstack((index, file))
    print(file)
    print(file.shape)

    target_file = '../prediction_index/wave1d/test_predictions.npy'
    np.save(target_file, file)


if __name__ == "__main__":

    config = Config(sampling_rate=22050,
                    audio_duration=1.5,
                    batch_size=128,
                    n_folds=5,
                    data_dir="../logmel+delta_w80_s10_m64",
                    model_dir='../model/mixup_logmel_delta_se_resnext50_32x4d',
                    prediction_dir='../prediction/mixup_logmel_delta_se_resnext50_32x4d',
                    arch='se_resnext50_32x4d_',
                    lr=0.01,
                    pretrain='imagenet',
                    epochs=100)

    # config = Config(debug=False,
    #                 sampling_rate=22050,
    #                 audio_duration=2,
    #                 data_dir="../data-22050",
    #                 arch='waveResnet18',
    #                 lr=0.01,
    #                 pretrain=False,
    #                 epochs=50)

    # predict()
    prediction = ensemble()
    make_a_submission_file(prediction)

    # test()
    # make_prediction_files()
    # tensor = txt2tensor()
    # prediction = ensemble()
    # make_a_submission_file(prediction)