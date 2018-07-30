
from util import *
from data_loader import *
from network import *
from tqdm import tqdm
import torch.nn.functional as F


def predict_one_model_with_wave(checkpoint, frame):

    print("=> loading checkpoint '{}'".format(checkpoint))
    checkpoint = torch.load(checkpoint)

    best_prec1 = checkpoint['best_prec1']
    # model = checkpoint['model']
    model = run_method_by_string(config.arch)(pretrained=config.pretrain)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()

    print("=> loaded checkpoint, best_prec1: {:.2f}".format(best_prec1))

    win_size = config.audio_length
    stride = int(config.sampling_rate * 0.2)

    if config.cuda is True:
        model.cuda()
    model.eval()

    file_names = []
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

            output = F.softmax(output, dim=1)

            prediction = torch.cat((prediction, output), dim=0)

            file_names.append(frame["fname"][idx])

    prediction = prediction[1:]

    return file_names, prediction


def predict_one_model_with_logmel(checkpoint, frame):

    print("=> loading checkpoint '{}'".format(checkpoint))
    checkpoint = torch.load(checkpoint)

    best_prec1 = checkpoint['best_prec1']
    #  model = checkpoint['model']
    model = run_method_by_string(config.arch)(pretrained=config.pretrain)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()

    print("=> loaded checkpoint, best_prec1: {:.2f}".format(best_prec1))

    input_frame_length = int(config.audio_duration * 1000 / config.frame_shift)
    stride = 20

    if config.cuda is True:
        model.cuda()

    model.eval()

    prediction = torch.zeros((1, 41)).cuda()

    file_names = []
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

            file_names.append(frame["fname"][idx])
    
    prediction = prediction[1:]
    return file_names, prediction


def predict():
    """
    Save test predictions.
    """

    for i in range(config.n_folds):
       ckp = config.model_dir + '/model_best.'+ str(i) + '.pth.tar'
       # prediction = predict_one_model_with_logmel(ckp, i)
       prediction = predict_one_model_with_wave(ckp)
       torch.save(prediction, config.prediction_dir + '/prediction_'+str(i)+'.pt')

    #  ckp = config.model_dir + '/model_best.6.pth.tar'
    #  prediction = predict_one_model_with_logmel(ckp, i)
    #  prediction = predict_one_model_with_wave(ckp)
    #  torch.save(prediction, config.prediction_dir + '/prediction_5.pt')


def ensemble():
    prediction_files = []
    # for i in range(config.n_folds):
    #     pf = '../prediction/mixup_mfcc_delta/prediction_' + str(i) + '.pt'
    #     prediction_files.append(pf)

    # for i in range(config.n_folds):
    #     pf = '../prediction/waveResnext101_32x4d/prediction_' + str(i) + '.pt'
    #     prediction_files.append(pf)

    pf = '../prediction/waveResnext101_32x4d/prediction_5.pt'
    prediction_files.append(pf)

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


def make_a_submission_file():
    
    prediction = pd.read_csv(os.path.join(config.prediction_dir, 'test_predictions.csv'), header=None)
    prediction = prediction[prediction.columns[1:]].values
    test_set = pd.read_csv('../input/sample_submission.csv')
    result_path = os.path.join(config.prediction_dir, 'sbm.csv')
    top_3 = np.array(config.labels)[np.argsort(-prediction, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test_set['label'] = predicted_labels
    test_set.set_index("fname", inplace=True)
    test_set[['label']].to_csv(result_path)
    print('Result saved as %s' % result_path)



def make_prediction_files(input, mean_method='arithmetic'):
    """
    make two prediction files for stacking. One for train and one for test. 
    Prediction matrix of samples x classes (9400 x 41)
    
    :param input: 'wave' or 'logmel'
    :param mean_method: 'arithmetic' or 'geometric'
    """

    model_dir = config.model_dir

    # make train prediction

    train = pd.read_csv('../input/train.csv')

    # train = train[:100] # for debug

    LABELS = list(train.label.unique())

    label_idx = {label: i for i, label in enumerate(LABELS)}
    train.set_index("fname")

    train["label_idx"] = train.label.apply(lambda x: label_idx[x])

    skf = StratifiedKFold(n_splits=config.n_folds)

    predictions = np.zeros((1, 41))
    file_names = []
    for foldNum, (train_split, val_split) in enumerate(skf.split(train, train.label_idx)):
        val_set = train.iloc[val_split]
        val_set = val_set.reset_index(drop=True)
        print("Fold {0}, Val samples:{1}"
              .format(foldNum, len(val_set)))

        ckp = os.path.join(model_dir, 'model_best.%d.pth.tar'%foldNum)

        if input == 'wave':
            fn, pred = predict_one_model_with_wave(ckp, val_set)

        elif input == 'logmel':
            fn, pred = predict_one_model_with_logmel(ckp, val_set)

        file_names.extend(fn)

        predictions = np.concatenate((predictions, pred.cpu().numpy()))

    predictions = predictions[1:]
    save_to_csv(file_names, predictions, 'train_predictions.csv')

    #  make test prediction
    test_set = pd.read_csv('../input/sample_submission.csv')

    #  test_set = test_set[:50] # for debug

    test_set.set_index("fname")
    frame = test_set

    pred_list = []

    for i in range(config.n_folds):
        ckp = config.model_dir + '/model_best.'+ str(i) + '.pth.tar'
        if input == 'wave':
            fn, pred = predict_one_model_with_wave(ckp, frame)
        elif input == 'logmel':
            fn, pred = predict_one_model_with_logmel(ckp, frame)

        pred = pred.cpu().numpy()
        pred_list.append(pred)

    if mean_method == 'arithmetic':
        predictions = np.zeros_like(pred_list[0])
        for pred in pred_list:
            predictions = predictions + pred
        predictions = predictions / len(pred_list)

        print(predictions.shape)
    elif mean_method == 'geometric':
        predictions = np.ones_like(pred_list[0])
        for pred in pred_list:
            predictions = predictions * pred
        predictions = predictions ** (1. / len(pred_list))
    else:
        print('mean_method not specified.')

    save_to_csv(fn, predictions, 'test_predictions.csv')


def save_to_csv(files_name, prediction, file):
    df = pd.DataFrame(index=files_name, data=prediction)
    path = os.path.join(config.prediction_dir, file)
    df.to_csv(path, header=None)


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

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    config = Config(sampling_rate=22050,
                   audio_duration=1.5,
                   batch_size=128,
                   n_folds=5,
                   data_dir="../logmel+delta_w80_s10_m64",
                   model_dir='../model/logmel_delta_resnet101',
                   prediction_dir='../prediction/logmel_delta_resnet101',
                   arch='resnet101_mfcc',
                   lr=0.01,
                   pretrain=None,
                   mixup=False,
                   epochs=100)

    #  config = Config(debug=False,
                    #  n_folds=5,
                    #  sampling_rate=44100,
                    #  audio_duration=1.5,
                    #  batch_size=16,
                    #  data_dir="../data-44100",
                    #  arch='waveResnext101_32x4d',
                    #  model_dir='../model/waveResnext101_32x4d_nopretrained',
                    #  prediction_dir='../prediction/waveResnext101_32x4d_nopretrained',
                    #  lr=0.01,
                    #  pretrain=None,
                    #  print_freq=60,
                    #  epochs=50)


    # test()
    make_prediction_files(input='logmel', mean_method='arithmetic')
    #  make_prediction_files(input='wave', mean_method='arithmetic')
    #  predict()
    #  prediction = ensemble()
    make_a_submission_file()
