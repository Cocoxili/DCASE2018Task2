
from util import *
from data_loader import *
from network import *
from tqdm import tqdm

def predict_one_model(checkpoint, fold):

    print("=> loading checkpoint '{}'".format(checkpoint))
    checkpoint = torch.load(checkpoint)

    best_prec1 = checkpoint['best_prec1']
    model = checkpoint['model']
    model = model.cuda()

    print("=> loaded checkpoint, best_prec1: {:.2f}".format(best_prec1))

    test_set = pd.read_csv('../sample_submission.csv')
    # test_set.set_index("fname")

    # print(test_set)

    testSet = Freesound(config=config, frame=test_set,
                        transform=transforms.Compose([
                            ToLogMel(config),
                            ToTensor()
                        ]),
                        mode="test")
    test_loader = DataLoader(testSet, batch_size=config.batch_size, shuffle=False, num_workers=4)

    if config.cuda is True:
        model.cuda()
    model.eval()

    prediction = torch.zeros((1, 41)).cuda()
    with torch.no_grad():
        for input in tqdm(test_loader):

            if config.cuda:
                input = input.cuda()

            # compute output
            # print("input size:", input.size())
            output = model(input)
            # print(output.size())
            # print(output.type())
            prediction = torch.cat((prediction, output), dim=0)

    prediction = prediction[1:]
    return prediction


def predict():
    """
    Save test predictions.
    """
    for i in range(config.n_folds):
        ckp = '../model/model_best.' + str(i) + '.pth.tar'
        prediction = predict_one_model(ckp, i)
        torch.save(prediction, '../prediction/prediction_'+str(i)+'.pt')



def ensemble():
    prediction_files = []
    for i in range(config.n_folds):
        pf = '../prediction/prediction_' + str(i) + '.pt'
        prediction_files.append(pf)

    pred_list = []
    for pf in prediction_files:
        pred_list.append(torch.load(pf))

    # prediction = np.ones_like(pred_list[0])
    prediction = torch.ones_like(pred_list[0]).cuda()
    for pred in pred_list:
        prediction = prediction * pred
    prediction = prediction ** (1. / len(pred_list))

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

# def make_a_submission_file1(checkpoint, fold):
#
#     print("=> loading checkpoint '{}'".format(checkpoint))
#     checkpoint = torch.load(checkpoint)
#
#     best_prec1 = checkpoint['best_prec1']
#     model = checkpoint['model']
#     model = model.cuda()
#
#     print("=> loaded checkpoint, best_prec1: {:.2f}".format(best_prec1))
#
#
#     test_set = pd.read_csv('../sample_submission.csv')
#     # test_set.set_index("fname")
#
#     # print(test_set)
#
#     testSet = Freesound(config=config, frame=test_set,
#                        transform=transforms.Compose([
#                            ToLogMel(config),
#                            ToTensor()
#                        ]),
#                        mode="test")
#     test_loader = DataLoader(testSet, batch_size=config.batch_size, shuffle=False, num_workers=4)
#
#     if config.cuda is True:
#         model.cuda()
#     model.eval()
#
#     predictions = torch.zeros((1, 41)).cuda()
#     with torch.no_grad():
#         for input in tqdm(test_loader):
#
#             if config.cuda:
#                 input = input.cuda()
#
#             # compute output
#             # print("input size:", input.size())
#             output = model(input)
#             # print(output.size())
#             # print(output.type())
#             predictions = torch.cat((predictions, output), dim=0)
#
#     predictions = predictions[1:]
#     print(predictions.size())
#     top_3 = np.array(config.labels)[np.argsort(-predictions, axis=1)[:, :3]]
#     # top_3 = np.argsort(-output, axis=1)[:, :3]
#     print(top_3)
#     predicted_labels = [' '.join(list(x)) for x in top_3]
#     test_set['label'] = predicted_labels
#     test_set.set_index("fname", inplace=True)
#     test_set[['label']].to_csv('./' + "predictions_%d.csv" % fold)
#
#     # # Save test predictions
#     # test_generator = DataGenerator(config, '../audio_test/', test.index, batch_size=128,
#     #                                preprocessing_fn=audio_norm)
#     # predictions = model.predict_generator(test_generator, use_multiprocessing=True,
#     #                                       workers=6, max_queue_size=20, verbose=1)
#     # np.save(PREDICTION_FOLDER + "/test_predictions_%d.npy" % i, predictions)
#     #
#     # # Make a submission file
#     # top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
#     # predicted_labels = [' '.join(list(x)) for x in top_3]
#     # test['label'] = predicted_labels
#     # test[['label']].to_csv(PREDICTION_FOLDER + "/predictions_%d.csv" % i)
#

if __name__ == "__main__":
    config = Config()
    predict()
    prediction = ensemble()
    make_a_submission_file(prediction)