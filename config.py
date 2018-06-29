import numpy as np

class Config(object):
    def __init__(self,
                 sampling_rate=22050, audio_duration=1.5, n_classes=41,
                 train_dir='../audio_train', test_dir='../audio_test',
                 data_dir='../data-22050',
                 model_dir='../model',
                 prediction_dir='../prediction',
                 arch='resnet101_m', pretrain=False,
                 cuda=True, print_freq=10, epochs=50,
                 batch_size=64,
                 momentum=0.9, weight_decay=0.0005,
                 n_folds=5, lr=0.01,
                 n_mels=64, frame_weigth=40, frame_shift=10,
                 debug=False):

        self.labels = ['Hi-hat', 'Saxophone', 'Trumpet', 'Glockenspiel', 'Cello', 'Knock',
                        'Gunshot_or_gunfire', 'Clarinet', 'Computer_keyboard', 'Keys_jangling',
                        'Snare_drum', 'Writing', 'Laughter', 'Tearing', 'Fart', 'Oboe', 'Flute',
                        'Cough', 'Telephone', 'Bark', 'Chime', 'Bass_drum', 'Bus', 'Squeak',
                        'Scissors', 'Harmonica', 'Gong', 'Microwave_oven', 'Burping_or_eructation',
                        'Double_bass', 'Shatter', 'Fireworks', 'Tambourine', 'Cowbell',
                        'Electric_piano', 'Meow', 'Drawer_open_or_close', 'Applause', 'Acoustic_guitar',
                        'Violin_or_fiddle', 'Finger_snapping']

        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.audio_length = int(self.sampling_rate * self.audio_duration)
        self.n_classes = n_classes
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.prediction_dir = prediction_dir
        self.arch = arch
        self.pretrain = pretrain
        self.cuda = cuda
        self.print_freq = print_freq
        self.epochs = epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.n_folds = n_folds
        self.lr = lr

        self.n_fft = int(frame_weigth / 1000 * sampling_rate)
        self.n_mels = n_mels
        self.frame_weigth = frame_weigth
        self.frame_shift = frame_shift
        self.hop_length = int(frame_shift / 1000 * sampling_rate)

        self.debug = debug


if __name__ == "__main__":
    config = Config()
