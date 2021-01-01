import torch


class Config(object):
    def __init__(self):

        self.TRAIN_FILE_PATH_TASK = '../data/Task1.train.csv'
        self.TEST_FILE_PATH_TASK = '../data/Task1.test.csv'
        self.SAVE_PATH = './saved_models/'
        self.MAX_SEQ_LENGTH = 512
        self.RANDOM_SEED = None
        self.BERT_MODEL = 'albert'

        # hyper-parameters
        self.NUM_EPOCHS = 300
        self.BATCH_SIZE = 16
        self.GRADIENT_ACCUMULATION_STEPS = 4
        self.LEARNING_RATE = 0.0001
        self.BETAS = (0.9, 0.99)
        self.WEIGHT_DECAY = 0.00005

        # model-parameters
        self.bert_embedding_size = 768
        self.label_embedding_size = 768
        self.lstm_hidden_dimension = 256
        self.d_a = 128
        self.num_of_class = 13
        self.threshold = 0.5

        # device configurations
        self.use_cuda = True
        self.gpu = 2
        if self.use_cuda and torch.cuda.is_available():
            self.DEVICE = torch.device('cuda')
            torch.cuda.set_device(self.gpu)
        else:
            self.DEVICE = torch.device('cpu')

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


config = Config()

