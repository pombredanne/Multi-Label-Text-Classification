import torch


class Config(object):
    def __init__(self):

        self.model = 'LSAN'
        self.train_file_path_task = './data/Task1.train.12t.csv'
        self.test_file_path_task = './data/Task1.test.12t.csv'
        self.save_path = './saved_models/'
        self.random_seed = None

        if self.model == 'LSAN':
            self.label_embedding_path = './cache/label_embeddings.12t.npy'
            self.max_seq_length = 256
            self.embedding_size = 300
            self.label_embedding_size = 768
            self.lstm_hidden_dimension = 768
            self.d_a = 200

        elif self.model == 'LSAN-BERT':
            self.bert_model = 'albert'
            self.max_seq_length = 512
            self.bert_embedding_size = 768
            self.lstm_hidden_dimension = 256
            self.d_a = 128

        elif self.model == 'BERT':
            self.bert_model = 'albert'
            self.max_seq_length = 512
            self.hidden_size = 768
            self.hidden_dropout_prob = 0.2

        self.loss = 'FocalLoss'
        self.focal_loss_reduction = 'mean'
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2

        # hyper-parameters
        self.num_epochs = 300
        self.batch_size = 16
        self.gradient_accumulation_steps = 4
        self.learning_rate = 0.0001
        self.betas = (0.9, 0.99)
        self.weight_decay = 0.00005

        self.num_of_class = 12
        self.threshold = 0.5

        self.early_stop_criterion = 'loss'
        self.early_stop_patience = 30

        # device configurations
        self.use_cuda = True
        self.gpu = 2
        if self.use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_device(self.gpu)
        else:
            self.device = torch.device('cpu')

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


config = Config()

