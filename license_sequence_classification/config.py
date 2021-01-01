import os

root_path = os.path.dirname(os.path.abspath(__file__))


class Config(object):
    def __init__(self):
        self.device = 'cuda'
        self.gpu = 0
        self.train_file = os.path.join(root_path, 'data', 'Task2.train.csv')
        self.test_file = os.path.join(root_path, 'data', 'Task2.test.csv')

        self.model_save_path = os.path.join(root_path, 'saved_models')
        self.load_path = None
        self.ckpt_name = None

        self.bert_model = 'albert'
        self.batch_size = 16
        self.lr = 5e-5
        self.epoch_num = 300

        self.random_seed = None

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])
