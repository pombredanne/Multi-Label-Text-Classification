import os
import torch
import torch.nn.functional as F

from config import *


# reference to:
# https://github.com/EMNLP2019LSAN/LSAN
class BasicModule(torch.nn.Module):

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))
        self.saved_models = []

    def load_model(self, path, name=None):
        if name is None:
            with open(os.path.join(path, 'checkpoint')) as file:
                content = file.read().strip()
                full_path = os.path.join(path, content)
        else:
            full_path = os.path.join(path, name)
        self.load_state_dict(torch.load(full_path))
        print('load model {} successfully'.format(name))

    def save_model(self, epoch, path=None):
        if path is None:
            raise ValueError('Please specify the saving path.')
        if not os.path.exists(path):
            os.mkdir(path)
        model_name = 'LSAN--epoch:{}'.format(epoch)
        full_path = os.path.join(path, model_name)
        torch.save(self.state_dict(), full_path)
        print('Model saved at epoch: ' + str(epoch))

        self.saved_models.append(full_path)
        if len(self.saved_models) > 5:
            os.remove(self.saved_models.pop(0))

        with open(os.path.join(path, 'checkpoint'), 'w') as file:
            file.write(model_name)
            print('Write to checkpoint')


class StructuredSelfAttention(BasicModule):

    def __init__(self, batch_size, lstm_hid_dim, d_a, n_classes, word_embedding, label_embedding):
        super(StructuredSelfAttention, self).__init__()
        self.n_classes = n_classes
        self.word_embeddings = self.load_embeddings(word_embedding)
        self.label_embeddings = self.load_embeddings(label_embedding)
        self.lstm = torch.nn.LSTM(input_size=embedding_size, hidden_size=lstm_hid_dim, num_layers=1,
                                  batch_first=True, bidirectional=True)
        self.linear_first = torch.nn.Linear(lstm_hid_dim * 2, d_a)
        self.linear_second = torch.nn.Linear(d_a, n_classes)

        self.weight1 = torch.nn.Linear(lstm_hid_dim * 2, 1)
        self.weight2 = torch.nn.Linear(lstm_hid_dim * 2, 1)

        self.output_layer = torch.nn.Linear(lstm_hid_dim * 2, n_classes)
        self.embedding_dropout = torch.nn.Dropout(p=0.1)
        self.batch_size = batch_size
        self.lstm_hid_dim = lstm_hid_dim

    def load_embeddings(self, word_embedding):
        """Load the embeddings based on flag"""
        word_embeddings = torch.nn.Embedding(word_embedding.size(0), word_embedding.size(1))
        word_embeddings.weight = torch.nn.Parameter(word_embedding)
        return word_embeddings

    def init_hidden(self, batch_size):
        h_0 = torch.randn(2, batch_size, self.lstm_hid_dim)
        c_0 = torch.randn(2, batch_size, self.lstm_hid_dim)
        return h_0, c_0

    def forward(self, x):
        word_embeddings = self.word_embeddings(x)
        word_embeddings = self.embedding_dropout(word_embeddings)
        batch_size = x.size(0)
        # step1 get LSTM outputs
        h_0, c_0 = self.init_hidden(batch_size)
        if word_embeddings.is_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        outputs, _ = self.lstm(word_embeddings, (h_0, c_0))
        # step2 get self-attention
        selfatt = torch.tanh(self.linear_first(outputs))
        selfatt = self.linear_second(selfatt)
        selfatt = F.softmax(selfatt, dim=1)
        selfatt = selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, outputs)
        # step3 get label-attention
        h1 = outputs[:, :, :self.lstm_hid_dim]
        h2 = outputs[:, :, self.lstm_hid_dim:]

        label = self.label_embeddings.weight.data
        m1 = torch.bmm(label.expand(batch_size, self.n_classes, self.lstm_hid_dim), h1.transpose(1, 2))
        m2 = torch.bmm(label.expand(batch_size, self.n_classes, self.lstm_hid_dim), h2.transpose(1, 2))
        label_att = torch.cat((torch.bmm(m1, h1), torch.bmm(m2, h2)), 2)
        # label_att = F.normalize(label_att, p=2, dim=-1)
        # self_att = F.normalize(self_att, p=2, dim=-1) #all can
        weight1 = torch.sigmoid(self.weight1(label_att))
        weight2 = torch.sigmoid(self.weight2(self_att))
        weight1 = weight1 / (weight1 + weight2)
        weight2 = 1 - weight1

        doc = weight1 * label_att + weight2 * self_att
        avg_sentence_embeddings = torch.sum(doc, 1) / self.n_classes

        pred = torch.sigmoid(self.output_layer(avg_sentence_embeddings))
        return pred
