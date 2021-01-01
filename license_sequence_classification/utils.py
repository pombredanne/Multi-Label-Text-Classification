import os
import datetime
import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset


class LicenseDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def read_corpus(data_path):
    df = pd.read_csv(data_path, header=0, encoding='unicode_escape')
    data_texts = preprocessing(df.values[:, 1])
    data_labels = df.values[:, 2]

    return data_texts.tolist(), data_labels.tolist()


def preprocessing(data_texts):
    for i in range(len(data_texts)):
        data_texts[i] = filter_sentence(data_texts[i].lower().strip())
    return data_texts


def filter_sentence(text):
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("'", "")
    text = text.replace("\"", "")
    text = text.replace(",", "")
    text = text.replace(".", "")
    text = text.replace(":", "")
    text = text.replace(";", "")
    return text


def read_prediction_data(data_path):
    df = pd.read_csv(data_path, header=0, encoding='unicode_escape')
    data_texts = preprocessing(df.values[:, 1])

    return data_texts.tolist()


def save_prediction_result(pred, file_path, save_path):
    df = pd.read_csv(file_path, header=0, encoding='unicode_escape')
    df.insert(df.shape[1], 'is_granted', pred)
    df.to_csv(save_path, index=False, sep=',')


def save_model(model, epoch, config, path='result', saved_models=list(), **kwargs):
    if not os.path.exists(path):
        os.mkdir(path)
    if kwargs.get('name', None) is None:
        cur_time = datetime.datetime.now().strftime('%Y-%m-%d#%H:%M:%S')
        name = cur_time + '--epoch:{}'.format(epoch)
        full_name = os.path.join(path, name)
        torch.save(model.state_dict(), full_name)
        print('Saved model at epoch {} successfully'.format(epoch))

        saved_models.append(full_name)
        if len(saved_models) > 5:
            os.remove(saved_models.pop(0))

        with open(os.path.join(path, 'checkpoint'), 'w') as file:
            file.write(name)
            print('Write to checkpoint')

        with open(os.path.join(path, 'config.pkl'), 'wb') as file:
            pickle.dump(config, file)
            print('Write to config')


def load_model(model, path='result', **kwargs):
    if kwargs.get('name', None) is None:
        with open(os.path.join(path, 'checkpoint')) as file:
            content = file.read().strip()
            name = os.path.join(path, content)
    else:
        name = kwargs['name']
        name = os.path.join(path, name)
    model.load_state_dict(torch.load(name, map_location=lambda storage, loc: storage))
    print('load model {} successfully'.format(name))
    return model

