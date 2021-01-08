from transformers import BertTokenizer, BertForSequenceClassification, AlbertTokenizer, AlbertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import *
from torch.utils.data import DataLoader
from config import Config
import numpy as np
import torch
import fire


def train(**kwargs):
    config = Config()
    config.update(**kwargs)
    if config.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)
    else:
        config.device = 'cpu'

    # train:valid:test = 6:2:2
    if not config.random_seed:
        config.random_seed = np.random.randint(1e3)

    print('current config:\n', config)

    data_texts, data_labels = read_corpus(config.train_file)
    train_texts, val_texts, train_labels, val_labels = train_test_split(data_texts, data_labels, test_size=.25,
                                                                        random_state=config.random_seed)

    print('train text num: ', len(train_texts))
    print('val text num: ', len(val_texts))

    if config.bert_model == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif config.bert_model == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
    else:
        raise Exception('Error Bert model.')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = LicenseDataset(train_encodings, train_labels)
    val_dataset = LicenseDataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    if config.bert_model == 'bert':
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    elif config.bert_model == 'albert':
        model = AlbertForSequenceClassification.from_pretrained('albert-base-v1')
    else:
        raise Exception('Error Bert model.')

    for name, param in model.named_parameters():
        param.requires_grad = True

    model.to(config.device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=config.lr)

    best_valid_loss = 10000
    best_epoch = 0
    dist_to_best = 0
    saved_models = []

    print('start training')
    for epoch in range(config.epoch_num):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()

        print('epoch: {}|  loss: {}'.format(epoch, loss.item()))

        # validation
        model.eval()
        loss_val = val(model, val_loader, epoch, config)
        model.train()

        # early stop
        if loss_val < best_valid_loss:
            save_model(model, epoch, config=config, path=config.model_save_path, saved_models=saved_models)
            best_epoch = epoch
            best_valid_loss = loss_val
            dist_to_best = 0

        dist_to_best += 1
        if dist_to_best > 30:
            break

    print("best epoch: ", best_epoch)
    print("best valid loss: ", best_valid_loss)


def val(model, dev_loader, epoch, config):
    eval_loss = 0
    true = []
    pred = []
    length = 0
    for i, batch in enumerate(dev_loader):
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        labels = batch['labels'].to(config.device)
        length += input_ids.size(0)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        logits = outputs[1]
        eval_loss += loss.item()
        pred.extend(torch.argmax(logits, dim=1).tolist())
        true.extend(labels.tolist())

    accuracy = accuracy_score(pred, true)
    print('eval epoch: {}| loss: {}| overall accuracy: {}'.format(epoch, eval_loss/length, accuracy))
    return eval_loss/length


def test(load_path, **kwargs):

    with open(os.path.join(load_path, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)

    config.update(**kwargs)
    config.load_path = load_path

    if config.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)
    else:
        config.device = 'cpu'

    print('current config:\n', config)

    test_texts, test_labels = read_corpus(config.test_file)

    print('test text num: ', len(test_texts))

    # Load Bert Tokenizer
    if config.bert_model == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif config.bert_model == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
    else:
        raise Exception('Error Bert model.')

    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    test_dataset = LicenseDataset(test_encodings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    if config.bert_model == 'bert':
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    elif config.bert_model == 'albert':
        model = AlbertForSequenceClassification.from_pretrained('albert-base-v1')
    else:
        raise Exception('Error Bert model.')

    if config.load_path:
        model = load_model(model, path=config.load_path, name=config.ckpt_name)
    model.to(config.device)
    model.eval()

    eval_loss = 0
    true = []
    pred = []
    length = 0
    for i, batch in enumerate(test_loader):
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        labels = batch['labels'].to(config.device)
        length += input_ids.size(0)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        logits = outputs[1]
        eval_loss += loss.item()
        pred.extend(torch.argmax(logits, dim=1).tolist())
        true.extend(labels.tolist())

    accuracy = accuracy_score(y_pred=pred, y_true=true)
    precision = precision_score(y_pred=pred, y_true=true)
    recall = recall_score(y_pred=pred, y_true=true)
    f1 = f1_score(y_pred=pred, y_true=true)
    print('loss: {}| overall accuracy: {}'.format(eval_loss/length, accuracy))
    print('precision: {}| recall: {}| f1: {}'.format(precision, recall, f1))


def predict(load_path, file_path='./data/Task2.predict.csv', save_path='./data/Task2.predict.result.csv', **kwargs):
    with open(os.path.join(load_path, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)

    config.update(**kwargs)
    config.load_path = load_path

    if config.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)
    else:
        config.device = 'cpu'

    data_texts = read_prediction_data(file_path)

    if len(data_texts) == 0:
        save_prediction_result([], file_path, save_path)
        return

    # Load Bert Tokenizer
    if config.bert_model == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif config.bert_model == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
    else:
        raise Exception('Error Bert model.')

    data_encodings = tokenizer(data_texts, truncation=True, padding=True)
    data_dataset = LicenseDataset(data_encodings)
    data_loader = DataLoader(data_dataset, batch_size=config.batch_size, shuffle=False)

    if config.bert_model == 'bert':
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    elif config.bert_model == 'albert':
        model = AlbertForSequenceClassification.from_pretrained('albert-base-v1')
    else:
        raise Exception('Error Bert model.')

    if config.load_path:
        model = load_model(model, path=config.load_path, name=config.ckpt_name)
    model.to(config.device)
    model.eval()

    pred = []
    for i, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        pred.extend(torch.argmax(logits, dim=1).tolist())

    save_prediction_result(pred, file_path, save_path)


if __name__ == '__main__':
    fire.Fire()
