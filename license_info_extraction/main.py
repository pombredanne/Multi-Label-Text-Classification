import numpy as np
import torch.nn as nn
from transformers import BertTokenizer, AlbertTokenizer
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm, trange
from torchtext import vocab
from collections import defaultdict

from DataModule import *
from LSANModule import StructuredSelfAttention, StructuredSelfAttentionBert, FocalLoss, BertLinear
from config import *
from utils import *
from label_embedding import label_definition_list
from sklearn.model_selection import train_test_split
import fire
import pandas as pd


def load_dataset(task_file_path, encoder):
    # Load task dataset
    if config.model == 'LSAN':
        train_dataset = SequenceDataset(task_file_path, encoder)
    elif config.model == 'LSAN-BERT' or 'BERT':
        train_dataset = SequenceDatasetBert(task_file_path, encoder)

    # Split task dataset into Train and Validation set, 6:2:2
    if not config.random_seed:
        config.random_seed = np.random.randint(1e3)

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    train_indices, val_indices, _, _ = train_test_split(indices, indices, test_size=.25,
                                                        random_state=config.random_seed)

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, sampler=validation_sampler)
    return train_loader, val_loader, len(train_indices), len(val_indices)


def load_label_description(tokenizer):
    text = []
    for label_definition in label_definition_list:
        text.append(label_definition[-1])

    assert len(text) == config.num_of_class

    label_encodings = tokenizer(text, truncation=True, padding=True)
    return label_encodings


def save_prediction_result(pred_label_dict, file_path, save_path):
    df = pd.read_csv(file_path, header=0, encoding='unicode_escape')
    if df.shape[1] > 2:
        df = df.take([0, 1], axis=1)

    for i in range(config.num_of_class):
        label_name = label_definition_list[i][0]
        predict_result = pred_label_dict[i + 1]
        df.insert(df.shape[1], label_name, predict_result)

    df.to_csv(save_path, index=False, sep=',')
    print('save successfully: ', save_path)


def train():
    if config.model == 'LSAN':
        # Load Glove Embedding
        glove_embeddings = vocab.GloVe(name='6B', dim=300, cache='./cache')
        # Load Label Embedding
        sbert_embeddings = np.load(config.label_embedding_path)
        # Load Dataset
        train_loader, val_loader, train_size, val_size = load_dataset(config.train_file_path_task, glove_embeddings)

        # Initialize LSAN Model
        word_embedding = glove_embeddings.vectors.float()
        label_embedding = torch.from_numpy(sbert_embeddings).float()  # self-constructed label embedding
        # reference to: https://www.sbert.net/
        model = StructuredSelfAttention(batch_size=config.batch_size,
                                        lstm_hid_dim=config.lstm_hidden_dimension,
                                        d_a=config.d_a,
                                        n_classes=config.num_of_class,
                                        word_embedding=word_embedding,
                                        label_embedding=label_embedding)

    elif config.model == 'LSAN-BERT':
        # Load Bert Tokenizer
        if config.bert_model == 'bert':
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif config.bert_model == 'albert':
            bert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
        else:
            raise Exception('Error Bert model.')

        # Load Dataset
        train_loader, val_loader, train_size, val_size = load_dataset(config.train_file_path_task, bert_tokenizer)
        label_encodings = load_label_description(bert_tokenizer)
        model = StructuredSelfAttentionBert(batch_size=config.batch_size,
                                            lstm_hid_dim=config.lstm_hidden_dimension,
                                            d_a=config.d_a,
                                            n_classes=config.num_of_class,
                                            is_train=True,
                                            label_input_ids=torch.tensor(label_encodings['input_ids']).to(
                                                config.device),
                                            label_attention_masks=torch.tensor(label_encodings['attention_mask']).to(
                                                config.device))

    elif config.model == 'BERT':
        # Load Bert Tokenizer
        if config.bert_model == 'bert':
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif config.bert_model == 'albert':
            bert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
        else:
            raise Exception('Error Bert model.')

        # Load Dataset
        train_loader, val_loader, train_size, val_size = load_dataset(config.train_file_path_task, bert_tokenizer)
        model = BertLinear(hidden_size=config.hidden_size,
                           hidden_dropout_prob=config.hidden_dropout_prob,
                           n_classes=config.num_of_class,
                           is_train=True)

    print(config)
    print('Training Set Size {}, Validation Set Size {}'.format(train_size, val_size))

    if config.use_cuda:
        model.cuda()

    if config.loss == 'BCELoss':
        criterion = torch.nn.BCELoss()
    elif config.loss == 'FocalLoss':
        criterion = FocalLoss(alpha=config.focal_loss_alpha, gamma=config.focal_loss_gamma,
                              reduction=config.focal_loss_reduction)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=config.weight_decay)

    # Initialize Early Stop Configurations
    counter = 0
    best_epoch = 0
    best_epoch_f1 = 0
    minimum_val_loss = 1000
    max_val_f1 = 0

    # epoch loop
    epoch_iterator = trange(int(config.num_epochs), desc="Epoch")
    for epoch in epoch_iterator:
        # training loop
        train_loss = []
        true_label_dict, pred_label_dict = defaultdict(list), defaultdict(list)
        accuracy_score_list = []
        precision_score_list = []
        recall_score_list = []
        f1_score_list = []
        train_iterator = tqdm(train_loader, desc="Train Iteration")
        model.train()
        optimizer.zero_grad()
        for batch_idx, train_batch in enumerate(train_iterator):
            if config.model == 'LSAN':
                x, y_true = train_batch[0], train_batch[1]
                if config.use_cuda:
                    x = x.cuda()
                    y_true = y_true.cuda()
                y_pred = model(x)

            elif config.model == 'LSAN-BERT' or 'BERT':
                input_ids, token_type_ids, attention_masks, y_true = train_batch
                if config.use_cuda:
                    input_ids = input_ids.cuda()
                    token_type_ids = token_type_ids.cuda()
                    attention_masks = attention_masks.cuda()
                    y_true = y_true.cuda()
                y_pred = model(input_ids, token_type_ids, attention_masks)

            loss = criterion(y_pred, y_true)
            loss.backward()
            # update model-parameters
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            # evaluation metrics: hamming loss, accuracy score, precision, recall, f1
            labels_cpu = y_true.data.cpu()
            pred_cpu = y_pred.data.cpu()

            train_loss.append(float(loss.item()))
            true_labels, pred_labels = make_label_all_class(labels_cpu, pred_cpu, config.threshold)
            for i in range(0, config.num_of_class):
                make_label_each_class(true_labels, pred_labels, i, true_label_dict[i+1], pred_label_dict[i+1])

        avg_train_loss = np.mean(train_loss)
        for i in range(0, config.num_of_class):
            accuracy_score, precision_score, recall_score, f1_score = get_metrics_each_class(
                                                                      true_label_dict[i + 1], pred_label_dict[i + 1])
            accuracy_score_list.append(accuracy_score)
            precision_score_list.append(precision_score)
            recall_score_list.append(recall_score)
            f1_score_list.append(f1_score)
        avg_accuracy_score = np.mean(accuracy_score_list)
        macro_precision_score = np.mean(precision_score_list)
        macro_recall_score = np.mean(recall_score_list)
        macro_f1_score = np.mean(f1_score_list)
        print("epoch %2d train end : avg_train_loss = %.4f" % (epoch + 1, avg_train_loss))
        print("epoch %2d train end : avg_accuracy_score = %.4f" % (epoch + 1, avg_accuracy_score))
        print("epoch %2d train end : macro_precision_score = %.4f" % (epoch + 1, macro_precision_score))
        print("epoch %2d train end : macro_recall_score = %.4f" % (epoch + 1, macro_recall_score))
        print("epoch %2d train end : macro_f1_score = %.4f" % (epoch + 1, macro_f1_score))
        # validation loop
        avg_val_loss, avg_accuracy_score, macro_precision_score, macro_recall_score, macro_f1_score \
            = val(model, val_loader, criterion)
        print("epoch %2d valid end : avg_val_loss = %.4f" % (epoch + 1, avg_val_loss))
        print("epoch %2d valid end : avg_accuracy_score = %.4f" % (epoch + 1, avg_accuracy_score))
        print("epoch %2d valid end : macro_precision_score = %.4f" % (epoch + 1, macro_precision_score))
        print("epoch %2d valid end : macro_recall_score = %.4f" % (epoch + 1, macro_recall_score))
        print("epoch %2d valid end : macro_f1_score = %.4f" % (epoch + 1, macro_f1_score))

        # check early stopping status
        if config.early_stop_criterion == 'loss':
            if avg_val_loss < minimum_val_loss:
                minimum_val_loss = avg_val_loss
                model.save_model(epoch=epoch + 1, path=config.save_path)
                best_epoch = epoch + 1
                counter = 0
            else:
                if macro_f1_score > max_val_f1:
                    max_val_f1 = macro_f1_score
                    model.save_model(epoch=epoch + 1, path=config.save_path)
                    best_epoch_f1 = epoch + 1

                counter = counter + 1

        elif config.early_stop_criterion == 'macro-f1':
            if macro_f1_score > max_val_f1:
                max_val_f1 = macro_f1_score
                model.save_model(epoch=epoch + 1, path=config.save_path)
                best_epoch_f1 = epoch + 1
                counter = 0
            else:
                counter = counter + 1

        if counter >= config.early_stop_patience:
            print("Early stopping at epoch: " + str(epoch+1))
            print("Best epoch: ", best_epoch)
            print("Best f1 epoch: ", best_epoch_f1)

            break
    print('Training and Validation Stage Done...')


def val(model, val_loader, criterion):
    with torch.no_grad():
        val_loss = []
        true_label_dict, pred_label_dict = defaultdict(list), defaultdict(list)
        accuracy_score_list = []
        precision_score_list = []
        recall_score_list = []
        f1_score_list = []
        model.eval()
        val_iterator = tqdm(val_loader, desc="Validation Iteration")
        for batch_idx, val_batch in enumerate(val_iterator):
            if config.model == 'LSAN':
                x, y_true = val_batch
                if config.use_cuda:
                    x = x.cuda()
                    y_true = y_true.cuda()
                y_pred_val = model(x)

            elif config.model == 'LSAN-BERT' or 'BERT':
                input_ids, token_type_ids, attention_masks, y_true = val_batch
                if config.use_cuda:
                    input_ids = input_ids.cuda()
                    token_type_ids = token_type_ids.cuda()
                    attention_masks = attention_masks.cuda()
                    y_true = y_true.cuda()
                y_pred_val = model(input_ids, token_type_ids, attention_masks)

            # calculate loss for model-parameters selection
            loss = criterion(y_pred_val, y_true)
            # evaluation metrics: hamming loss, accuracy score, precision, recall, f1
            labels_cpu = y_true.data.cpu()
            pred_cpu = y_pred_val.data.cpu()

            val_loss.append(float(loss.item()))
            true_labels, pred_labels = make_label_all_class(labels_cpu, pred_cpu, config.threshold)
            for i in range(0, config.num_of_class):
                make_label_each_class(true_labels, pred_labels, i, true_label_dict[i+1], pred_label_dict[i+1])

        avg_val_loss = np.mean(val_loss)
        for i in range(0, config.num_of_class):
            accuracy_score, precision_score, recall_score, f1_score = get_metrics_each_class(
                                                                      true_label_dict[i + 1], pred_label_dict[i + 1])
            accuracy_score_list.append(accuracy_score)
            precision_score_list.append(precision_score)
            recall_score_list.append(recall_score)
            f1_score_list.append(f1_score)
        avg_accuracy_score = np.mean(accuracy_score_list)
        macro_precision_score = np.mean(precision_score_list)
        macro_recall_score = np.mean(recall_score_list)
        macro_f1_score = np.mean(f1_score_list)

        print("valid_accuracy_score_each_class = {}".format(str(["%.4f" % accuracy for accuracy in accuracy_score_list])))
        print("valid_precision_score_each_class = {}".format(str(["%.4f" % precision for precision in precision_score_list])))
        print("valid_recall_score_each_class = {}".format(str(["%.4f" % recall for recall in recall_score_list])))
        print("valid_f1_score_each_class = {}".format(str(["%.4f" % f1 for f1 in f1_score_list])))

        return avg_val_loss, avg_accuracy_score, macro_precision_score, macro_recall_score, macro_f1_score


def test(load_path):
    if config.model == 'LSAN':
        # Load Glove Embedding
        glove_embeddings = vocab.GloVe(name='6B', dim=300, cache='./cache')
        # Load Label Embedding
        sbert_embeddings = np.load(config.label_embedding_path)
        # Load Dataset
        test_dataset = SequenceDataset(config.test_file_path_task, glove_embeddings)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size)

        # Initialize LSAN Model
        word_embedding = glove_embeddings.vectors.float()
        label_embedding = torch.from_numpy(sbert_embeddings).float()  # self-constructed label embedding
        # reference to: https://www.sbert.net/
        model = StructuredSelfAttention(batch_size=config.batch_size,
                                        lstm_hid_dim=config.lstm_hidden_dimension,
                                        d_a=config.d_a,
                                        n_classes=config.num_of_class,
                                        word_embedding=word_embedding,
                                        label_embedding=label_embedding)

    elif config.model == 'LSAN-BERT':
        # Load Bert Tokenizer
        if config.bert_model == 'bert':
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif config.bert_model == 'albert':
            bert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
        else:
            raise Exception('Error Bert model.')

        # Load Dataset
        test_dataset = SequenceDatasetBert(config.test_file_path_task, bert_tokenizer)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size)
        label_encodings = load_label_description(bert_tokenizer)
        model = StructuredSelfAttentionBert(batch_size=config.batch_size,
                                            lstm_hid_dim=config.lstm_hidden_dimension,
                                            d_a=config.d_a,
                                            n_classes=config.num_of_class,
                                            is_train=False,
                                            label_input_ids=torch.tensor(label_encodings['input_ids']).to(
                                                config.device),
                                            label_attention_masks=torch.tensor(label_encodings['attention_mask']).to(
                                                config.device))

    elif config.model == 'BERT':
        # Load Bert Tokenizer
        if config.bert_model == 'bert':
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif config.bert_model == 'albert':
            bert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
        else:
            raise Exception('Error Bert model.')

        # Load Dataset
        test_dataset = SequenceDatasetBert(config.test_file_path_task, bert_tokenizer)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size)
        model = BertLinear(hidden_size=config.hidden_size,
                           hidden_dropout_prob=config.hidden_dropout_prob,
                           n_classes=config.num_of_class,
                           is_train=True)

    print(config)
    print('Test Set Size {}'.format(len(test_dataset)))

    model.load_model(load_path)
    if config.use_cuda:
        model.cuda()

    with torch.no_grad():
        test_loss = []
        true_label_dict, pred_label_dict = defaultdict(list), defaultdict(list)
        accuracy_score_list = []
        precision_score_list = []
        recall_score_list = []
        f1_score_list = []
        if config.loss == 'BCELoss':
            criterion = torch.nn.BCELoss()
        elif config.loss == 'FocalLoss':
            criterion = FocalLoss(alpha=config.focal_loss_alpha, gamma=config.focal_loss_gamma, reduction=config.focal_loss_reduction)
        model.eval()
        test_iterator = tqdm(test_loader, desc="Test Iteration")
        for batch_idx, test_batch in enumerate(test_iterator):
            if config.model == 'LSAN':
                x, y_true = test_batch
                if config.use_cuda:
                    x = x.cuda()
                    y_true = y_true.cuda()
                y_pred_val = model(x)

            elif config.model == 'LSAN-BERT' or 'BERT':
                input_ids, token_type_ids, attention_masks, y_true = test_batch
                if config.use_cuda:
                    input_ids = input_ids.cuda()
                    token_type_ids = token_type_ids.cuda()
                    attention_masks = attention_masks.cuda()
                    y_true = y_true.cuda()
                y_pred_val = model(input_ids, token_type_ids, attention_masks)

            # calculate loss for model-parameters selection
            loss = criterion(y_pred_val, y_true)
            # evaluation metrics: hamming loss, accuracy score, precision, recall, f1
            labels_cpu = y_true.data.cpu()
            pred_cpu = y_pred_val.data.cpu()

            test_loss.append(float(loss.item()))
            true_labels, pred_labels = make_label_all_class(labels_cpu, pred_cpu, config.threshold)
            for i in range(0, config.num_of_class):
                make_label_each_class(true_labels, pred_labels, i, true_label_dict[i+1], pred_label_dict[i+1])

        avg_test_loss = np.mean(test_loss)
        for i in range(0, config.num_of_class):
            accuracy_score, precision_score, recall_score, f1_score = get_metrics_each_class(
                                                                      true_label_dict[i + 1], pred_label_dict[i + 1])
            accuracy_score_list.append(accuracy_score)
            precision_score_list.append(precision_score)
            recall_score_list.append(recall_score)
            f1_score_list.append(f1_score)
        avg_accuracy_score = np.mean(accuracy_score_list)
        macro_precision_score = np.mean(precision_score_list)
        macro_recall_score = np.mean(recall_score_list)
        macro_f1_score = np.mean(f1_score_list)
        print("avg_val_loss = %.4f" % avg_test_loss)
        print("avg_accuracy_score = %.4f" % avg_accuracy_score)
        print("macro_precision_score = %.4f" % macro_precision_score)
        print("macro_recall_score = %.4f" % macro_recall_score)
        print("macro_f1_score = %.4f" % macro_f1_score)

        for i in range(config.num_of_class):
            print('---class {}---'.format(i + 1))
            print("amount_of_true_label = {}".format(len([j for j in true_label_dict[i + 1] if j == 1])))
            print("accuracy_score = %.4f" % accuracy_score_list[i])
            print("precision_score = %.4f" % precision_score_list[i])
            print("recall_score = %.4f" % recall_score_list[i])
            print("f1_score = %.4f" % f1_score_list[i])


def predict(load_path, file_path='./data/Task1.predict.csv', save_path='./data/Task1.predict.result.csv'):
    if config.model == 'LSAN':
        # Load Glove Embedding
        glove_embeddings = vocab.GloVe(name='6B', dim=300, cache='./cache')
        # Load Label Embedding
        sbert_embeddings = np.load(config.label_embedding_path)
        # Load Dataset
        predict_dataset = SequenceDataset(file_path, glove_embeddings)
        predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=config.batch_size, shuffle=False)

        # Initialize LSAN Model
        word_embedding = glove_embeddings.vectors.float()
        label_embedding = torch.from_numpy(sbert_embeddings).float()  # self-constructed label embedding
        # reference to: https://www.sbert.net/
        model = StructuredSelfAttention(batch_size=config.batch_size,
                                        lstm_hid_dim=config.lstm_hidden_dimension,
                                        d_a=config.d_a,
                                        n_classes=config.num_of_class,
                                        word_embedding=word_embedding,
                                        label_embedding=label_embedding)

    elif config.model == 'LSAN-BERT':
        # Load Bert Tokenizer
        if config.bert_model == 'bert':
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif config.bert_model == 'albert':
            bert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
        else:
            raise Exception('Error Bert model.')

        # Load data
        predict_dataset = SequenceDatasetBert(file_path, bert_tokenizer)
        predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=config.batch_size, shuffle=False)
        label_encodings = load_label_description(bert_tokenizer)
        model = StructuredSelfAttentionBert(batch_size=config.batch_size,
                                            lstm_hid_dim=config.lstm_hidden_dimension,
                                            d_a=config.d_a,
                                            n_classes=config.num_of_class,
                                            is_train=False,
                                            label_input_ids=torch.tensor(label_encodings['input_ids']).to(
                                                config.device),
                                            label_attention_masks=torch.tensor(label_encodings['attention_mask']).to(
                                                config.device))

    elif config.model == 'BERT':
        # Load Bert Tokenizer
        if config.bert_model == 'bert':
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif config.bert_model == 'albert':
            bert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
        else:
            raise Exception('Error Bert model.')

        # Load Dataset
        predict_dataset = SequenceDatasetBert(file_path, bert_tokenizer)
        predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=config.batch_size)
        model = BertLinear(hidden_size=config.hidden_size,
                           hidden_dropout_prob=config.hidden_dropout_prob,
                           n_classes=config.num_of_class,
                           is_train=True)

    if len(predict_dataset) == 0:
        return

    # print(config)
    # print('Test Set Size {}'.format(len(predict_dataset)))

    model.load_model(load_path)
    if config.use_cuda:
        model.cuda()

    with torch.no_grad():
        pred_label_dict = defaultdict(list)

        model.eval()
        for batch_idx, predict_batch in enumerate(predict_loader):
            if config.model == 'LSAN':
                x, y_true = predict_batch
                if config.use_cuda:
                    x = x.cuda()
                    y_true = y_true.cuda()
                y_pred_val = model(x)

            elif config.model == 'LSAN-BERT' or 'BERT':
                input_ids, token_type_ids, attention_masks, y_true = predict_batch
                if config.use_cuda:
                    input_ids = input_ids.cuda()
                    token_type_ids = token_type_ids.cuda()
                    attention_masks = attention_masks.cuda()
                    y_true = y_true.cuda()
                y_pred_val = model(input_ids, token_type_ids, attention_masks)

            pred_cpu = y_pred_val.data.cpu()

            pred_labels = (pred_cpu >= config.threshold).int()
            for i in range(0, config.num_of_class):
                pred_label_class = [label.item() for label in pred_labels[:, i]]
                pred_label_dict[i + 1].extend(pred_label_class)

    save_prediction_result(pred_label_dict, file_path, save_path)


if __name__ == '__main__':
    fire.Fire()



