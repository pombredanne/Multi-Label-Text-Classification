import numpy as np
import torch.nn as nn
from torchtext import vocab
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm, trange

from DataModule import SequenceDataset
from LSANModule import StructuredSelfAttention
from config import *
from utils import *


def load_dataset(task_file_path, word_embeddings):
    # Load task dataset
    train_dataset = SequenceDataset(task_file_path, word_embeddings)
    # print(train_dataset[43])

    # Split task dataset into Train and Validation set
    val_split = 0.2
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    shuffle_dataset = True
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=validation_sampler)
    return train_loader, val_loader, len(train_indices), len(val_indices)


def train(LSAN_model, train_loader, val_loader, criterion, optimizer):
    print_config()
    # Initialize Early Stopping Configurations
    patience = 5
    counter = 0
    minimum_val_loss = 1000
    best_epoch = 0
    # epoch loop
    epoch_iterator = trange(int(NUM_EPOCHS), desc="Epoch")
    for epoch in epoch_iterator:
        # training loop
        train_loss = []
        true_label_dict = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
            10: [],
            11: [],
            12: [],
            13: []
        }
        pred_label_dict = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
            10: [],
            11: [],
            12: [],
            13: []
        }
        accuracy_score_list = []
        precision_score_list = []
        recall_score_list = []
        f1_score_list = []
        train_iterator = tqdm(train_loader, desc="Train Iteration")
        LSAN_model.train()
        optimizer.zero_grad()
        for batch_idx, train_batch in enumerate(train_iterator):
            x, y_true = train_batch[0], train_batch[1]
            if use_cuda:
                x = x.cuda()
                y_true = y_true.cuda()
            y_pred = LSAN_model(x)
            loss = criterion(y_pred, y_true)
            loss.backward()
            # update model-parameters
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            # evaluation metrics: accuracy score, precision, recall, f1
            labels_cpu = y_true.data.cpu()
            pred_cpu = y_pred.data.cpu()

            train_loss.append(float(loss.item()))
            true_labels, pred_labels = make_label_all_class(labels_cpu, pred_cpu, threshold)
            for i in range(0, num_of_class):
                make_label_each_class(true_labels, pred_labels, i, true_label_dict[i+1], pred_label_dict[i+1])

        avg_train_loss = np.mean(train_loss)
        for i in range(0, num_of_class):
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
            = val(LSAN_model, val_loader, criterion)
        print("epoch %2d valid end : avg_val_loss = %.4f" % (epoch + 1, avg_val_loss))
        print("epoch %2d valid end : avg_accuracy_score = %.4f" % (epoch + 1, avg_accuracy_score))
        print("epoch %2d valid end : macro_precision_score = %.4f" % (epoch + 1, macro_precision_score))
        print("epoch %2d valid end : macro_recall_score = %.4f" % (epoch + 1, macro_recall_score))
        print("epoch %2d valid end : macro_f1_score = %.4f" % (epoch + 1, macro_f1_score))
        # check early stopping status
        if avg_val_loss < minimum_val_loss:
            minimum_val_loss = avg_val_loss
            LSAN_model.save_model(epoch=epoch + 1, path=SAVE_PATH)
            best_epoch = epoch + 1
            counter = 0
        else:
            counter = counter + 1
        if counter >= patience:
            print("Early stopping at epoch: " + str(epoch+1))
            print("Best epoch: ", best_epoch)
            break
    print('Training and Validation Stage Done...')


def val(LSAN_model, val_loader, criterion):
    with torch.no_grad():
        val_loss = []
        true_label_dict = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
            10: [],
            11: [],
            12: [],
            13: []
        }
        pred_label_dict = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
            10: [],
            11: [],
            12: [],
            13: []
        }
        accuracy_score_list = []
        precision_score_list = []
        recall_score_list = []
        f1_score_list = []
        LSAN_model.eval()
        val_iterator = tqdm(val_loader, desc="Validation Iteration")
        for batch_idx, val_batch in enumerate(val_iterator):
            x, y_true = val_batch[0], val_batch[1]
            if use_cuda:
                x = x.cuda()
                y_true = y_true.cuda()
            y_pred_val = LSAN_model(x)
            # calculate loss for model-parameters selection
            loss = criterion(y_pred_val, y_true)
            # evaluation metrics: accuracy score, precision, recall, f1
            labels_cpu = y_true.data.cpu()
            pred_cpu = y_pred_val.data.cpu()

            val_loss.append(float(loss.item()))
            true_labels, pred_labels = make_label_all_class(labels_cpu, pred_cpu, threshold)
            for i in range(0, num_of_class):
                make_label_each_class(true_labels, pred_labels, i, true_label_dict[i+1], pred_label_dict[i+1])

        avg_val_loss = np.mean(val_loss)
        for i in range(0, num_of_class):
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
        return avg_val_loss, avg_accuracy_score, macro_precision_score, macro_recall_score, macro_f1_score


if __name__ == '__main__':

    # Load Glove Embedding
    glove_embeddings = vocab.GloVe(name='6B', dim=300, cache='./cache')
    # Load Label Embedding
    sbert_embeddings = np.load('./cache/label_embeddings.npy')
    # Load Dataset
    train_loader, val_loader, train_size, val_size = load_dataset(TRAIN_FILE_PATH_TASK, glove_embeddings)
    print('Training Set Size {}, Validation Set Size {}'.format(train_size, val_size))

    # Initialize LSAN Model
    word_embedding = glove_embeddings.vectors.float()
    label_embedding = torch.from_numpy(sbert_embeddings).float()  # self-constructed label embedding
                                                                  # reference to: https://www.sbert.net/
    LSAN_model = StructuredSelfAttention(batch_size=BATCH_SIZE,
                                         lstm_hid_dim=lstm_hidden_dimension,
                                         d_a=d_a,
                                         n_classes=num_of_class,
                                         word_embedding=word_embedding,
                                         label_embedding=label_embedding)
    if use_cuda:
        LSAN_model.cuda()

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(LSAN_model.parameters(), lr=LEARNING_RATE, betas=BETAS, weight_decay=WEIGHT_DECAY)

    # Train and Validate Model
    train(LSAN_model, train_loader, val_loader, criterion, optimizer)






