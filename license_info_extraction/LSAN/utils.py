import os
import random
import torch
import numpy as np
from sklearn import metrics


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


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


def precision_k(true_mat, score_mat, k):
    p = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    for k in range(k):
        score_mat = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            score_mat[i][rank_mat[i, :-(k + 1)]] = 0
        score_mat = np.ceil(score_mat)
        # kk = np.argwhere(score_mat>0)
        mat = np.multiply(score_mat, true_mat)
        # print("mat",mat)
        num = np.sum(mat, axis=1)
        p[k] = np.mean(num / (k + 1))
    return np.around(p, decimals=4)


def NDCG_k(true_mat, score_mat, k):
    res = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    label_count = np.sum(true_mat, axis=1)

    for m in range(k):
        y_mat = np.copy(true_mat)
        for i in range(rank_mat.shape[0]):
            y_mat[i][rank_mat[i, :-(m + 1)]] = 0
            for j in range(m + 1):
                y_mat[i][rank_mat[i, -(j + 1)]] /= np.log(j + 1 + 1)

        dcg = np.sum(y_mat, axis=1)
        factor = get_factor(label_count, m + 1)
        ndcg = np.mean(dcg / factor)
        res[m] = ndcg
    return np.around(res, decimals=4)


def get_factor(label_count, k):
    res = []
    for i in range(len(label_count)):
        n = int(min(label_count[i], k))
        f = 0.0
        for j in range(1, n+1):
            f += 1/np.log(j+1)
        res.append(f)
    return np.array(res)


def make_label_all_class(true_mat, pred_mat, threshold):
    true_label_mat = true_mat.int()
    pred_label_mat = (pred_mat >= threshold).int()
    return true_label_mat.numpy(), pred_label_mat.numpy()


def make_label_each_class(true_arr, pred_arr, index, true_label_list, pred_label_list):
    true_label_class = true_arr[:, index]
    pred_label_class = pred_arr[:, index]
    true_label_list.extend(true_label_class)
    pred_label_list.extend(pred_label_class)


def get_metrics_each_class(y, y_pred):
    accuracy_score = metrics.accuracy_score(y, y_pred)
    precision_score = metrics.precision_score(y, y_pred, zero_division=1)
    recall_score = metrics.recall_score(y, y_pred, zero_division=1)
    f1_score = metrics.f1_score(y, y_pred, zero_division=1)
    return accuracy_score, precision_score, recall_score, f1_score

