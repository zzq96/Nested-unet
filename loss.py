import torch
import matplotlib.pyplot as plt
# from torchsummary import summary
import torchvision
from torch import nn
import matplotlib
import utils
import time
import PIL
from PIL import Image
import numpy as np
import torchvision.models as models
import os

def _fast_hist(label_true, label_pred, n_class):
    """
    Inputs:
    - label_true: numpy, (W, )
    = label_pred: numpy, (W, )

    Returns:
    - hist: numpy, (n_class, n_class), hist[i, j] shows the number of the true label i to pred label j
    """
    mask = (label_true < 255) & (label_true < n_class)
    x = n_class * label_true[mask].astype(int) + label_pred[mask]
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """
    Inputs:
    - label_trues: list, (H, w), numpy
    - label_preds: list, (H, w), numpy
    - n_class: int
    Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def iou_score(scores, label):
    iou = 0
    eval_iou= 0
    tmp = scores.max(dim=1)
    label_pred = tmp[1].data.cpu().numpy()
    label_true = label.data.cpu().numpy()
    for lbt, lbp in zip(label_true, label_pred):
        _, _, iou, _ = label_accuracy_score(lbt, lbp, scores.shape[1])
        eval_iou += iou
    return eval_iou / label.shape[0]

def evaluate_accuracy(data_iter ,net, lossf, device):
    """
    Inputs:
    - data_iter:
    - net:
    - lossf:
    - device:
    """
    assert isinstance(net, nn.Module)
    loss, acc, acc_cls, mean_iu, fwavacc = 0, 0, 0, 0, 0
    eval_acc, eval_acc_cls, eval_mean_iu, eval_fwavacc = 0, 0, 0, 0
    loss_sum, n = 0.0, 0
    cnt = 0
    for X, y in data_iter:
        cnt += 1
        X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        l = lossf(y_hat, y)
        loss += l.cpu().item()
        tmp = y_hat.max(dim=1)
        label_pred = tmp[1].data.cpu().numpy()
        label_true = y.data.cpu().numpy()
        for lbt, lbp in zip(label_true, label_pred):
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, y_hat.shape[1])
            eval_acc += acc
            eval_acc_cls += acc_cls
            eval_mean_iu += mean_iu
            eval_fwavacc += fwavacc
        n += y.shape[0]
    return loss / cnt, eval_acc / n, eval_acc_cls / n, eval_mean_iu / n, eval_fwavacc / n