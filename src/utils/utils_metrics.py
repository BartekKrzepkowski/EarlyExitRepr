import torch


def correct_metric(y_pred, y_true):
    correct = (torch.argmax(y_pred.data, dim=1) == y_true).sum().item()
    return correct