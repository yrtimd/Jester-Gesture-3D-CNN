import torch
import torch.nn as nn

class MonitorLRDecay(object):
    """
    Decay learning rate with some patience
    """
    def __init__(self, decay_factor, patience):
        self.best_loss = 999999
        self.decay_factor = decay_factor
        self.patience = patience
        self.count = 0

    def __call__(self, current_loss, current_lr):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.count = 0
        elif self.count > self.patience:
            current_lr = current_lr * self. decay_factor
            print(" > New learning rate -- {0:}".format(current_lr))
            self.count = 0
        else:
            self.count += 1
        return current_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.cpu().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
