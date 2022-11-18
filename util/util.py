import torch.nn as nn
import numpy as np
import math
import functools
import operator

def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

def get_n_params(model: nn.Module) -> int:

    res = sum((functools.reduce(operator.mul, p.size()) for p in model.parameters()))

    return '{:,}'.format(res)
