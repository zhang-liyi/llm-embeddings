import torch
import numpy as np

def get_loss_tv(output, target, softmax=False, reduction='mean'):

    if softmax:
        output = torch.softmax(output, 1)

    diff = torch.abs(output-target)

    if reduction == 'mean':
        return torch.mean(torch.max(diff, dim=1).values)
    elif reduction == 'sum':
        return torch.sum(torch.max(diff, dim=1).values)
    elif reduction == 'none':
        return torch.max(diff, dim=1).values

def get_loss_l2(output, target, softmax=False, reduction='mean'):

    if softmax:
        output = torch.softmax(output, 1)

    if reduction == 'mean':
        return torch.mean(torch.sum((output-target)**2, dim=1))
    elif reduction == 'sum':
        return torch.sum(torch.sum((output-target)**2, dim=1))
    elif reduction == 'none':
        return torch.sum((output-target)**2, dim=1)

def get_loss_ce(output, target, softmax=False, reduction='mean'):
    # Assumes that inputs are probabilities (post-softmax) when softmax=False
    if softmax:
        output = torch.softmax(output, 1)

    if reduction == 'mean':
        return -torch.mean(torch.sum(torch.multiply(torch.log(output), target), dim=1))
    elif reduction == 'sum':
        return -torch.sum(torch.sum(torch.multiply(torch.log(output), target), dim=1))
    elif reduction == 'none':
        return torch.sum(torch.multiply(torch.log(output), target), dim=1)