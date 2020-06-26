'''Customised Metrics'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from consts import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

def NLLLoss(inp, target, mask):
    dim =inp.size(2)
    mask = mask.unsqueeze(2).repeat(1,1,dim)
    target_one_hot = torch.nn.functional.one_hot(target, num_classes = dim)
    crossEntropy = -torch.log(torch.gather(inp, 1, target_one_hot))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss