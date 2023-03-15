import numpy as np
from .utils_pas import get_model_op
import torch.fx
import torch.nn as nn


def limit_loss(model, modify_names, ratio, threshold=0.5):
    branches = torch.tensor([]).cuda()
    for name, weights in model.named_modules():
        if '.scale' in name:
            w = weights.weight.detach()
            binary_w = (w > threshold).float()
            residual = w - binary_w
            branch_out = weights.weight - residual
            branches = torch.cat((branches, torch.sum(torch.squeeze(branch_out), dim=0, keepdim=True)), dim=0)


