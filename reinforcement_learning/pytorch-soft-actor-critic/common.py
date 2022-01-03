import torch
import torch.nn as nn


def hard_update(target: nn.Module, source: nn.Module):
    for target_params, source_params in zip(target.parameters(), source.parameters()):
        target_params.data.copy_(source_params.data)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0.001)
