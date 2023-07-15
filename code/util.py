import torch
import torch.nn as nn
import numpy as np
import gym
import random
import copy
import torch.nn.functional as F
from torch.autograd import Variable
import math
import sys
import argparse
import os
import importlib
import torch.optim as optim
import wandb

def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def global_dict_init():
    global _global_dict
    _global_dict = {}


def set_global_dict_value(key, value):
    _global_dict[key] = value


def get_global_dict_value(key):
    return _global_dict[key]
