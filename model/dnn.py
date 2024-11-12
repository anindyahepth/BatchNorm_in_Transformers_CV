import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as dsets
import matplotlib.pylab as plt
from datetime import datetime
from torchvision.transforms import v2 as transforms_v2
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Batch_Norm(nn.Module):
  def __init__(self, feature_dim):
    super().__init__()

    self.BN = nn.BatchNorm1d(feature_dim)

  def forward(self, x):
    x = self.BN(x)
    return x



class DNN(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_class, depth):
    super().__init__()

    self.Lin1 = nn.Linear(input_dim, hidden_dim)
    self.act1 = nn.Sigmoid()
    self.layers = nn.ModuleList([])
    for _ in range(depth):
       self.layers.append(nn.ModuleList([
          nn.Linear(hidden_dim, hidden_dim),
          nn.Sigmoid()
        ]))
    self.Linf = nn.Linear(hidden_dim, hidden_dim)
    self.actf = nn.Sigmoid()
    self.out = nn.Linear(hidden_dim, num_class)
    self.input_dim = input_dim


  def forward(self, x):
    size = x.size()
    b = size[0]
    x = rearrange(x, 'b c h w -> b (c h w)', b=b)
    x = self.Lin1(x)
    x = self.act1(x)
    for Lin, act in self.layers:
      x = Lin(x)
      x = act(x)
    x = self.Linf(x)
    #y = x[:,99]
    x = self.actf(x)
    x = self.out(x)
    return x



class DNNBN(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_class, depth):
    super().__init__()

    self.Lin1 = nn.Linear(input_dim, hidden_dim)
    self.norm1 = Batch_Norm(hidden_dim)
    self.act1 = nn.Sigmoid()
    self.layers = nn.ModuleList([])
    for _ in range(depth):
       self.layers.append(nn.ModuleList([
          nn.Linear(hidden_dim, hidden_dim),
          Batch_Norm(hidden_dim),
          nn.Sigmoid()
        ]))
    self.Linf = nn.Linear(hidden_dim, hidden_dim)
    self.normf = Batch_Norm(hidden_dim)
    self.actf = nn.Sigmoid()
    self.out = nn.Linear(hidden_dim, num_class)

  def forward(self, x):
    size = x.size()
    b = size[0]
    x = rearrange(x, 'b c h w -> b (c h w)', b=b)
    x = self.norm1(self.Lin1(x))
    x = self.act1(x)
    for Lin, norm, act in self.layers:
      x = Lin(x)
      x = norm(x)
      x = act(x)
    x = self.normf(self.Linf(x))
    # w = x[:,0]
    x = self.actf(x)
    x = self.out(x)
    return x

