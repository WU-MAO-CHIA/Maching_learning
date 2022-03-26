# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 05:13:16 2021

@author: narut
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class FutureDataset(Dataset):
  def __init__(self, data, seq_len = 60):
    self.data = data
    self.X = torch.tensor(data[:, 0:6], dtype=torch.double)
    self.y = torch.tensor(data[:, 6], dtype=torch.double)
    self.seq_len = seq_len
    
    
  def __len__(self):
    return self.data.shape[0]
  
  def __getitem__(self, idx):
    """
    X: idx-seq ~ idx-1
    y: idx
    """
    if idx >= self.seq_len:
      idx_start = idx - self.seq_len
      x = self.X[idx_start:idx, :]
    else:
      padding = self.X[0].repeat(self.seq_len - idx, 1)
      x = self.X[0:idx, :]
      x = torch.cat((padding, x), 0)

    return x, self.y[idx]
