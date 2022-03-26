# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 05:12:38 2021

@author: narut
"""
import torch
from torch import nn

class FutureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, device = None, num_layers = 1, dropout = 0.5):
        super().__init__()
        self.input_size = input_size  # this is the number of features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            batch_first = True,
            num_layers = self.num_layers,
            dropout = dropout
        )

        self.linear = nn.Linear(in_features = self.hidden_size, out_features = 1)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        #h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.double).requires_grad_().to(self.device)
        #c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.double).requires_grad_().to(self.device)
        """
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.
        """
        #out, _ = self.lstm(x, (h0, c0))
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = out[:, -1, :] 
        out = self.linear(out).squeeze()
        return out
