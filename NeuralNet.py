# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:36:02 2019

@author: mille
"""

import torch
import numpy as np


class NN(object):
    def __init__(self, D_in=14, H1=100, H2=100, D_out=3):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Running on CUDA")
        else:
            device = torch.device("cpu")
            print("Running on CPU")
        
        self.model = torch.nn.Sequential(torch.nn.Linear(D_in, H1),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(H1, H2),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(H2, D_out))
        
        self.loss_fn = torch.nn.MSELoss(reduction="sum")
        self.lr = 1e-4
    
    def forward(self, x, y=None):
        y_pred = self.model(x)
        
        return y_pred
        
        if y != None:
            loss = self.loss_fn(y_pred, y)
            self.model.zero_grad()
            loss.backward()
        
        with torch.no_grad():
            for param in self.model.parameters():
                param -= self.lr * param.grad