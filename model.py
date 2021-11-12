import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepCharacteristics(nn.Module):
    """

    """
    def __init__(self, n_layers, in_channels, out_channels, features, activation_type= 'relu', bias=True):
        super().__init__()
        self.layers = []
        self.layers.append(DenseBlock(in_channels=in_channels,
                                      out_channels=features,
                                      activation_type=activation_type,
                                      bias=bias))
        #intermediate layers
        for _ in range(n_layers-2):
            self.layers.append(DenseBlock(in_channels=features,
                                          out_channels=features,
                                          activation_type=activation_type,
                                          bias=bias))
        #last layer
        self.layers.append(DenseBlock(in_channels=features,
                                          out_channels=out_channels,
                                          activation_type=activation_type,
                                          bias=bias))
        self.FC = nn.Sequential(*self.layers)
    
    def forward(self,Z):
        Y = self.FC(Z)
        return Y

class DenseBlock(nn.Module):

    def __init__(self, in_channels, out_channels, activation_type= 'relu', bias=True):
        super().__init__()

        self.linear = nn.linear(in_channels, out_channels, bias=True)

        if activation_type == 'linear':
            self.activate = nn.Identity()
        elif activation_type == 'relu':
            self.activate = nn.ReLU(inplace=True)
        elif activation_type == 'lrelu':
            self.activate = nn.LeakyReLU(inplace=True)
        else:
            raise NotImplementedError(f'Not implemented activation function: '
                                      f'`{activation_type}`!')
    def forward(self, Z):
        Z = self.linear(Z)
        Z = self.activate(Z)
        return Z

class NonLinearRankWeight(nn.Module):
    def __init__(self,)
        super().__init__()
    
    def forward(self, Y):
