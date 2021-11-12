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

        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

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

class SortedFactorModel(nn.Module):
    def __init__(self, n_layers,
                        in_channels,
                        features,
                        n_deep_factors,
                        n_BM_factors,
                        n_portfolio,
                        activation_type= 'relu',
                        bias=True):
        super().__init__()
        
        self.DC_network = DeepCharacteristics(n_layers, in_channels, n_deep_factors, features, activation_type, bias)
        self.beta = nn.Parameter(torch.randn(n_portfolio, n_deep_factors), requires_grad=True)
        self.gamma = nn.Parameter(torch.randn(n_portfolio, n_BM_factors), requires_grad=True)
        self.register_parameter(name='gamma', param=self.gamma)
        self.register_parameter(name='beta', param=self.beta)

    def forward(self, Z, r, g):
        """
        Args:
            Z ([Tensor(MxK)]): firm characteristics
            r ([Tensor(M)]): firm returns
            g ([Tensor(D)]): benchmark factors
        """
        Y = self.DC_network(Z)
        print()
        W = rank_weight(Y) # M x P \ P := n_deep_factors
        f = torch.matmul(W.transpose(1, 2), r) # P x 1
        R = torch.matmul(self.beta[None, :], f) + torch.matmul(self.gamma[None, :], g)
        return R


def rank_weight(Y, method="softmax"):
    """Applies the rank weight operation

    Args:
        Y      ([Tensor(M x N)])
        method (string)
    """
    eps = 1-6
    mean = torch.mean(Y, axis=0)
    var = torch.var(Y, axis=0)
    normalised_data = (Y - mean) / (var + eps)
    if method == "softmax":
        y_p = -50 * torch.exp(-5 * normalised_data)
        y_n = -50 * torch.exp(5 * normalised_data)
        softmax = nn.Softmax(dim=1)
        W = softmax(y_p) - softmax(y_n)
    if method == "equal_ranks":
        pass
        
    return W
