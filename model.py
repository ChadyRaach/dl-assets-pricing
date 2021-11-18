import warnings

import torch
import torch.nn as nn


class DeepCharacteristics(nn.Module):
    """ """

    def __init__(self, n_layers, in_channels, out_channels, features, activation_type="relu", bias=True):
        super().__init__()
        self.layers = []
        self.layers.append(
            DenseBlock(in_channels=in_channels, out_channels=features, activation_type=activation_type, bias=bias)
        )
        # intermediate layers
        for _ in range(n_layers - 2):
            self.layers.append(
                DenseBlock(in_channels=features, out_channels=features, activation_type=activation_type, bias=bias)
            )
        # last layer
        self.layers.append(
            DenseBlock(in_channels=features, out_channels=out_channels, activation_type=activation_type, bias=bias)
        )
        self.FC = nn.Sequential(*self.layers)

    def forward(self, Z):
        Y = self.FC(Z)
        return Y


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation_type="relu", bias=True):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

        if activation_type == "linear":
            self.activate = nn.Identity()
        elif activation_type == "relu":
            self.activate = nn.ReLU(inplace=True)
        elif activation_type == "lrelu":
            self.activate = nn.LeakyReLU(inplace=True)
        else:
            raise NotImplementedError(f"Not implemented activation function: " f"`{activation_type}`!")

    def forward(self, Z):
        Z = self.linear(Z)
        Z = self.activate(Z)
        return Z


class SortedFactorModel(nn.Module):
    def __init__(
        self,
        n_layers,
        in_channels,
        features,
        n_deep_factors,
        n_BM_factors,
        n_portfolio,
        activation_type="relu",
        bias=True,
        ranking_method="softmax",
    ):
        super().__init__()
        self.DC_network = DeepCharacteristics(n_layers, in_channels, n_deep_factors, features, activation_type, bias)
        self.beta = nn.Parameter(torch.randn(n_portfolio, n_deep_factors), requires_grad=True)
        self.gamma = nn.Parameter(torch.randn(n_portfolio, n_BM_factors), requires_grad=True)
        self.register_parameter(name="gamma", param=self.gamma)
        self.register_parameter(name="beta", param=self.beta)
        self.ranking_method = ranking_method

    def forward(self, Z, r, g):
        """
        Args:
            Z ([Tensor(T x M x K)]): firm characteristics
            r ([Tensor(T x M)]): firm returns
            g ([Tensor(T x D)]): benchmark factors
        """
        if len(Z.size()) == 2:
            Z = Z[None, :, :]
        Y = self.DC_network(Z)
        W = rank_weight(Y, method=self.ranking_method)  # T x M x P \ P := n_deep_factors
        f = torch.matmul(W.transpose(1, 2), r)  # T x P x 1
        R = torch.matmul(self.beta[None, :], f) + torch.matmul(self.gamma[None, :], g)
        return R


def rank_weight(Y, method="softmax"):
    """Applies the rank weight operation

    Args:
        Y      ([Tensor(T x M x N)])
        method (string)
    """
    eps = 1 - 6
    mean = torch.mean(Y, axis=1)
    var = torch.var(Y, axis=1)
    normalised_data = (Y - mean[:, None, :]) / (var[:, None, :] + eps)
    if method == "softmax":
        y_p = -50 * torch.exp(-5 * normalised_data)
        y_n = -50 * torch.exp(5 * normalised_data)
        softmax = nn.Softmax(dim=2)
        W = softmax(y_p) - softmax(y_n)
    elif method == "equal_ranks":
        pass
        M, P = Y.size()[-2], Y.size()[-1]
        uniform_weight = 1 / (M // 3)
        sorted, indices = torch.sort(Y, dim=1)
        W = torch.zeros(Y.size())
        for i in range(P):
            W[indices.T[i][2 * M // 3:], i] = uniform_weight
            W[indices.T[i][M // 3: 2 * M // 3], i] = 0
            W[indices.T[i][: M // 3], i] = -uniform_weight
    else:
        warnings.warn(f"{method} not implemented yet. Softmax ranking will be applied.")
        return rank_weight(Y, method="softmax")

    return W
