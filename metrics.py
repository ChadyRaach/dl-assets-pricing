from scipy.optimize import minimize
import torch
import numpy as np


def compute_Rts(R, R_p):

    R = torch_to_numpy(R)
    R_p = torch_to_numpy(R_p)
    R_i = R.mean(axis=0)
    return 1 - ((R - R_p)**2).mean() / ((R - R_i)**2).mean()


def compute_PE(R, R_p):

    R = torch_to_numpy(R)
    R_p = torch_to_numpy(R_p)
    alpha_i = (R - R_p).mean(axis=0)
    return (alpha_i**2).mean()


def compute_CS(R, betas):

    betas = torch_to_numpy(betas)
    R = torch_to_numpy(R)

    _, N = R.shape
    R_i = R.mean(axis=0)

    X = np.concatenate([np.ones([N, 1]), betas], axis=1)

    Q = minimize(fun=lambda x: (X @ x - R_i).T @ (X @ x - R_i),
                 x0=np.ones(X.shape[1]),
                 jac=lambda x: 2 * X.T @ X @ x - 2 * R_i.T @ X)["fun"]

    Q0 = minimize(fun=lambda x: (np.ones(N) * x - R_i).T @ (np.ones(N) * x - R_i),
                  x0=0,
                  jac=lambda x: 4 * N * x - 2 * R_i.sum())["fun"]

    return 1 - Q / Q0


def torch_to_numpy(var):
    if type(var) == torch.nn.parameter.Parameter or type(var) == torch.Tensor:
        var = var.detach().numpy()
    return var
