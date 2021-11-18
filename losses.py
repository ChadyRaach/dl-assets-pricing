import torch
import torch.nn as nn

loss = nn.MSELoss()


def time_series_variation(gt_returns, pred_returns):
    """Computes times series error on portfolio returns.

    Args:
        gt_returns (T x N)
        pred_returns (T x N)
    """
    return loss(gt_returns, pred_returns)


def pricing_error(gt_returns, pred_returns):
    """Computes pricing errors on portfolio returns.

    Args:
        gt_returns (T x N)
        pred_returns (T x N)
    """
    cumulated_error = torch.mean(pred_returns - gt_returns, dim=0)
    total_error = torch.mean(torch.square(cumulated_error))
    return total_error
