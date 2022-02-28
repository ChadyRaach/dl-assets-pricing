import datetime as dt
from dateutil import parser
import pandas as pd
import torch
from torch.utils.data import Dataset


def next_month(date: dt.datetime) -> dt.datetime:

    date = date.replace(day=1)
    date = date + dt.timedelta(days=32)
    return date.replace(day=1)


def previous_month(date: dt.datetime) -> dt.datetime:

    date = date.replace(day=1)
    date = date - dt.timedelta(days=27)
    return date.replace(day=1)


def previous_x_month(date: dt.datetime, x: int) -> dt.datetime:

    date = date.replace(day=1)
    for _ in range(x):
        date = previous_month(date)
    return date


def last_day_of_month(any_day):
    # this will never fail
    # get close to the end of the month for any day, and add 4 days 'over'
    next_month = any_day.replace(day=28) + dt.timedelta(days=4)
    # subtract the number of remaining 'overage' days to get last day of current month, or said programattically said, the previous day of the first of next month
    return next_month - dt.timedelta(days=next_month.day)


class Dataset(Dataset):
    def __init__(
            self, firm_characteristics, firm_returns, benchmark_factors, portfolios_returns, transform=None,
            start_date=None, end_date=None):
        self.firm_characteristics = torch.Tensor(firm_characteristics)
        self.firm_returns = torch.Tensor(firm_returns)
        self.portfolios_returns = torch.Tensor(portfolios_returns)
        self.benchmark_factors = torch.Tensor(benchmark_factors)
        self.start_date = start_date
        self.end_date = end_date
        self.lag_adjustment()

    def __getitem__(self, index):
        Z = self.firm_characteristics[index]
        r = self.firm_returns[index]
        g = self.benchmark_factors[index]
        R = self.portfolios_returns[index]

        return Z, r, g, R

    def __len__(self):
        return len(self.firm_characteristics)

    def lag_adjustment(self):
        self.firm_characteristics = self.firm_characteristics[:-1]
        self.firm_returns = self.firm_returns[1:]
        self.benchmark_factors = self.benchmark_factors[1:]
        self.portfolios_returns = self.portfolios_returns[1:]


def train_val_split(ds, split_size=0.8):
    n = ds.__len__()
    assert 12 * (ds.end_date.year - ds.start_date.year) + (ds.end_date.month - ds.start_date.month) == n
    val_size = int(n * (1 - split_size))
    start_val = n - val_size
    val_set = Dataset(ds[start_val:][0],
                      ds[start_val:][1],
                      ds[start_val:][2],
                      ds[start_val:][3])
    train_set = Dataset(ds[:start_val][0],
                        ds[:start_val][1],
                        ds[:start_val][2],
                        ds[:start_val][3])
    print(
        f'Validation data on time range: {dt.datetime.strftime(previous_x_month(ds.end_date, val_size-1), "%Y-%m-%d")} to {dt.datetime.strftime(last_day_of_month(ds.end_date), "%Y-%m-%d")}')

    return train_set, val_set


def batchify(dataset, batch_size):
    return [dataset[i:i + batch_size] for i in range(len(dataset) - batch_size + 1)]
