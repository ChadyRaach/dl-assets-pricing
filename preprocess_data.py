import datetime as dt
from dateutil import parser
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

CHAR_TO_FILE = {"assets": 'Total assets [Annual sliding Eq ref]',
                "liabilities": 'Total liabilities [Annual sliding Eq ref]',
                "closing_adj": 'Closing adj. by FX and div.',
                "closing_ref": 'Closing ref.',
                "market_cap_ref": 'Market cap ref.',
                "market_cap": 'Market cap'}


def next_month(date: dt.datetime) -> dt.datetime:

    date = date.replace(day=1)
    date = date + dt.timedelta(days=32)
    return date.replace(day=1)


def as_float(a: str) -> float:
    try:
        return float(a)
    except Exception:
        return None


def get_latest_available_value(month: dt.datetime, df: pd.DataFrame, key: str) -> float:
    subdf = df.copy()
    subdf["dateTime"] = subdf.dateTime.apply(parser.parse)
    subdf = subdf[subdf["dateTime"] < month].sort_values("perEndDate", ascending=False)
    if len(subdf) > 0:
        return subdf[key].iloc[0]
    else:
        return 0


def average_monthly_value(month: dt.datetime, df: pd.DataFrame) -> float:

    unvalid_key = df.keys()[0]
    df["value"] = df[unvalid_key].apply(as_float)
    df = df.drop(unvalid_key, axis=1)
    df = df[df.value.notnull()]
    df["dateTime"] = df.index
    df["dateTime"] = df.dateTime.apply(parser.parse)
    df = df[(df["dateTime"] >= month) & (df["dateTime"] < next_month(month))]
    if len(df) > 0:
        return df.value.mean()
    else:
        return 0


def transform_to_monthly_freq(data, start_month, end_month):
    res = {}
    month = start_month
    while month < end_month:
        res[month] = {char: [] for char in CHAR_TO_FILE.keys()}
        for firm in data.keys():
            for char, file in CHAR_TO_FILE.items():
                if char in ["assets", "liabilities"]:
                    df = pd.DataFrame.from_dict(data[firm][file], orient="index")
                    res[month][char].append(get_latest_available_value(month, df, "value"))
                elif char in ["closing_adj", "closing_ref", "market_cap_ref", "market_cap"]:
                    df = pd.DataFrame.from_dict(data[firm][file], orient="columns")
                    if "Instrument" in df.index:
                        df = df.drop("Instrument", axis=0)
                    res[month][char].append(average_monthly_value(month, df))
        month = next_month(month)
    return res


class MyDataset(Dataset):
    def __init__(self, firm_characteristics, firm_returns, benchmark_factors, portfolios_indexes, transform=None):
        self.firm_characteristics = torch.Tensor(firm_characteristics)
        self.firm_returns = torch.Tensor(firm_returns)
        self.portfolios_indexes = portfolios_indexes
        self.benchmark_factors = torch.Tensor(benchmark_factors)
        self.temporal_relations()
        self.transform = transform

    def __getitem__(self, index):
        Z = self.firm_characteristics[index]
        r = self.firm_returns[index]
        g = self.benchmark_factors[index]
        R = self.portfolios_returns[index]

        if self.transform:
            Z, r = self.transform(Z, r)

        return Z, r, g, R

    def __len__(self):
        return len(self.firm_characteristics)

    def temporal_relations(self):
        self.firm_characteristics = self.firm_characteristics[: -1]
        portfolios_returns = []
        for portfolio_indexes in self.portfolios_indexes:
            portfolios_returns.append(torch.mean(self.firm_returns[1:, portfolio_indexes], axis=-1).view(-1, 1))
        self.portfolios_returns = torch.concat(portfolios_returns, axis=1)
        self.firm_returns = torch.Tensor(self.firm_returns)[:-1]
        self.benchmark_factors = self.benchmark_factors[:-1]


def transform(Z, r):
    p = torch.ones(Z.shape[0])
    for i in range(Z.shape[1]):
        p = p * (Z[:, i] != 0) * 1
    indexes = p == 1
    return Z[indexes], r[indexes]
