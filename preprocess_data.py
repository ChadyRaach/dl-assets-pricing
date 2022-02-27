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


def last_day_of_month(any_day):
    # this will never fail
    # get close to the end of the month for any day, and add 4 days 'over'
    next_month = any_day.replace(day=28) + dt.timedelta(days=4)
    # subtract the number of remaining 'overage' days to get last day of current month, or said programattically said, the previous day of the first of next month
    return next_month - dt.timedelta(days=next_month.day)


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
        f'Validation data on time range: {dt.datetime.strftime((ds.end_date - dt.timedelta(days=30*val_size)).replace(day=1), "%Y-%m-%d")} to {dt.datetime.strftime(last_day_of_month(ds.end_date), "%Y-%m-%d")}')

    return train_set, val_set


def batchify(dataset, batch_size):
    return [dataset[i:i + batch_size] for i in range(len(dataset) - batch_size + 1)]
