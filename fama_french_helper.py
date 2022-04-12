import os
import pandas as pd
import numpy as np
import datetime as dt
import statsmodels.api as smf
from sklearn.linear_model import LinearRegression


def process_ff_file(ff_file):
    ff_path = os.path.join(os.getcwd(), ff_file)
    ff_df = pd.read_csv(ff_path, index_col=[0])
    ff_df.dropna(subset=["date"], inplace=True)
    if '/' in ff_df["date"].iloc[0]:
        str_format =  "%d/%m/%Y"
    else:
        str_format =  "%Y-%m-%d"
    ff_df["date"] = ff_df["date"].apply(lambda x: dt.datetime.strptime(x, str_format))
    ff_df[['year', 'month']] = ff_df[['year', 'month']].astype(int)
    return ff_df


def ff_returns(
        ff_ptf_file: str,
        ff_factors_file: str,
        start_date: dt.datetime,
        end_date: dt.datetime,
        ret_method: str = "vwret",
        factors: list = ['mktrf', 'smb', 'hml']):
    ''''
    Returns Fama French regression coefficients for each portfolio in a given time range for a given portfolios returns file

    Attributes:
    ff_ptf_file : portfolio returns file name
    ff_factors_file: fama french factros file name
    ret_method: "vwret" or "ewret" 

    target_portfolio : portfolio name , example : "49_Industry_Portfolios_CSV"
  '''
    ff_factors_df = process_ff_file(ff_factors_file)
    ff_ptf_df = process_ff_file(ff_ptf_file)
    merged_data = pd.merge(ff_ptf_df, ff_factors_df, how='inner', on=['year', 'month'])
    merged_data = merged_data.rename(columns={"date_y": "date"})
    ptf_returns_cols = [key for key in ff_ptf_df.keys() if ret_method in key]
    ptf_list = [ret_col[:-6] for ret_col in ptf_returns_cols]
    ptf_dict = {}
    for i in range(len(ptf_list)):
        target_cols = list(set(['date', 'year', 'month', ptf_returns_cols[i], 'rf']).union(set(factors)))
        ptf_ret_df = merged_data[target_cols]
        ptf_ret_df = ptf_ret_df[(ptf_ret_df['date'] >= start_date) & (ptf_ret_df['date'] <= end_date)]
        ptf_ret_df['ptf_excess_return'] = ptf_ret_df[ptf_returns_cols[i]] - ptf_ret_df['rf']
        # OLS regression
        y = ptf_ret_df['ptf_excess_return']
        X = ptf_ret_df[factors]
        X_aug = smf.add_constant(X)
        model = smf.OLS(y, X_aug)
        results = model.fit()
        ptf_ret_df['predicted_return'] = results.predict()
        ptf_dict[ptf_list[i]] = results.params
    return ptf_dict
