import pandas as pd
from zipfile import ZipFile
from io import BytesIO
import urllib.request as urllib2
import numpy as np
import datetime as dt


def extract_returns_data(target_portfolio: str, start_month: dt.datetime, end_month: dt.datetime):
    ''''
    Returns Portfolio returns dataframe extracted from the fama french website : 
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html#Research

    Attributes:
    target_portfolio : portfolio name , example : "49_Industry_Portfolios_CSV"

    Remarks:
    The portfolios are constructed at the end of June.
    Missing data are indicated by -99.99 or -999.

    '''

    url = f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{target_portfolio}.zip"
    url_data = urllib2.urlopen(url).read()
    zip_file = ZipFile(BytesIO(url_data))
    csv_file = zip_file.open(zip_file.namelist()[0])
    data = pd.read_csv(csv_file, header=6)
    data.rename(columns={data.columns[0]: 'date'}, inplace=True)
    target_index = data.index[data['date'] == '202112'].tolist()[0]
    data = data[:target_index+1]
    data['date'] = data['date'].apply(lambda str_time: dt.datetime.strptime(str_time, '%Y%m'))
    target_data = data[(data['date'] >= start_month) & (data['date'] <= end_month)]
    return target_data
