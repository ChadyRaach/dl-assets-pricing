import wrds
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import *
from pandas.tseries.offsets import *


def get_monthly_data():
    
     ## tokens required for wrds connection
    conn = wrds.Connection()
    crsp_m = conn.raw_sql("""
                          select a.permno, a.permco, b.ncusip, b.comnam, a.date, 
                          b.shrcd, b.exchcd, b.siccd,
                          a.retx, a.ret, a.prc,a.ask, a.bid, a.spread, a.cfacpr, a.cfacshr, a.vol, a.shrout
                          from crsp.msf as a
                          left join crsp.msenames as b
                          on a.permno=b.permno
                          and b.namedt<=a.date
                          and a.date<=b.nameendt
                          and b.shrcd between 10 and 11
                          """, date_cols = ['date'])

    crsp_m[['permco','permno']]=\
        crsp_m[['permco','permno']].astype(int)

    # Line up date to be end of month
    crsp_m['jdate']=crsp_m['date']+MonthEnd(0)

    # adjusted price
    crsp_m['ajd prc']=crsp_m['prc'].abs()/crsp_m['cfacpr'] 

    # total shares out adjusted
    crsp_m['adj tso']=crsp_m['shrout']*crsp_m['cfacshr']*1e3 

    # market cap in $
    crsp_m['me'] = crsp_m['ajd prc']*crsp_m['adj tso']

    # sum of me across different permno belonging to same permco a given date
    crsp_summe = crsp_m.groupby(['jdate','permco'])['me'].sum().reset_index()\
        .rename(columns={'me':'me_comp'})
    crsp_m=pd.merge(crsp_m, crsp_summe, how='inner', on=['jdate','permco'])
    
    return crsp_m

def get_quarterly_data():
    
    conn = wrds.Connection()
    comp = conn.raw_sql("""
                    select gvkey, datadate, cusip, 
                    seqq, pstkq, txditcq, ceqq, atq, ltq,
                    cogsq, revtq, tieq, tieq, xsgaq, oibdpq, chq, scfq, 
                    dd1q, dlcq, dlttq, dpq, dpactq, rdipq, drcq, saleq, 
                    xoprq, nopiq, intanq, icaptq, ivltq, npatq
                    from comp.fundq
                    where indfmt='INDL' 
                    and datafmt='STD'
                    and popsrc='D'
                    and consol='C'
                    """, date_cols=['datadate'])

    comp['year']=comp['datadate'].dt.year
    comp['month'] =comp['datadate'].dt.month

    # select sample where seq is greater than 0
    comp = comp.loc[comp['seqq']>0]

    # create shareholders’ equity
    # 1st choice: shareholders' equity SEQQ
    # 2nd choice: shareholders' equity CEQQ + PSTKQ
    # 3rd choice: shareholders' equity ATQ - LTQ
    comp['sheq']=np.where(comp['seqq'].isnull(), comp['ceqq']+comp['pstkq'], comp['seqq'])
    comp['sheq']=np.where(comp['sheq'].isnull(),comp['atq'] - comp['ltq'], comp['sheq'])
    comp['sheq']=np.where(comp['sheq'].isnull(),0,comp['sheq'])

    # fill in missing values for deferred taxes and investment tax credit
    comp['txditcq']=comp['txditcq'].fillna(0)

    # create book equity
    # Daniel and Titman (JF 1997):    
    # BE = stockholders' equity + deferred taxes + investment tax credit - Preferred Stock
    comp['book equity']=comp['sheq']+comp['txditcq']-comp['pstkq']

    # keep only records with non-negative book equity
    comp = comp.loc[comp['book equity']>=0]
    
    # CCM linking table#
    ccm=conn.raw_sql("""
                      select gvkey, lpermno as permno, lpermco as permco, linktype, linkprim, 
                      linkdt, linkenddt
                      from crsp.ccmxpf_linktable
                      where (linktype ='LU' or linktype='LC')
                      and linkprim in ('P', 'C') and usedflag=1
                      """, date_cols=['linkdt', 'linkenddt'])

    # if linkenddt is missing then set to today date
    ccm['linkenddt']=ccm['linkenddt'].fillna(pd.to_datetime('today'))

    ccm_merged=pd.merge(comp,ccm,how='left',on=['gvkey'])
    ccm_merged['jdate']=ccm_merged['datadate']+MonthEnd(0)
    ccm_merged['year']=ccm_merged.datadate.dt.year

    # Impose date ranges
    ccm_valid = ccm_merged[(ccm_merged['datadate']>=ccm_merged['linkdt'])&(ccm_merged['datadate']<=ccm_merged['linkenddt'])]
    
    return ccm_valid