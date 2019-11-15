#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 22:00:17 2019

@author: l-r-h
"""

#%%

#dataimport
import requests
import alpha_vantage
import json
import pandas as pd
import quandl
import datetime



#api_key = 'XIDQMKSM1WH47UEQ'
#symbols = ['FLWS','FCTY','FCCY','SRCE','FUBC']
#symbols_2 = ['VNET','TWOU','DGLD','JOBS','EGHT']
#symbols_3 = ['AVHI','SHLM','AAON','ABAX','XLRN']
#symbols_4 = ['ACTA','BIRT','MULT','YPRO','AEGR']
#symbols_5 = ['MDRX','EPAX','DOX','UHAL','MTGE']
#symbols_6 = ['CRMT','FOLD','BCOM','BOSC','HAWK']
#symbols_7 = ['CFFI','CHRW','KOOL','HOTR','PLCE']
#symbols_8 = ['JRJC','CHOP','HGSH','HTHT','IMOS']
#symbols_9 = ['DAEG','DJCO','SATS','WATT','INBK']
#symbols_10 = ['FTLB','QABA','GOOG', 'FUBC']
##nasdaq = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=" + symbol + "&outputsi ze=full&apikey=" + api_key
#
#
#API_URL = "https://www.alphavantage.co/query" 
#
#
##%%
#
#data, metadata= ts.get_daily('PIH',outputsize="full")
#data.rename(columns={'4. close': 'PIH'}, inplace=True)
#
#
#por_sp500 = data['PIH']
#
##%%
#
#for symbol in symbols_10:
#    data = quandl.get('WIKI/' + symbol)
#    data.rename(columns={'Close': symbol}, inplace=True)
#    pf = pd.concat([pf, data[symbol]], axis=1)
#    print(symbol)

#%%
#
#
#
portfolio = por_sp500.loc['2007-31-12':]

sp, sp_meta = ts.get_daily('^GSPC',outputsize="full")
sp_adj = sp.loc['2007-31-12':]
sp_adj['4. close'] = sp_adj['4. close'].astype(float)
sp_adj['squared'] = sp_adj['4. close']**2
sp_adj['root'] = sp_adj['4. close']**0.5
sp_adj.index = pd.DatetimeIndex(sp_adj.index).date


nasdaq, ns_meta = ts.get_daily('NDAQ',outputsize="full")
nasdaq_adj = nasdaq.loc['2007-31-12':]
nasdaq_adj.rename(columns={'4. close': 'nasdaq_close'}, inplace=True)
nasdaq_adj['squared_nq'] = nasdaq_adj['4. close']**2
nasdaq_adj['root_nq'] = nasdaq_adj['4. close']**0.5
nasdaq_adj.index = pd.DatetimeIndex(nasdaq_adj.index).date



oil = quandl.get('OPEC/ORB')
oil_adj = oil.loc['2007-12-31':] 
oil_adj['squared_oil'] = oil_adj['Value']**2
oil_adj['root_oil'] = oil_adj['Value']**0.5
oil_adj.index = pd.DatetimeIndex(oil_adj.index).date



treasury = quandl.get('USTREASURY/YIELD')
treas_adj = treasury.loc['2007-12-31':]
treas_adj['squared_tr'] = treas_adj['1 YR']**2
treas_adj['root_tr'] = treas_adj['1 YR']**0.5
treas_adj.index = pd.DatetimeIndex(treas_adj.index).date




a = quandl.get('WIKI/FCTY')

portfolio.to_csv('portfolio.csv', sep=';')

all_factors = sp_adj[['4. close', 'squared', 'root']]
all_factors = pd.merge(all_factors, nasdaq_adj[['nasdaq_close', 'squared_nq', 'root_nq']], how='outer',left_index= True, right_index =True)
all_factors = pd.merge(all_factors, oil_adj[['Value', 'squared_oil', 'root_oil']], how='outer',left_index= True, right_index =True)
all_factors = pd.merge(all_factors, treas_adj[['1 YR', 'squared_tr', 'root_tr']], how='outer',left_index= True, right_index =True)

all_factors.to_csv('all_factors.csv', sep=';')
