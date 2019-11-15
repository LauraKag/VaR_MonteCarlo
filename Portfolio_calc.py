#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:33:10 2019

@author: l-r-h
"""

#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
 
#%%

portfolio = pd.read_csv('/Users/l-r-h/Desktop/IE/GitHub/VaR/portfolio.csv', sep=';')
factors = pd.read_csv('/Users/l-r-h/Desktop/IE/GitHub/VaR/all_factors.csv', sep=';')
portfolio.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
factors.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

nv= portfolio.isnull().sum(axis=0)
nvr = portfolio.isnull().sum(axis=1)

#%%

nv2 = factors.isnull().sum(axis=0)

#%%
#fill factors
factors.fillna(method='bfill', inplace = True)

#%%
portfolio.fillna(method='ffill', inplace= True)
#%%

#plt.plot(factors_2['Date'], factors_2['SP500'], 'r')
#plt.plot(factors_2['Date'], factors_2['Nasdaq'], 'b')
#plt.plot(factors_2['Date'], factors_2['Oil'], 'g')
#plt.plot(factors_2['Date'], factors_2['Treasury'], 'c' )
#plt.show
#%%

factors['SP500'] = factors['4. close'] + factors['squared'] + factors['root']
factors['Nasdaq'] = factors['nasdaq_close'] + factors['squared_nq'] + factors['root_nq']
factors['Oil'] = factors['Value'] + factors['squared_oil'] + factors['root_oil']
factors['Treasury'] = factors['1 YR'] + factors['squared_tr'] + factors['root_tr']
#%%

factors_2 = factors[['SP500', 'Nasdaq', 'Oil', 'Treasury']].pct_change(4)

#portfolio_2 = portfolio.pct_change(4)
pf = portfolio.loc[:, portfolio.columns != 'Date']
pf_2 = pf.pct_change(4)
pf_2['Date'] = portfolio['Date']
factors_2['Date'] = factors['Date']

#%%

#Open questions_
#
#
#0. How to deal with imputing of missing values for Positions?
#
#1. change in percentage for all values and factors?
#1.1 Change before or after factorization?
#
#2. Model for each position: how can we automatize the process?
#%%

weights = np.array([1.0/48]*48)
cov_matrix = portfolio.cov()
avg_returns = portfolio.mean()
port_mean = avg_returns.dot(weights)
port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
#will work with clean dataset only, otherwise nan
#%%

all_data = pd.merge(pf_2, factors_2, how='inner', on=['Date'])

Var_95 = norm.ppf(0.05, port_mean, port_stdev)

#%%
#missing FUBC
factors = ['SP500', 'Nasdaq', 'Oil', 'Treasury']
symbols = ['FLWS', 'FCTY', 'FCCY', 'SRCE', 'VNET', 'TWOU',
       'DGLD', 'JOBS', 'EGHT', 'AVHI', 'SHLM', 'AAON', 'ABAX', 'XLRN',
       'ACTA', 'BIRT', 'MULT', 'YPRO', 'AEGR', 'MDRX', 'EPAX', 'DOX',
       'UHAL', 'MTGE', 'CRMT', 'FOLD', 'BCOM', 'BOSC', 'HAWK', 'CFFI',
       'CHRW', 'KOOL', 'HOTR', 'PLCE', 'JRJC', 'CHOP', 'HGSH', 'HTHT',
       'IMOS', 'DAEG', 'DJCO', 'SATS', 'WATT', 'INBK', 'FTLB', 'QABA', 'GOOG']


#'Linear Model' = linear_model.Ridge(alpha=10, max_iter=1000)
asd = all_data.GOOG.dropna()
df = all_data[np.isfinite(all_data['GOOG'])]

#%%
regression = []
def run_regression(factors, symbols, dataframe):
    for symbol in symbols:
        if dataframe[symbol].isnull().sum(axis=0) <= 420:
            dataframe[symbol].fillna(method='bfill', inplace = True)
            dataframe[symbol].fillna(method='ffill', inplace = True)
            
            X = dataframe[factors].values.reshape(-1,1)
            y = dataframe[symbol].values.reshape(-1,1)
            reg = LinearRegression()
            reg.fit(X, y)
            regression.append(reg.intercept_, reg.coef_)
        
        else:
           df_temp = all_data[np.isfinite(all_data[symbol])]
           X = df_temp[factors].values.reshape(-1,1)
           y = df_temp[symbol].values.reshape(-1,1)
           reg = LinearRegression()
           reg.fit(X, y)
           regression.append(reg.intercept_, reg.coef_)

#%%
          
#
X = factors
y = symbol
reg = LinearRegression()
reg.fit(X, y)
print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))



#%%
#We have assumed that the returns of factors follow a normal distribution.
#Show graphically that this is the case.

f = factors_2[['SP500', 'Nasdaq', 'Oil', 'Treasury']]
f.fillna(method='bfill', inplace = True)

def distribution(factors):
    for factor in factors:
        h=factors[factor].values
        h.sort()    
        hmean = np.mean(h)       
        hstd = np.std(h)
        pdf = stats.norm.pdf(h, hmean, hstd)
        plt.plot(h, pdf) 
        
distribution(f)













