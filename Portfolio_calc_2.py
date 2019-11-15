#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:36:50 2019

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
# imports
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from datetime import timedelta
from itertools import islice
import statsmodels.api as sm
from os import listdir
from os.path import isfile, join
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from statsmodels.nonparametric.kde import KDEUnivariate
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import math
import statistics
import itertools



#%%

portfolio = pd.read_csv('C:/Users/laura/OneDrive/Desktop/DataScience/portfolio.csv', sep=';')
factors = pd.read_csv('C:/Users/laura/OneDrive/Desktop/DataScience/all_factors.csv', sep=';')

portfolio.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
factors.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

#%% 
nv= portfolio.isnull().sum(axis=0)
nvr = portfolio.isnull().sum(axis=1)
nv2 = factors.isnull().sum(axis=0)


#%%
#fill factors Nan Values 
factors.fillna(method='bfill', inplace = True)


#%%
#portfolio.fillna(method='ffill', inplace= True)
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

weights = np.array([1.0/48]*48)
cov_matrix = portfolio.cov()
avg_returns = portfolio.mean()
port_mean = avg_returns.dot(weights)
port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
Var_95 = norm.ppf(0.05, port_mean, port_stdev)

#will work with clean dataset only, otherwise nan
#%%

all_data = pd.merge(pf_2, factors_2, how='inner', on=['Date'])


#%%
factors = ['SP500', 'Nasdaq', 'Oil', 'Treasury']
symbols = ['FLWS', 'FCTY', 'FCCY', 'SRCE', 'VNET', 'TWOU',
       'DGLD', 'JOBS', 'EGHT', 'AVHI', 'SHLM', 'AAON', 'ABAX', 'XLRN',
       'ACTA', 'BIRT', 'MULT', 'YPRO', 'AEGR', 'MDRX', 'EPAX', 'DOX',
       'UHAL', 'MTGE', 'CRMT', 'FOLD', 'BCOM', 'BOSC', 'HAWK', 'CFFI',
       'CHRW', 'KOOL', 'HOTR', 'PLCE', 'JRJC', 'CHOP', 'HGSH', 'HTHT',
       'IMOS', 'DAEG', 'DJCO', 'SATS', 'WATT', 'INBK', 'FTLB', 'QABA', 'GOOG']
#missing FUBC



#'Linear Model' = linear_model.Ridge(alpha=10, max_iter=1000)
#asd = all_data.GOOG.dropna()
#df = all_data[np.isfinite(all_data['GOOG'])]




#%%
          
regression = []
def run_regression(factors, symbols, dataframe):
    for symbol in symbols:
        if dataframe[symbol].isnull().sum(axis=0) <= 420:
            dataframe[symbol].fillna(method='bfill', inplace = True)
            dataframe[symbol].fillna(method='ffill', inplace = True)
            
            X = dataframe[factors].values
            X1= np.where(np.isnan(X), 0, X)
            y = dataframe[symbol].values
            y1 = np.where(np.isnan(y), 0,y)
            reg = LinearRegression()
            reg.fit(X1, y1)
            regression.append(reg.coef_)
        
        else:
           df_temp = all_data[np.isfinite(all_data[symbol])]
           X = df_temp[factors].values
           X1= np.where(np.isnan(X), 0, X)
           y = df_temp[symbol].values
           y1 = np.where(np.isnan(y), 0,y)
           reg = LinearRegression()
           reg.fit(X1, y1)
           regression.append(reg.coef_)
   
#%%  Individual Distribution 
           
import numpy as np



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
        
#%%      
factors_new1=factors_2[['4. close', 'nasdaq_close', 'Value', '1 YR']]
#factors_new1=factors_new.pct_change(4)
factors_new1.fillna(method='ffill', inplace = True)
factors_new1.fillna(method='bfill', inplace = True)
        
           
#%%
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from statsmodels.nonparametric.kde import KDEUnivariate
import matplotlib.pyplot as plt
import scipy

def plotDistribution(samples, plot=True, numSamples=100):
    vmin = min(samples)
    vmax = max(samples)
    stddev = np.std(samples)
    
    domain = np.arange(vmin, vmax, (vmax-vmin)/numSamples)
    
    # a simple heuristic to select bandwidth
    bandwidth = 1.06 * stddev * pow(len(samples), -.2)
    
    # estimate density
    kde = KDEUnivariate(samples)
    kde.fit(bw=bandwidth)
    density = kde.evaluate(domain)
    
    # plot
    # We do a little change because later we will use the data but plot it after
    if(plot):
        plt.plot(domain, density)
        plt.show()
    else:
        return domain,density

plotDistribution(factors_new1['4. close'])
plotDistribution(factors_new1['nasdaq_close'])
plotDistribution(factors_new1['Value'])
plotDistribution(factors_new1['1 YR'])

#%%
factorCov=factors_new1.cov()

factor1Mean=sum(factors_new1['4. close'])/len(factors_new1['4. close'])
factor2Mean=sum(factors_new1['nasdaq_close'])/len(factors_new1['nasdaq_close'])
factor3Mean=sum(factors_new1['Value'])/len(factors_new1['Value'])
factor4Mean=sum(factors_new1['1 YR'])/len(factors_new1['1 YR'])
factorMeans = [factor1Mean,factor2Mean,factor3Mean,factor4Mean]

sample = np.random.multivariate_normal(factorMeans, factorCov)
print(factorCov)
print(factorMeans)
print(sample)


#%%
# Compare our normal estimates and the real distributions 
factorsNames = ['SP500', 'Nasdaq', 'Oil', 'Treasury']
    
numSamples = 50000 # to plot normal distributions
f, axarr = plt.subplots(2, 2)
f.set_figwidth(20)
f.set_figheight(10)

for (idx, column) in enumerate(factors_new1):
    i, j = divmod(idx, 2)
    ax = axarr[i, j]
    normalEstimates = [np.random.multivariate_normal(factorMeans, factorCov)[idx] for k in range(numSamples)]
    domainEstimates, densityEstimates = plotDistribution(normalEstimates, plot=False)
    domainFactor, densityFactor = plotDistribution(factors_new1[column].values, plot=False)
    ax.plot(domainEstimates, densityEstimates, lw=2)
    ax.plot(domainFactor, densityFactor, lw=2)
    
f.subplots_adjust(hspace=0.3)          
           

#%%
import time
def fivePercentVaR(trials):
    numTrials = trials.count()
    topLosses = trials.takeOrdered(max(round(numTrials/20.0), 1))
    return topLosses[-1]

# an extension of VaR
def fivePercentCVaR(trials):
    numTrials = trials.count()
    topLosses = trials.takeOrdered(max(round(numTrials/20.0), 1))
    return sum(topLosses)/len(topLosses)

def bootstrappedConfidenceInterval(
      trials, computeStatisticFunction,
      numResamples, pValue):
    stats = []
    t = time.time()
    for i in range(0, numResamples):
        # This is just to know when it'll be finished when it runs on a laptop.
        update(i,numResamples,t)
        
        resample = trials.sample(True, 1.0)
        stats.append(computeStatisticFunction(resample))
        
    sorted(stats)
    lowerIndex = int(numResamples * pValue / 2 - 1)
    upperIndex = int(np.ceil(numResamples * (1 - pValue / 2)))
    return (stats[lowerIndex], stats[upperIndex])

#%%

def simulateTrialReturns(numTrials, factorMeans, factorCov, weights):
    trialReturns = []
    for i in range(0, numTrials):
        # generate sample of factors' returns
        trialFactorReturns = np.random.multivariate_normal(factorMeans, factorCov)
        
        # featurize the factors' returns
        trialFeatures = featurize(trialFactorReturns)
        
        # insert weight for intercept term
        trialFeatures.insert(0,1)
        
        trialTotalReturn = 0
        
        # calculate the return of each instrument
        # then calulate the total of return for this trial features
        for stockWeights in weights:
            instrumentReturn = sum([stockWeights[i] * trialFeatures[i] for i in range(len(trialFeatures))])
            trialTotalReturn += instrumentReturn
        
        trialReturns.append(trialTotalReturn)
    return trialReturns


parallelism = 2
numTrials = 10000
trial_indexes = list(range(0, parallelism))
seedRDD = sc.parallelize(trial_indexes, parallelism)
bFactorWeights = sc.broadcast(weights)

trials = seedRDD.flatMap(lambda idx: \
                simulateTrialReturns(
                    max(int(numTrials/parallelism), 1), 
                    factorMeans, factorCov,
                    bFactorWeights.value
                ))
trials.cache()

valueAtRisk = fivePercentVaR(trials)
conditionalValueAtRisk = fivePercentCVaR(trials)

print ("Value at Risk(VaR) 5%:", valueAtRisk)
print ("Conditional Value at Risk(CVaR) 5%:", conditionalValueAtRisk)

#%%