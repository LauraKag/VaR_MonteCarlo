#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 23:40:06 2019

@author: l-r-h
"""

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
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
#%%
#importing data and initial cleaning of factors
portfolio = pd.read_csv('/Users/l-r-h/Desktop/IE/GitHub/VaR/portfolio.csv', sep=';')
factors = pd.read_csv('/Users/l-r-h/Desktop/IE/GitHub/VaR/all_factors.csv', sep=';')
portfolio.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
factors.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

factors.fillna(method='bfill', inplace = True)

#%%

#transforming from absolute values to returns
factors_returns = factors.loc[:, factors.columns != 'Date']
factors_returns = factors_returns.pct_change(4)
factors_returns['Date'] = factors['Date']

portfolio_returns = portfolio.loc[:, portfolio.columns != 'Date']
portfolio_returns = portfolio_returns.pct_change(4)
portfolio_returns['Date'] = portfolio['Date']
#merging of dataset
all_data = pd.merge(portfolio_returns, factors_returns, how='inner', on=['Date'])

#%%
#create list to iterate for regression function
factor_list = ['4. close', 'squared', 'root', 'nasdaq_close','squared_nq','root_nq','Value','squared_oil','root_oil','1 YR','squared_tr','root_tr']
symbol_list = ['FLWS', 'FCTY', 'FCCY', 'SRCE', 'VNET', 'TWOU',
       'DGLD', 'JOBS', 'EGHT', 'AVHI', 'SHLM', 'AAON', 'ABAX', 'XLRN',
       'ACTA', 'BIRT', 'MULT', 'YPRO', 'AEGR', 'MDRX', 'EPAX', 'DOX',
       'UHAL', 'MTGE', 'CRMT', 'FOLD', 'BCOM', 'BOSC', 'HAWK', 'CFFI',
       'CHRW', 'KOOL', 'HOTR', 'PLCE', 'JRJC', 'CHOP', 'HGSH', 'HTHT',
       'IMOS', 'DAEG', 'DJCO', 'SATS', 'WATT', 'INBK', 'FTLB', 'QABA', 'GOOG']





#%%
#get weights for regression on each item          
regression = []
def run_regression(dataframe,f=factor_list, s=symbol_list):
    for symbol in symbol_list:
        if dataframe[symbol].isnull().sum(axis=0) <= 420:
            dataframe[symbol].fillna(method='bfill', inplace = True)
            dataframe[symbol].fillna(method='ffill', inplace = True)
            
            X = dataframe[factor_list].values
            X1= np.where(np.isnan(X), 0, X)
            y = dataframe[symbol].values
            y1 = np.where(np.isnan(y), 0,y)
            reg = LinearRegression()
            reg.fit(X1, y1)
            regression.append(reg.coef_)
        
        else:
           df_temp = all_data[np.isfinite(all_data[symbol])]
           X = df_temp[factor_list].values
           X1= np.where(np.isnan(X), 0, X)
           y = df_temp[symbol].values
           y1 = np.where(np.isnan(y), 0,y)
           reg = LinearRegression()
           reg.fit(X1, y1)
           regression.append(reg.coef_)
           
#%% 
def sign(number):
    if number<0:
        return -1
    else:
        return 1           
           
def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def featurize(factorReturns):
    factorReturns = list(factorReturns)
    squaredReturns = [sign(i)*(i)**2 for i in factorReturns]
    squareRootedReturns = [sign(i)*abs(i)**0.5 for i in factorReturns]
    # concat new features
    return squaredReturns + squareRootedReturns + factorReturns
#%%
#from itertools import izip  
    
def mean_square(computedValues, realValues):
#    real = tuple(realValues)
    zipped = list(zip(computedValues, realValues))
    zipped_2 = np.where(np.isnan(zipped), 0,zipped)
    squares = [np.subtract(x[0],x[1])**2 for x in zipped_2]
    
#    for x in squares:
#        x=x.replace('nan', 0)
    return np.sum(squares)/len(squares)

#%%
test = list(zip(((predict(factor_columns['Value'], regression[4])), list(symbol_columns[4]))))
#%%
sq1 = (np.subtract(test[0],test[1]))**2
squares = [(x[0] - x[1])**2 for x in test]
squares.isna(0)
#%%
for x in squares:
    x=x.replace('nan', 0)
res = sum(squares)/len(squares)
#def predict(feat, weights):
#    l = [feat.iloc[i]*weights[j] for i in range(len(feat))]
#    return sum(l)
#%%
    

def predict(feat, weights):
    l = []
    for i in range(len(feat)):
        l.append(sum(feat.iloc[i]*weights))
#    for j in l:
#        ll.append(sum(j))
    return l
    

#%%
#transpose matrix of regression values    
factorMat = transpose(regression)      

factorFeatures = list(map(featurize,factorMat))     


factor_columns = sm.add_constant(all_data[factor_list], prepend=True)
symbol_columns = sm.add_constant(all_data[symbol_list], prepend=True)
#%%
symbol_columns = symbol_columns.rename(columns= {
        'FLWS': 0, 
        'FCTY':1,
        'FCCY':2, 
        'SRCE':3, 
        'VNET':4, 
        'TWOU':5,
        'DGLD':6, 
        'JOBS':7, 
        'EGHT':8, 
        'AVHI':9, 
        'SHLM':10, 
        'AAON':11, 
        'ABAX':12, 
        'XLRN':13,
        'ACTA':14, 
        'BIRT':15, 
        'MULT':16, 
        'YPRO':17, 
        'AEGR':18, 
        'MDRX':19, 
        'EPAX':20, 
        'DOX':21,
        'UHAL':22, 
        'MTGE':23, 
        'CRMT':24, 
        'FOLD':25, 
        'BCOM':26, 
        'BOSC':27, 
        'HAWK':28, 
        'CFFI':29,
        'CHRW':30, 
        'KOOL':31, 
        'HOTR':32, 
        'PLCE':33, 
        'JRJC':34, 
        'CHOP':35, 
        'HGSH':36, 
        'HTHT':37,
        'IMOS':38, 
        'DAEG':39, 
        'DJCO':40, 
        'SATS':41, 
        'WATT':42, 
        'INBK':43, 
        'FTLB':44, 
        'QABA':45, 
        'GOOG':46
        })
           


#%%


##j = 0
##i=0
#for i in range(len(all_data[factor_list])):
##while i< len(all_data[factor_list]):
#    for j in range(len(factor_list)):
#
##    while j < len(factor_list):
#    l.append(all_data[factor_list].iloc[i]*regression)
# 

del symbol_columns['const']
del factor_columns['const']
factor_columns.dropna(0)
    
#%%       

listOfMeanSquares = []
listOfVariances = []    
for j in range(len(regression)):
    
    for i in factor_columns:
        predictions = predict(factor_columns[i],regression[j])
    
#    predictions = map(lambda x: predict(x, regression[j]), factor_columns) 
        meanError = mean_square(predictions, symbol_columns[j])
        listOfMeanSquares.append(meanError)
        varianceSquare = np.std(symbol_columns[j])**2
        listOfVariances.append(varianceSquare)
    
plt.bar(range(len(listOfMeanSquares)), listOfMeanSquares, alpha=0.4, align="center")
plt.title("Figure 5.1: Mean square error by stock")
plt.xlabel("index of stock")
plt.ylabel("Mean square error")
plt.xlim(0, len(listOfMeanSquares))
plt.show()
    
plt.bar(range(len(listOfVariances)), listOfVariances, alpha=0.4, align="center")
plt.title("Figure 5.2: Variance by stock")
plt.xlabel("index of stock")
plt.ylabel("Variance")
plt.xlim(0, len(listOfVariances))
plt.show()


#%%

def plotPredictionsVSReals(nStock):
    predictions = map(lambda x: predict(factor_columns, regression[nStock]),factor_columns)
    plt.figure(figsize=(20,10))
    plt.plot(list(predictions), lw=2)
    plt.plot(symbol_columns[nStock], lw=2)
    plt.legend(["predictions", "real"], fontsize="xx-large")
    plt.xlabel("2 weeks window") 
    plt.ylabel("variation over the window") 
    plt.title("Predictions vs Real Stock n{}".format(nStock))
    plt.xlim(0, len(symbol_columns[nStock]))
    plt.show()


#%%

import matplotlib.pyplot as plt
s = factor_columns.dropna(0)

def plotDistribution(samples, plot=True, numSamples=100):
    samples.dropna(0)
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

plotDistribution(s['4. close'])
plotDistribution(s['nasdaq_close'])
plotDistribution(s['Value'])
plotDistribution(s['1 YR'])
#%%

def plotDistribution_2(samples, plot=True, numSamples=100):
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


#%%
correlation = s.corr()
r = s[['4. close', 'nasdaq_close', 'Value', '1 YR']]
corr_2 = r.corr()


#%%

factorCov = s.cov()
factorMeans = [np.sum(s[factor])/len(s[factor]) for factor in s]
sample = np.random.multivariate_normal(factorMeans, factorCov)
print(factorCov)
print(factorMeans)
print(sample)

#%%


factorsNames = factor_list 
    
numSamples = 50000 # to plot normal distributions
f, axarr = plt.subplots(2, 2)
f.set_figwidth(20)
f.set_figheight(10)
for (idx, factorReturn) in enumerate(r):
    i, j = divmod(idx, 2)
    ax = axarr[i, j]
    normalEstimates = [np.random.multivariate_normal(factorMeans, factorCov)[idx] for k in range(numSamples)]
    domainEstimates, densityEstimates = plotDistribution_2(normalEstimates, plot=False)
#    r.astype(float)
    domainFactor, densityFactor = plotDistribution_2(r[factorReturn], plot=False)
    ax.plot(domainEstimates, densityEstimates, lw=2)
    ax.plot(domainFactor, densityFactor, lw=2)
    ax.legend(["estimate", "real"], fontsize="xx-large")
f.subplots_adjust(hspace=0.3)


#normalEstimates = [np.random.multivariate_normal(factorMeans, factorCov)[idx] for k in range(numSamples)]
#domainEstimates, densityEstimates = plotDistribution_2(normalEstimates, plot=False)
#domainFactor, densityFactor = plotDistribution_2(r, plot=False)


#%%

factorsNames = ['SP500', 'Nasdaq', 'Oil', 'Treasury']
    
numSamples = 50000 # to plot normal distributions
f, axarr = plt.subplots(2, 2)
f.set_figwidth(20)
f.set_figheight(10)

for idx, column in enumerate(r):
    i, j = divmod(idx, 2)
    ax = axarr[i, j]
    normalEstimates = [np.random.multivariate_normal(factorMeans, factorCov)[idx] for k in range(numSamples)]
    domainEstimates, densityEstimates = plotDistribution_2(normalEstimates, plot=False)
    domainFactor, densityFactor = plotDistribution_2(r.values, plot=False)
    ax.plot(domainEstimates, densityEstimates, lw=2)
    ax.plot(domainFactor, densityFactor, lw=2)
    
f.subplots_adjust(hspace=0.3)

#%%
numTrials = 100

def simulateTrialReturns(numTrials, factorMeans, factorCov, regression):
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
        for stockWeights in regression:
            instrumentReturn = sum([stockWeights[i] * trialFeatures[i] for i in range(len(trialFeatures))])
            trialTotalReturn += instrumentReturn
        
        trialReturns.append(trialTotalReturn)
    print(trialReturns)


#parallelism = 2
#
#trial_indexes = list(range(0, parallelism))
#seedRDD = sc.parallelize(trial_indexes, parallelism)
#bFactorWeights = sc.broadcast(weights)
#
#trials = seedRDD.flatMap(lambda idx: \
#                simulateTrialReturns(
#                    max(int(numTrials/parallelism), 1), 
#                    factorMeans, factorCov,
#                    bFactorWeights.value
#                ))
#trials.cache()
#
#valueAtRisk = fivePercentVaR(trials)
#conditionalValueAtRisk = fivePercentCVaR(trials)
#
#print ("Value at Risk(VaR) 5%:", valueAtRisk)
#print ("Conditional Value at Risk(CVaR) 5%:", conditionalValueAtRisk)


#%%

trialFactorReturns = np.random.multivariate_normal(factorMeans, factorCov)

# featurize the factors' returns
trialFeatures = featurize(trialFactorReturns)

# insert weight for intercept term
trialFeatures.insert(0,1)

trialTotalReturn = 0
#%%
# calculate the return of each instrument
# then calulate the total of return for this trial features
for stockWeights in regression:
    instrumentReturn = sum([stockWeights[i] * trialFactorReturns[i] for i in range(len(trialFactorReturns))])
    trialTotalReturn += instrumentReturn

#trialReturns.append(trialTotalReturn)
print(trialTotalReturn)
























