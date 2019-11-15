from _future_ import print_function
import numpy as np
import sys
from operator import add
from pyspark.sql import SparkSession, SQLContext
from pyspark import SparkContext, SparkConf

sc = spark.sparkContext
sqlContext = SQLContext(sc)

factorMeans = sqlContext.read.format('com.databricks.spark.csv').load("factorMeans.csv")
factorCov = sqlContext.read.format('com.databricks.spark.csv').load("factorCov.csv")
weights  = sqlContext.read.format('com.databricks.spark.csv').load("weights2.csv")


factorMeans = factorMeans.values.tolist()
weights = weights.values.tolist()

def fivePercentVaR(trials):
    numTrials = trials.count()
    topLosses = trials.takeOrdered(max(round(numTrials/20.0), 1))
    return topLosses[-1]


def sign(number):
    if number<0:
        return -1
    else:
        return 1

def featurize(factorReturns):
    factorReturns = list(factorReturns)
    squaredReturns = [sign(element)(element)*2 for element in factorReturns]
    squareRootedReturns = [sign(element)abs(element)*0.5 for element in factorReturns]
    # concat new features
    return squaredReturns + squareRootedReturns + factorReturns

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
            instrumentReturn = sum([stockWeights[i] * trialFeatures[i] for i in range(len(tria$
            trialTotalReturn += instrumentReturn
        
        trialReturns.append(trialTotalReturn)
    return trialReturns


simulateTrialReturns(1000,factorMeans,factorCov, weights)


if name == "main":
#    if len(sys.argv) != 3:
#        print("Usage: pageRank <urlFile> <iterations>", file=sys.stderr)
#        sys.exit(-1)
        
        
    
    spark = SparkSession\
        .builder\
        .appName("MonteCarlo")\
        .getOrCreate()

    sc = spark.sparkContext
    
    parallelism = 12
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

    print ("Value at Risk(VaR) 5%:", valueAtRisk)



spark.stop()
