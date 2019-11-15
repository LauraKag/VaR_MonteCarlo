
from pyspark import SparkContext
from pyspark.conf import SparkConf

conf = SparkConf()
conf.setMaster("local[4]")
conf.setAppName("MC") #we can change the name here
sc = SparkContext(conf=conf)
sc.addFile("Final.py") #here should be the name of our file

import Final.py #here same as before


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

