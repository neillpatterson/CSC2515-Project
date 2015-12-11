import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier as LR
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier

def load(synthetic=False):
  if (not synthetic):
    train = pd.read_csv('train/X_train.txt', header=None, delim_whitespace=True)
    test = pd.read_csv('test/X_test.txt', header=None, delim_whitespace=True)

    label = pd.read_csv('train/y_train.txt', header=None, delim_whitespace=True)
    test_label = pd.read_csv('test/y_test.txt', header=None, delim_whitespace=True)

    return train, label, test, test_label

  mu1 = np.array([0, 1])
  mu2 = np.array([-0.5, -1.0])
  mu3 = np.array([3.2, .6])
  mu4 = np.array([3, -1])

  s1 = np.matrix([[1, 0.1], [0.1, 0.1]])
  s2 = np.matrix([[0.3, 0.2], [0.2, 0.4]])
  s3 = np.matrix([[0.5, 0.01], [0.01, 0.1]])
  s4 = np.matrix([[0.5, -0.2], [-0.2, 0.2]])

  d1 = multivariate_normal(mu1, s1, 500)
  d2 = multivariate_normal(mu2, s2, 500)
  d3 = multivariate_normal(mu3, s3, 500)
  d4 = multivariate_normal(mu4, s4, 500)
  syntheticTrain = np.vstack((d1,d2,d3,d4))
  syntheticTrainLabel = np.hstack(([1] * 500, [2] * 500, [3] * 500, [4] * 500))


  t1 = multivariate_normal(mu1, s1, 500)
  t2 = multivariate_normal(mu2, s2, 500)
  t3 = multivariate_normal(mu3, s3, 500)
  t4 = multivariate_normal(mu4, s4, 500)
  syntheticTest = np.vstack((t1,t2,t3,t4))
  syntheticTestLabel = np.hstack(([1] * 500, [2] * 500, [3] * 500, [4] * 500))

  fig, (ax1) = plt.subplots(1,1)

  ax1.scatter(d1[:,0], d1[:,1], color='red')
  ax1.scatter(d2[:,0], d2[:,1], color='blue')
  ax1.scatter(d3[:,0], d3[:,1], color='green')
  ax1.scatter(d4[:,0], d4[:,1], color='black')

  return syntheticTrain, syntheticTrainLabel, syntheticTest, syntheticTestLabel
