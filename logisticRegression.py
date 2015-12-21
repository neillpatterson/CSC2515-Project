import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loadData import load
from sklearn.linear_model import LogisticRegression as LR


def logisticRegressionTrain(trainData, trainLabels, solver="liblinear"):
  model = LR(solver=solver)
  model.fit(trainData, np.array(trainLabels).ravel())
  return model

def logisticRegressionEvaluate(model, data, labels):
  predictions = model.predict(data)
  errors = (predictions != np.array(labels).ravel())
  return 100 * np.mean(errors)


if __name__ == '__main__':
  trainData, trainLabels, testData, testLabels = load()
  model = logisticRegressionTrain(trainData, trainLabels)

  trainError = logisticRegressionEvaluate(model, trainData, trainLabels)
  print "Training Error: " + trainError

  testError = logisticRegressionEvaluate(model, testData, testLabels)
  print "Test Error: " + testError
