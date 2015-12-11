import numpy as np
from loadData import load
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier as SGDC

def adaBoostTrain(trainData, trainLabels, nClassifiers = 10000):
  nTrain = trainData.shape[0]
  alphas = np.zeros(nClassifiers)
  dataWeights = np.ones((nTrain, nClassifiers)) / float(nTrain)
  models = []

  for iteration in xrange(nClassifiers):
    #model = LR()
    model = DecisionTreeClassifier(max_depth=2)
    model.fit(trainData, np.array(trainLabels).ravel(), sample_weight=dataWeights[:, iteration]*nTrain)
    models.append(model)
    predict = model.predict(trainData)
    incorrect = predict != np.array(trainLabels).ravel()
    correct = np.invert(incorrect)
    weightedErrors = (incorrect) * dataWeights[:,iteration]
    errorRate = np.sum(weightedErrors)
    print errorRate * 100
    alphas[iteration] = np.log((1 - errorRate) / errorRate) + np.log(5)
    if (iteration < nClassifiers - 1):
      dataWeights[:, iteration + 1] = dataWeights[:, iteration] * np.exp(alphas[iteration] * incorrect)
      dataWeights[:, iteration + 1] /= np.sum(dataWeights[:, iteration + 1])

  return models, alphas

def adaBoostEvaluate(models, alphas, data, labels):
  # There are 6 possible classes. We need to keep a tally
  # For each class for each data point
  tallies = np.zeros((data.shape[0], 6))
  for i in xrange(len(alphas)):
      modelPredictions = models[i].predict(data)

      for j in xrange(len(modelPredictions)):
          prediction = modelPredictions[j]
          tallies[j, prediction - 1] += alphas[i]

  predictions = np.add(np.argmax(tallies, axis=1), 1)
  errors = (predictions != np.array(labels).ravel())
  return str(100 * np.mean(errors))



if __name__ == '__main__':
  trainData, trainLabels, testData, testLabels = load()
  models, alphas = adaBoostTrain(trainData, trainLabels)

  trainError = adaBoostEvaluate(models, alphas, trainData, trainLabels)
  print "Training Error: " + trainError

  testError = adaBoostEvaluate(models, alphas, testData, testLabels)
  print "Test Error: " + testError
