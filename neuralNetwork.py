import numpy as np
import matplotlib.pyplot as plt
from loadData import load
from sklearn.neural_network import MLPClassifier

def neuralNetworkTrain(trainData, trainLabels, numHiddenNodes, activationFcn="relu", learningRate=0.0001, regularization=0.0001):
  model = MLPClassifier(hidden_layer_sizes=(numHiddenNodes,), random_state=1, activation=activationFcn, learning_rate_init=learningRate, alpha=regularization)
  model.fit(trainData, np.array(trainLabels).ravel())
  return model

def neuralNetworkEvaluate(model, data, labels):
  predictions = model.predict(data)
  errors = (predictions != np.array(labels).ravel())
  return str(100 * np.mean(errors))


if __name__ == '__main__':
  x = []
  trainErrors = []

  trainData, trainLabels, testData, testLabels = load()
  for numHiddenNodes in xrange(5, 300, 5):
    model = neuralNetworkTrain(trainData, trainLabels, numHiddenNodes)

    trainError = neuralNetworkEvaluate(model, trainData, trainLabels)
    print "Training Error: " + trainError

    testError = neuralNetworkEvaluate(model, testData, testLabels)
    print "Test Error: " + testError

    x.append(numHiddenNodes)
    trainErrors.append(trainError)

  plt.plot(x, trainErrors)
  plt.show()
