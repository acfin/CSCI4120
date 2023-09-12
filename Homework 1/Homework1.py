# Example of kNN implemented from Scratch in Python

import pandas as pd
import random
import math
import operator
import matplotlib.pyplot as plt
import numpy as np

def loadDataset(filename, split):
    trainingSet = []
    testSet = []
    df = pd.read_csv(filename, header=None)
    array = df.to_numpy()
    random.shuffle(array)
    training_len = int(len(array) * split)
    trainingSet = array[:training_len]
    testSet = array[training_len:]
    return trainingSet, testSet

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

def main():
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    url = 'https://raw.githubusercontent.com/ruiwu1990/CSCI_4120/master/KNN/iris.data'
    # generate predictions
    predictions = []
    kAccuracies = []
    totalAccuracy = 0
    for k in range(20):
        totalAccuracy = 0
        for attempt in range(5):
            accuracy = 0
            trainingSet, testSet = loadDataset(url, split)
            #print('Train set: ' + repr(len(trainingSet)))
            #print('Test set: ' + repr(len(testSet)))
            for x in range(len(testSet)):
                # TODO starts here
                # get neighbor between current test record and all training datasets
                neighbors = getNeighbors(trainingSet, testSet[x], k+1)
                # get response
                result = getResponse(neighbors)
                # append current prediction result to predictions list
                predictions.append(result)
                # print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
                # TODO ends here
            accuracy = getAccuracy(testSet, predictions)
            totalAccuracy += accuracy
            print('Accuracy: ' + repr(accuracy) + '%')
        avgAccuracy = totalAccuracy/5
        print('Accuracy for k={} is {}'.format(k+1, avgAccuracy))
        kAccuracies.append(avgAccuracy)

    x = np.arange(1,21)
    plt.plot(x, kAccuracies)
    plt.show()

main()
