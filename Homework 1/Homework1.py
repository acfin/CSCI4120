import pandas as pd
import numpy as np
import random
import math
import operator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



def loadDataset(url, split):
    trainingSet=[]
    testSet=[]
    df = pd.read_csv(url, header=None)
    array = df.to_numpy()
    random.shuffle(array)
    training_len = int(len(array)*split)
    trainingSet = array[:training_len]
    testSet = array[training_len:]
    return trainingSet, testSet

"""
This function translates the target strings into integers.
targets: a list containing the iris data set
targetNames: a list of the iris target types
targetIndex: the index of the target name string in the targets list
"""
def parseTargets(targets, targetNames, targetIndex):
    
    if(len(targetNames) >= 2):
        for target in targets:
            if(target[targetIndex] == targetNames[0]):
                target[targetIndex] = 0
            elif(target[targetIndex] == targetNames[1]):
                target[targetIndex] = 1
            elif(target[targetIndex] == targetNames[2]):
                target[targetIndex] = 2
            else:
                print(target)
    else:
        print("Target names size error")
    
    return [i[4] for i in targets]
    
def classifier(k):
    # prepare data
    trainingSet=[]
    testSet=[]
    split = 0.67
    url = 'https://raw.githubusercontent.com/ruiwu1990/CSCI_4120/master/KNN/iris.data'
    print("\tLoading dataset...")
    trainingSet, testSet = loadDataset(url, 0.66)
    #print('Train set: ' + repr(len(trainingSet)))
    #print('Test set: ' + repr(len(testSet)))
    
    targetNames = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    print("\tPreparing training data...")
    trainX = [i[0:4] for i in trainingSet]
    trainY = parseTargets(trainingSet, targetNames, 4)
    print("\tPreparing testing data...")
    testX = [i[0:4] for i in testSet]
    testY = parseTargets(testSet, targetNames, 4)
    
    print("\tGenerating predictions...")
    # generate predictions
    predictions=[]
    result = KNeighborsClassifier(n_neighbors = k)
    result.fit(trainX, trainY)
    predictions = result.predict(testX)
    print("\tComputing accuracy...")
    accuracy = accuracy_score(predictions, testY)
    print('\tAccuracy: ' + repr(np.around(accuracy,4)*10*10) + '%')
    return accuracy
    
def main():
    kAccuracies = []
    totalAccuracy = 0
    highestK = 20
    cyclesPerK = 5
    for k in range(1,highestK+1):
        print("Running classifier for k = " + str(k))
        totalAccuracy = 0
        for attempt in range(cyclesPerK):
            accuracy = 0
            accuracy = classifier(k)
            totalAccuracy += accuracy
            #print('Accuracy: ' + repr(accuracy) + '%')
        avgAccuracy = totalAccuracy/cyclesPerK
        print('\tAccuracy for k={} is {}'.format(k, avgAccuracy))
        kAccuracies.append(avgAccuracy)

    x = np.arange(1,highestK+1)
    plt.plot(x, kAccuracies)
    plt.xticks(range(1,highestK+1))
    plt.show()

main()