# -*- coding: utf-8 -*-
"""
Created on Sun May 12 19:57:23 2024

@author: jacob
"""
import os

#Local imports
from dataManager import generateData, generics_kb, bookcorpus
from ciphers import genericSubstitution, vigenere, substitution, column
from classModel import classifierDataLoader
from utils import alphaspacelower, oneHotEncoder, saveData, loadData
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch.optim as optim
import torch

import numpy as np

import time
import math

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib as mpl


#Local Modules
from dataManager import generateData, bookcorpus, generics_kb
from ciphers import genericSubstitution, vigenere, substitution, column
from utils import alphaspacelower, alphabetEmbedder, dealphabetEmbedder, oneHotEncoder,deOneHotEncoder, plotConfusion, loadData, saveData, saveLosses, loadLosses, saveModel, loadModel
from classModel import Classifier, classifierDataLoader, train


#Takes a range of hyper parameters and provides loss plots, models and datsets
def testSolver(device,\
               lengthRange, dataSetRange, embedderRange,\
               modelRange\
               ):
    dataNames = os.listdir('data')
    modelNames = os.listdir('models')

    for length in lengthRange:
        for dataset in dataSetRange:
            for embedder in embedderRange:
                for cipher in cipherRange:
                    hashData(length, dataset, embedder, cipher)
                    #Generate data
                    Y, X = generateData(device, embedder, [cipher], dataset, length = length)
                    for modelDataLoader, model in modelRange:
                        hashData(length, dataset, embedder, cipher)


def hashData(data):
    return str(data)

def genData(device,\
            lengthRange, dataSetRange, cleanerRange,\
            noiseRange0, noiseRange1,\
            classes):
    dataNames = os.listdir('data')
    modelNames = os.listdir('models')

    classes = [substitution, vigenere, column]
    for length in lengthRange:
        for dataset in dataSetRange:
            for cleaner in cleanerRange:
                for noise0 in noiseRange0:
                    for noise1 in noiseRange1:
                        X, Y = generateData(device, alphaspacelower, classes, generics_kb, noise = [noise0, noise1], length = length)
        
                        trainData, testData = classifierDataLoader(Y, oneHotEncoder, 0.8, device, length)
                        name = '!'.join([str(length), dataset.__name__, cleaner.__name__, str(noise0), str(noise1)])
                        saveData(trainData, 'train!classifier!' + name)
                        saveData(testData, 'test!classifier!' + name)

#Takes a range of hyper parameters and provides loss plots, models and datsets
def testClassifier(device,\
            lengthRange, dataSetRange, cleanerRange,\
            noiseRange0, noiseRange1,\
            classes):
    classNames = [i.__name__ for i in classes]
    for length in lengthRange:
        for dataset in dataSetRange:
            for cleaner in cleanerRange:
                for noise0 in noiseRange0:
                    for noise1 in noiseRange1:
                        name = '!'.join([str(length), dataset.__name__, cleaner.__name__, str(noise0), str(noise1)])
                        trainData = loadData('train!classifier!' + name, device)
                        testData = loadData('test!classifier!' + name, device)

                        if len(trainData) <= 300:
                            epochs = 300
                        elif len(trainData) <= 5000:
                            epochs = 100
                        else:
                            epochs = 30
                        batchSize = 30
                        trainloader = DataLoader(trainData, batch_size=batchSize)
                        classifier = Classifier(trainloader, 3).to(device)

                        losses = train(classifier, trainloader, epochs, batchSize)
                        saveLosses(losses, 'classifier!' + name)
                        saveModel(classifier, 'classifier!' + name)

                        classNames = [i.__name__ for i in classes]
                        confusionMatrix = classifier.test(testData, classNames)

                        plotConfusion(confusionMatrix, classNames, save = 'plots\classifier!' + name)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)


noiseRange0 = [0, 0.1, 0.3]
noiseRange1 = [0, 0.1, 0.3]
lengthRange = [100, 1000, 10000]
dataSetRange = [generics_kb, bookcorpus]
hiddenSizeRange = [8, 16, 32, 64]
cipherRange = [genericSubstitution, vigenere, substitution, column]
cleanerRange = [alphaspacelower]

classes = [substitution, vigenere, column]
# genData('cpu',\
#             lengthRange, dataSetRange, cleanerRange,
#             noiseRange0, noiseRange1,
#             classes)

testClassifier('cpu',\
            lengthRange, dataSetRange, cleanerRange,
            noiseRange0, noiseRange1,
            classes)


