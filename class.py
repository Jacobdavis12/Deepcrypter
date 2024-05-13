# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 06:12:33 2024

@author: jacob
"""

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch.optim as optim

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)

## Load data
length = 100
classes = [substitution, vigenere, column]
X, Y = generateData(device, alphaspacelower, classes, generics_kb, length = length)

trainData, testData = classifierDataLoader(Y, oneHotEncoder, 0.8, device, length)
saveData(trainData, 'trainData')
saveData(testData, 'testData')
trainData = loadData('trainData', device)
testData = loadData('testData', device)

trainloader = DataLoader(trainData, batch_size=30)
print('loaded')

## train model
epochs = 30
batchSize = 30
classifier = Classifier(trainloader, 3).to(device)

losses = train(classifier, trainloader, epochs, batchSize)
saveLosses(losses, 'ggek')
losses = loadLosses('ggek')

saveModel(classifier, 'classifierTrans1')
classifier = loadModel('classifierTrans1', device)

t = np.arange(len(losses))

fig, ax = plt.subplots()
ax.plot(t, losses)

ax.set(xlabel='iteration', ylabel='loss',
       title='Loss plot')
ax.grid()

fig.savefig("lossClass1.png")
plt.show()

print('Finished Training')

#classifier = Classifier(102*length, 3).to(device)
#classifier.load_state_dict(torch.load('classifier.pth'))

##Test
classNames = [i.__name__ for i in classes]
confusionMatrix = classifier.test(testData, classNames)

plotConfusion(confusionMatrix, classNames)



print('end')