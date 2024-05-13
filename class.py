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
from dataManager import generateData
from ciphers import genericSubstitution, vigenere, substitution, column
from utils import alphaspacelower, alphabetEmbedder, dealphabetEmbedder, oneHotEncoder,deOneHotEncoder, plotConfusion, loadData, saveData
from classModel import Classifier, classifierDataLoader

print('p')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)

## Load data
length = 1020
classes = ['substitution', 'vigenere', 'column']
# X, Y = generateData(device, alphaspacelower, [substitution, vigenere, column], length = length)

# # print(next(Y[0]))
# # print(next(Y[1]))
# # print(next(Y[2]))
# # print(X[0])
# # del X[0]

# ## format into dataloader
# x = [item for sublist in Y for item in sublist]
# label = np.arange(3)
# label = np.tile(label, (length, 1))
# label = label.T.flatten()

# trainData, testData = classifierDataLoader(x, label, oneHotEncoder, 0.8, device = device)
# saveData(trainData, 'trainData')
# saveData(testData, 'testData')
trainData = loadData('trainData')
testData = loadData('testData')
print('gg')

trainloader = DataLoader(trainData, batch_size=30)
print('loadedF')

## train model
classifier = Classifier(1056, 3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

losses = []
for epoch in range(15):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 500 == 499:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            losses.append([running_loss])
            running_loss = 0.0

torch.save(classifier.state_dict(), 'classifierTrans1.pth')

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
confusionMatrix = classifier.test(testData, classes)

plotConfusion(confusionMatrix, classes)



print('end')