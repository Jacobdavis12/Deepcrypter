# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 06:12:33 2024

@author: jacob
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

import numpy as np


def classifierDataLoader(Y, embedder, split, device, length):
    X = [item for sublist in Y for item in sublist]
    label = np.arange(3)
    label = np.tile(label, (length, 1))
    label = label.T.flatten()

    X = embedder(X, 'qwertyuiopasdfghjklzxcvbnm ')
    data = TensorDataset(torch.unsqueeze(torch.LongTensor(X).to(device).float(), 1),
                         torch.LongTensor(label).to(device))
    train_size = int(split * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    
    print(len(train_dataset), len(test_dataset))
    return train_dataset, test_dataset

class Classifier(nn.Module):
    def __init__(self, trainloader, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        #Claclulate mid: hacky but works
        x = trainloader
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        midSize = x.shape[1]

        self.fc1 = nn.Linear(midSize, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def test(self, data, classes, testSize = 1000):
        testloader = DataLoader(data, batch_size=testSize)
        dataiter = iter(testloader)
        sentences, labels = next(dataiter)
        outputs = self(sentences)
        _, predicted = torch.max(outputs, 1)

        confusionMatrix = np.zeros((len(classes), len(classes)))
        for acc, pred in zip(labels, predicted):
            confusionMatrix[acc, pred] += 1

        confusionMatrix = (confusionMatrix / confusionMatrix.sum(axis=1)).round(decimals=2)

        return confusionMatrix

def train(classifier, trainloader, epochs, batchSize):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

    losses = []
    for epoch in range(epochs):  # loop over the dataset multiple times
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
        if True:# i % 50 == 49:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss /batchSize:.3f}')
            losses.append([running_loss/batchSize])
            running_loss = 0.0

    return losses