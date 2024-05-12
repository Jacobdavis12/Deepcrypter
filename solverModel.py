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

import math
import time

import matplotlib

from transformerModel import embed, deembed

from utils import substitutionEmbedder, deSubstitutionEmbedder

def solverDataLoader(X, y, embedder, sos, eos, pad, MAX_LENGTH, device, split = 0.8):
    y = substitutionEmbedder(X, y)
    X = list(embed(list(X), embedder, sos, eos, pad, MAX_LENGTH))
    data = TensorDataset(torch.LongTensor(X).to(device),
                         torch.LongTensor(y).to(device))
    train_size = int(split * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    
    print(len(train_dataset), len(test_dataset))
    return train_dataset, test_dataset

#Loss function
class CustomLoss(nn.Module):
    def __init__(self): 
        super(CustomLoss, self).__init__() 
 
    def forward(self, predicted, target):
        crit = nn.CrossEntropyLoss()
        d = int(predicted.size()[1]**(1/2))
        predicted = predicted.reshape(predicted.size()[0], d, d)

        return crit(predicted, target)

def train(encoder, classifier, trainloader, epochs, batchSize):
    criterion = CustomLoss()
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
            encoder_outputs, _ = encoder(inputs)
            outputs = classifier(encoder_outputs.unsqueeze(1))
            loss = criterion(outputs, labels.to(float))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        if True:# i % 50 == 49:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss /batchSize:.3f}')
            losses.append([running_loss/batchSize])
            running_loss = 0.0

    return losses

def evaluate(encoder, classifier, trainloader, testSize = 100):
    dataiter = iter(trainloader)
    X, labels = next(dataiter)
    embedding, _ = encoder(X)
    output = classifier(embedding.unsqueeze(1))

    d = int(output.size()[1]**(1/2))

    output = output.reshape(output.size()[0], d, d)
    _, predicted = torch.max(output, 1)
    return deSubstitutionEmbedder(X, predicted)