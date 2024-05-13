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

import networkx as nx

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
def maximal_matching(probs):
    res = torch.zeros_like(probs)
    for i, probsi in enumerate(probs):
        # Create a bipartite graph
        G = nx.Graph()
        
        # Add nodes for the rows and columns
        rows = range(27)
        cols = range(27, 27+27, 1)
        G.add_nodes_from(rows, bipartite=0)
        G.add_nodes_from(cols, bipartite=1)
        
        # Add edges with weights equal to the probabilities
        for j in rows:
            for k in cols:
                prob = probsi[j][k-27]
                G.add_edge(j, k, weight=float(prob))
        
        # Find the maximum weighted matching
        match = nx.max_weight_matching(G, maxcardinality=True)
        try:
            matching = {i[1]:i[0] for i in match} | {i[0]:i[1] for i in match}
            # Print the mapping of rows to columns
            indexr = [matching[j]-27 for j in rows]
            indexc = [matching[j] for j in cols]
            res[i, indexc, rows] = probs[i, indexc, rows]
        except:
            print('bread')

    return res

class MatchingLoss(nn.Module):
    def __init__(self): 
        super(MatchingLoss, self).__init__()
 
    def forward(self, predicted, target):
        crit = nn.CrossEntropyLoss()
        
        d = int(predicted.size()[1]**(1/2))
        predicted = predicted.reshape(predicted.size()[0], d, d)
        predicted = maximal_matching(predicted)

        return crit(predicted, target)

class CustomLoss(nn.Module):
    def __init__(self): 
        super(CustomLoss, self).__init__() 
 
    def forward(self, predicted, target):
        crit = nn.CrossEntropyLoss()
        d = int(predicted.size()[1]**(1/2))
        predicted = predicted.reshape(predicted.size()[0], d, d)

        return crit(predicted, target)

class JointLoss(nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()
        self.loss1 = CustomLoss()
        self.loss2 = MatchingLoss()

    def forward(self, predicted, target, i):
        if (i+1)%10 == 0:
            return self.loss1(predicted, target)*5/(i+1) + self.loss2(predicted, target)/2
        else:
            return self.loss1(predicted, target)

def train(encoder, classifier, trainloader, epochs, batchSize, criterion = 'standard'):
    if criterion == 'standard':
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
            loss = criterion(outputs, labels.to(float), epoch)
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

def evaluateMatching(encoder, classifier, trainloader, testSize = 100):
    dataiter = iter(trainloader)
    X, labels = next(dataiter)
    embedding, _ = encoder(X)
    output = classifier(embedding.unsqueeze(1))

    d = int(output.size()[1]**(1/2))

    output = output.reshape(output.size()[0], d, d)
    _, predicted = torch.max(maximal_matching(output).abs(), 1)
    return deSubstitutionEmbedder(X, predicted)
