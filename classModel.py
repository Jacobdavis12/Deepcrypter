# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 06:12:33 2024

@author: jacob
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np


def classifierDataLoader(X, label, embedder, split, device):
    X = embedder(X, 'qwertyuiopasdfghjklzxcvbnm ', 102)
    data = TensorDataset(torch.unsqueeze(torch.LongTensor(X).to(device).float(), 1),
                         torch.LongTensor(label).to(device))
    train_size = int(split * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    
    print(len(train_dataset), len(test_dataset))
    return train_dataset, test_dataset

class Classifier(nn.Module):
    def __init__(self, midSize, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
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
