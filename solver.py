# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 06:12:33 2024

@author: jacob
"""
import pandas
from datasets import load_dataset

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import numpy as np

import time
import math

#Local Modules
from dataManager import generateData, generics_kb, onestop_english
from ciphers import genericSubstitution, vigenere, substitution
from utils import *
from transformerModel import EncoderRNN, deembed
from solverModel import solverDataLoader, CustomLoss, train, evaluate
from classModel import Classifier

#torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'3
print(device)
MAX_LENGTH = 101
hidden_size = 64
SOS_token = 27
batchsize = 30
name = 'Gen64'


Y, X = generateData(device, alphaspacelower, [genericSubstitution], onestop_english, length = 10000)
trainData, testData = solverDataLoader(list(X[0]), list(Y), alphabetEmbedder, SOS_token, 28, 29, MAX_LENGTH, device)
saveData(trainData, 'trainSolveAmazon')
saveData(testData, 'testSolveAmazon')

trainData = loadData('trainSolveAmazon')
testData = loadData('testSolveAmazon')
trainloader = DataLoader(trainData, batch_size=1)
print('loaded Data')

encoder = EncoderRNN(batchsize, hidden_size, dropout_p = 0).to(device)
classifier = Classifier(4576, 27*27).to(device)

losses = train(encoder, classifier, trainloader, 20, batchsize)
saveModel(encoder, 'solveEncoder' + name)
saveModel(classifier, 'solveClass' + name)

t = np.arange(len(losses))

fig, ax = plt.subplots()
ax.plot(t, losses)

ax.set(xlabel='iteration', ylabel='loss',
        title='Loss plot')
ax.grid()

fig.savefig("lossClass1.png")
plt.show()

encoder = loadModel('solveEncoder' + name)
classifier = loadModel('solveClass' + name)

decrypt = evaluate(encoder, classifier, trainloader)

ciphertext, _ = next(iter(trainloader))
ciphertext = list(deembed(ciphertext, dealphabetEmbedder))

for c, d in zip(ciphertext[:10], decrypt[:10]):
    print('Ciphertext: ', c.replace('{', '').replace('}', '').replace('|', ''))
    print('Decryption: ', d)
    print()

print()