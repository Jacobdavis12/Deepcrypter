# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 06:12:33 2024

@author: jacob
"""

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
from dataManager import generateData, generics_kb, bookcorpus
from ciphers import genericSubstitution, vigenere, substitution
from utils import *
from transformerModel import EncoderRNN, deembed
from solverModel import solverDataLoader, CustomLoss, train, evaluate, MatchingLoss, evaluateMatching,JointLoss
from classModel import Classifier

#torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)
MAX_LENGTH = 346
hidden_size = 64
SOS_token = 27
batchsize = 30
named = 'noise1000Book'
name = 'noise1000Book'


Y, X = generateData(device, alphaspacelower, [genericSubstitution], bookcorpus, noise = [0, 0], length = 1000)

trainData, testData = solverDataLoader(list(X[0]), list(Y), alphabetEmbedder, SOS_token, 28, 29, MAX_LENGTH, device)
saveData(trainData, 'trainSolve' + named)
saveData(testData, 'testSolve' + named)

trainData = loadData('trainSolve' + named, device)
testData = loadData('testSolve' + named, device)
print('dataLoded')
trainloader = DataLoader(trainData, batch_size=30)
print('loaded Data')

encoder = EncoderRNN(batchsize, hidden_size, dropout_p = 0).to(device)
encoder_outputs, _ = encoder(next(iter(trainloader))[0])
classifier = Classifier(encoder_outputs.unsqueeze(1).to('cpu'), 27*27).to(device)

losses = train(encoder, classifier, trainloader, 100, batchsize, JointLoss())
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

encoder = loadModel('solveEncoder' + name, device)
classifier = loadModel('solveClass' + name, device)

decrypt = evaluateMatching(encoder, classifier, trainloader)

ciphertext, target = next(iter(trainloader))
ciphertext = list(deembed(ciphertext, dealphabetEmbedder))

for c, d, t in zip(ciphertext[:10], decrypt[:10], target[:10]):
    print('Ciphertext: ', c.replace('{', '').replace('}', '').replace('|', ''))
    print('Decryption: ', d)
    _, t1 = torch.max(t, 1)
    #print('Target: ', t1)
    print()

print()