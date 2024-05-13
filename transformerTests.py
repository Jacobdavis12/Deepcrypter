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
from dataManager import generateData
from ciphers import genericSubstitution, vigenere, substitution
from utils import *
from transformerModel import EncoderRNN, AttnDecoderRNN, DecoderRNN, train, transformDataLoader, BahdanauAttention, deembed, evaluateAndShowAttention, printEval

#torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)
MAX_LENGTH = 101
hidden_size = 128
SOS_token = 27

class TestLoss1(nn.Module):
    def __init__(self, a):
        self.a = a
        super(TestLoss1, self).__init__()
 
    def forward(self, predicted, target):

        #Cipher
        cipherLoss = torch.zeros((target.size()))
        crit = nn.CrossEntropyLoss()
        for i in range(target.size()[0]):
            for j in range(predicted.size()[2]):
                if j in target[i]:
                    equals = predicted[i][target[i] == j]
                    top = equals.topk(1)[1]
                    equalsLoss = crit(equals, top.mode(axis = 0)[0].repeat(len(equals)))

                    notEquals = predicted[i][target[i] != j]
                    notEqualsLoss = 1/crit(notEquals, top.mode(axis = 0)[0].repeat(len(notEquals)))

                    cipherLoss[i][j] = equalsLoss*(1-self.a) + self.a*notEqualsLoss

        #print(list(deembed(predicted[-1].topk(1)[1].squeeze().unsqueeze(0), dealphabetEmbedder)))

        return cipherLoss[cipherLoss != 0].mean()#/2 + \
        #return lossFunc(predicted.view(-1, predicted.size(-1)), target.view(-1))#/2

def testLoss():
    #Load Data
    trainData = loadData('trainDataTrans1')
    testData = loadData('testDataTrans1')
    datalode = DataLoader(trainData, batch_size=30)

    for i in range(1, 10):
        encoder = EncoderRNN(30, hidden_size, dropout_p = 0).to(device)
        decoder = AttnDecoderRNN(hidden_size, 30, device = device, SOS_token = SOS_token, MAX_LENGTH = MAX_LENGTH).to(device)

        a = i/10
        train(datalode, encoder, decoder, 4, print_every=2, plot_every=1, criterion = TestLoss1(a))
        saveModel(encoder, decoder, 'testloss1-a' + str(a))

        printEval(encoder, decoder, datalode, dealphabetEmbedder, 1)

testLoss()
print('end')
