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
from dataManager import generateData, generics_kb
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


Y, X = generateData(device, alphaspacelower, [genericSubstitution], generics_kb)
X = list(X[0])
Y = list(Y)
trainData, testData = transformDataLoader(X[:2], Y[:2], alphabetEmbedder, SOS_token, 28, 29, MAX_LENGTH, device)
saveData(trainData, 'traingenTrans1')
saveData(testData, 'testgenTrans1')

trainData = loadData('trainDataTrans1')
testData = loadData('testDataTrans1')
datalode = DataLoader(trainData, batch_size=1)
print('loaded Data')

encoder = EncoderRNN(30, hidden_size, dropout_p = 0).to(device)
decoder = AttnDecoderRNN(hidden_size, 30, device = device, SOS_token = SOS_token, MAX_LENGTH = MAX_LENGTH).to(device)

# encoder = torch.load('encoder.pt')
# decoder = torch.load('decoder.pt')
# encoder.train()
# decoder.train()
# print('loaded Model')

train(datalode, encoder, decoder, 20, print_every=1, plot_every=2)

saveModel(encoder, decoder, '5')

encoder, decoder = loadModel('5')
decoder.device = device

printEval(encoder, decoder, datalode, dealphabetEmbedder)


c, p = next(iter(datalode))
print('Attention for:', list(deembed(p[:1], dealphabetEmbedder)))
evaluateAndShowAttention(encoder, decoder, c[:1], dealphabetEmbedder)

# for d in datalode:
#     break
# p,f = d[0], d[1]

# encoder.eval()
# decoder.eval()
# encoder_outputs, encoder_hidden = encoder(p)
# decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

# _, topi = decoder_outputs.topk(1)
# decoded_ids = topi.squeeze()

# decrypt = list(deembed(decoded_ids, dealphabetEmbedder))
# cipher = list(deembed(f, dealphabetEmbedder))
# plain = list(deembed(p, dealphabetEmbedder))

# for p,c,d in zip(plain[:10], cipher[:10], decrypt[:10]):1%1
#     print('Plaintext: ', p.replace('{', '').replace('}', '').replace('|', ''))
#     print('Ciphertext:', c.replace('{', '').replace('}', '').replace('|', ''))
#     print('Decryption:', d.replace('{', '').replace('}', '').replace('|', ''))
#     print()

print('end')
