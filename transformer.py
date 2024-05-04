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
from dataManager import generateData
from ciphers import genericSubstitution, vigenere, substitution
from utils import alphaspacelower, alphabetEmbedder, dealphabetEmbedder
from transformerModel import EncoderRNN, AttnDecoderRNN, train, transformDataLoader, BahdanauAttention, deembed, evaluateAndShowAttention, printEval


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)

X, Y = generateData(device, alphaspacelower, [vigenere])

#print(X[0])
#print(Y[0])

datalode = transformDataLoader(X, Y[0], alphabetEmbedder, 27, 28, 29, 32, device)
hidden_size = 64
SOS_token = 27
MAX_LENGTH = 101

encoder = EncoderRNN(30, hidden_size, dropout_p = 0).to(device)
decoder = AttnDecoderRNN(hidden_size, 30, dropout_p = 0, device = device, SOS_token = SOS_token, MAX_LENGTH = MAX_LENGTH).to(device)

train(datalode, encoder, decoder, 4, print_every=1, plot_every=2)

torch.save(encoder, 'encoder1.pt')
torch.save(decoder, 'decoder1.pt')

encoder = torch.load('encoder1.pt')
decoder = torch.load('decoder1.pt')
decoder.device = device

printEval(encoder, decoder, datalode, dealphabetEmbedder)


p, f = next(iter(datalode))
print('Attention for:', list(deembed(p[:1], dealphabetEmbedder)))
evaluateAndShowAttention(encoder, decoder, f[:1], dealphabetEmbedder)

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

# for p,c,d in zip(plain[:10], cipher[:10], decrypt[:10]):
#     print('Plaintext: ', p.replace('{', '').replace('}', '').replace('|', ''))
#     print('Ciphertext:', c.replace('{', '').replace('}', '').replace('|', ''))
#     print('Decryption:', d.replace('{', '').replace('}', '').replace('|', ''))
#     print()

print('end')
