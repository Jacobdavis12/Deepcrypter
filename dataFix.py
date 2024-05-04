# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 06:12:33 2024

@author: jacob
"""
import torch
from transformer import EncoderRNN, AttnDecoderRNN, train, transformDataLoader, BahdanauAttention
hidden_size = 64
SOS_token = 27
MAX_LENGTH = 101
device = 'cuda'

encoder = torch.load('encoder.pt')
torch.save(encoder, 'encoder.pt')

# decoder = AttnDecoderRNN(hidden_size, 30, MAX_LENGTH = MAX_LENGTH, SOS_token = SOS_token)
decoder = torch.load('decoder.pt')
decoder.device = device
decoder.MAX_LENGTH = MAX_LENGTH
decoder.SOS_token = SOS_token
torch.save(decoder, 'decoder.pt')