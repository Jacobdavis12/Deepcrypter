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

from utils import dealphabetEmbedder

def transformDataLoader(X, y, embedder, sos, eos, pad, MAX_LENGTH, device, split = 0.8):
    X = list(embed(list(X), embedder, sos, eos, pad, MAX_LENGTH))
    y = list(embed(list(y), embedder, sos, eos, pad, MAX_LENGTH))
    data = TensorDataset(torch.LongTensor(X).to(device),
                         torch.LongTensor(y).to(device))
    train_size = int(split * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    
    print(len(train_dataset), len(test_dataset))
    return train_dataset, test_dataset

def embed(dataset, embedder, sos, eos, pad, length):
    for data in dataset:
        yield [sos] + [embedder(value) for value in data] + (length - len(data) - 2)*[pad] + [eos]

def deembed(dataset, deembedder):
    for data in dataset:
        yield ''.join([deembedder(value) for value in data])

#Loss functions

class CustomLoss(nn.Module):
    def __init__(self): 
        super(CustomLoss, self).__init__() 
 
    def forward(self, predicted, target):
        #Natural language
        lossFunc = nn.NLLLoss()

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

                    cipherLoss[i][j] = equalsLoss/2 + notEqualsLoss/2

        print(list(deembed(predicted[-1].topk(1)[1].squeeze().unsqueeze(0), dealphabetEmbedder)))

        return cipherLoss[cipherLoss != 0].mean()/2 + \
         lossFunc(predicted.view(-1, predicted.size(-1)), target.view(-1))/2

#Models

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.lstm(embedded)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device, MAX_LENGTH, SOS_token):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.device = device
        self.MAX_LENGTH = MAX_LENGTH
        self.SOS_token = SOS_token

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if False:#target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output)
        return output, hidden

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs,
            target_tensor
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, MAX_LENGTH, SOS_token, dropout_p=0.1, device = 'cpu'):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.device = device
        self.MAX_LENGTH = MAX_LENGTH
        self.SOS_token = SOS_token

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.SOS_token)
        decoder_hidden = encoder_hidden[0]
        decoder_outputs = []
        attentions = []

        for i in range(self.MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if False:#target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

def norm(val, std=1/2):
    return torch.exp(-(val/std)**2)

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100, criterion = 'standard'):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    if criterion == 'standard':
        criterion = CustomLoss()


    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def printEval(encoder, decoder, dataloader, dembedder, length = 10):
    criterion = CustomLoss()
    c, p = next(iter(dataloader))

    decoded_ids, decoder_attn = evaluate(encoder, decoder, c)

    decrypt = list(deembed(decoded_ids, dembedder))
    cipher = list(deembed(c, dembedder))
    plain = list(deembed(p, dembedder))

    #print(criterion(f, decoded_ids))

    for p,c,d in zip(plain[:length], cipher[:length], decrypt[:length]):
        print('Plaintext: ', p.replace('{', '').replace('}', '').replace('|', ''))
        print('Ciphertext:', c.replace('{', '').replace('}', '').replace('|', ''))
        print('Decryption:', d.replace('{', '').replace('}', '').replace('|', ''))
        print()

def evaluate(encoder, decoder, f):
    encoder.eval()
    decoder.eval()
    encoder_outputs, encoder_hidden = encoder(f)
    decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

    _, topi = decoder_outputs.topk(1)
    decoded_ids = topi.squeeze()
    if len(decoded_ids.size()) == 1:
        decoded_ids = decoded_ids.unsqueeze(0)

    return decoded_ids, decoder_attn

def showAttention(input_sentence, output_words, attentions):
    font = {'size'   : 10}

    matplotlib.rc('font', **font)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().detach().numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + list(input_sentence) +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + list(output_words))

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(encoder, decoder, input_sentence, dembedder):
    decoded_ids, decoder_attn = evaluate(encoder, decoder, input_sentence)
    decoded_ids = decoded_ids.unsqueeze(0)

    decrypt = list(deembed(decoded_ids, dembedder))[0].replace('{', '').replace('}', '').replace('|', '')
    cipher = list(deembed(input_sentence, dembedder))[0].replace('{', '').replace('}', '').replace('|', '')

    showAttention(cipher, decrypt, decoder_attn[0, :len(decrypt), :])
