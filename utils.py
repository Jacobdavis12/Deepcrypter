# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 06:12:33 2024

@author: jacob
"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl

import torch

import numpy as np

import os
dirname = os.path.realpath('..')


def characterMap(dataset, function):
    return [''.join([function(value) for value in data]) for data in dataset]

#Cleaners
def alphaspacelower(value):
    value = value.lower()
    if ord(value) >= ord('a') and ord(value) <= ord('z') or value == ' ':
        return value
    else:
        return ''

def alphalower(value):
    value = value.lower()
    if ord(value) >= ord('a') and ord(value) <= ord('z'):
        return value
    else:
        return ''

#Embedders
def alphabetEmbedder(value):
    if value == ' ':
        return 0
    else:
        return ord(value) - ord('a') + 1

def dealphabetEmbedder(value):
    if value == 0:
        return ' '
    else:
        return chr(value + ord('a') - 1)

def oneHotEncoder(data, alph):
    length = max([len(i) for i in data])
    alphDict = {a:i for i, a in enumerate(alph)}
    matrix = np.empty((len(data), length, len(alph)))

    for i, d in enumerate(data):
        for j, char in enumerate(d):
            matrix[i][j][alphDict[char]] = 1

    return matrix

def deOneHotEncoder(data, alph):
    out = []
    for d in data:
        sent = ''
        for char in d[0]:
            i = (char ==1).nonzero(as_tuple=False)
            if len(i) == 0:
                break
            else:
                sent += alph[i[0][0]]
        out.append(sent)

    return out

def substitutionEmbedder(X, y, alph = ' abcdefghijklmnopqrstuvwxyz'):
    alphDict = {a:i for i, a in enumerate(alph)}
    matrix = np.zeros((len(y), len(alph), len(alph)))

    for i, d in enumerate(y):
        for j, char in enumerate(d):
            matrix[i][alphDict[char]][alphDict[X[i][j]]] = 1

    return matrix

def deSubstitutionEmbedder(X, y, alph = ' abcdefghijklmnopqrstuvwxyz'):
    out = []

    for i, d in enumerate(y):
        sent = ''
        for j, char in enumerate(X[i]):
            if char not in [27,29,28]:
                sent += alph[d[char]]
        out.append(sent)

    return out


#Plots
def plotConfusion(confusionMatrix, classes, save =False):
    fig, ax = plt.subplots()
    im = ax.imshow(confusionMatrix)
    ax.set_xticks(np.arange(len(classes)), labels=classes)
    ax.set_yticks(np.arange(len(classes)), labels=classes)
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, confusionMatrix[i, j],
                           ha="center", va="center", color="w")
    
    ax.set_title("Confusion matrix")
    fig.tight_layout()

    if save != False:
        plt.savefig(save + '.png')

    plt.show()

#Saves
def saveData(data, name):
    filename = 'data/' + name
    torch.save(data, filename + '.pt')

def loadData(name, device):
    filename = 'data/' + name + '.pt'
    return torch.load(filename, map_location=device)

def saveClass(model, name):
    filename = 'models/' + name
    torch.save(model.state_dict(), filename + '.pth')

def loadClass(name, device):
    filename = 'models/' + name
    return torch.load(filename + '.pth', map_location=device)

def saveModel(model, name):
    filename = 'models/' + name
    torch.save(model, filename + '.pt')

def loadModel(name, device):
    filename = 'models/' + name
    return torch.load(filename + '.pt', map_location=device)

def saveTrans(encoder, decoder, name):
    filename = 'models/' + name
    torch.save(decoder, filename + 'decoder.pt')
    torch.save(encoder, filename + 'encoder.pt')

def loadLosses(name):
    filename = 'losses/' + name
    with open(filename + '.npy', 'rb') as f:
        a = np.load(f)

    return a

def saveLosses(losses,name):
    filename = 'losses/' + name
    with open(filename + '.npy', 'wb') as f:
        np.save(f, np.array(losses))

