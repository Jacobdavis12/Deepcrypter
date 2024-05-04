# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 06:12:33 2024

@author: jacob
"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl


def characterMap(dataset, function):
    return [''.join([function(value) for value in data]) for data in dataset]

#Cleaners
def alphaspacelower(value):
    value = value.lower()
    if ord(value) >= ord('a') and ord(value) <= ord('z') or value == ' ':
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

def oneHotEncoder(data, alph, length):
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

#Plots
def plotConfusion(confusionMatrix, classes):
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
    plt.show()