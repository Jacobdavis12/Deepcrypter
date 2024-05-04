# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 06:12:33 2024

@author: jacob
"""

from datasets import load_dataset

#Local
from utils import characterMap

#Cleaner is applied to plain then encryption is applied to cipher
#Returned as pytorch
def generateData(device, cleaner, encryptions, length = 1):
    dataset = load_dataset("generics_kb").with_format("torch")
    dataset = dataset['train'].train_test_split(test_size=1-0.02)['train']['generic_sentence']
    print(len(dataset))
    X = characterMap(dataset, cleaner)

    Y = []
    for encryption in encryptions:
        Y.append(encryption(X))

    return X, Y

