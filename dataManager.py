# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 06:12:33 2024

@author: jacob
"""

from datasets import load_dataset

import random

#Local
from utils import characterMap

#Cleaner is applied to plain then encryption is applied to cipher
#Returned as pytorch
def generateData(device, cleaner, encryptions, dataset, noise = [0, 0], length = 1):
    dataset = dataset(length)
    print(len(dataset))
    X = characterMap(dataset, cleaner)

    Y = []
    for encryption in encryptions:
        Y.append(addNoise(encryption(X), noise))

    return X, Y

def addNoise(data, noise):
    if noise == [0,0]:
        return list(data)

    out = []
    for arr in data:
        arr1 = [char for char in arr if random.randint(0, 100)/100 > noise[0]]
        arr1 = [char if random.randint(0, 100)/100 > noise[0] else randomChar() for char in arr1]
        out.append(''.join(arr1))

    return out

def randomChar():
    return 'qwertyuiopasdfghjklzxcvbnm '[random.randint(0, 26)]

def generics_kb(length):
    ds = load_dataset('generics_kb', split="train", streaming=True)
    dataset = ds.take(length).with_format("torch")
    dataset = [d['generic_sentence'] for d in dataset]

    return dataset

def bookcorpus(length):
    ds = load_dataset('bookcorpus', split="train", streaming=True)
    dataset = ds.take(length).with_format("torch")
    dataset = [d['text'] for d in dataset]

    return dataset
