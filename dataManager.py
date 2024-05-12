# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 06:12:33 2024

@author: jacob
"""

from datasets import load_dataset
import torchtext


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
        Y.append(encryption(X))

    return X, Y

def generics_kb(length):
    dataset = load_dataset("onestop_english").with_format("torch")
    perc = (len(dataset['train']) - length)/len(dataset['train'])
    dataset = dataset['train'].train_test_split(test_size=perc)['train']['generic_sentence']

    return dataset

def onestop_english(length):
    ds = load_dataset('bookcorpus', split="train", streaming=True)
    dataset = ds.take(length).with_format("torch")
    dataset = dataset['text']

    return dataset

