# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 06:12:33 2024

@author: jacob
"""

import random
import numpy as np

def genericSubstitution(dataset):
    for data in dataset:
        cipherAlphabet = {' ': ' '}

        i = 0
        for value in data:
            if value not in cipherAlphabet:
                cipherAlphabet[value] = chr(i + ord('a'))
                i += 1

        yield ''.join([cipherAlphabet[i] for i in data])

def substitution(dataset, key = 'randomised'):
    if key == 'randomised':
        alph = [chr(i + ord('a')) for i in range(26)]
    else:
        alph = key

    for data in dataset:
        if key == 'randomised':
            random.shuffle(alph)

        yield ''.join([alph[ord(value)-ord('a')] if value != ' ' else ' ' for value in data])

def vigenere(dataset, keyLength = 3):
    key = [chr(i + ord('a')) for i in range(26)]
    for data in dataset:
        data = np.asarray(list(data))
        for i in range(keyLength):
            random.shuffle(key)
            data[i::keyLength] = np.asarray([key[ord(value)-ord('a')] if value != ' ' else ' ' for value in data[i::keyLength]])

        yield ''.join(data)

def column(dataset, keyLength = 3):
    key = list(range(keyLength))
    random.shuffle(key)
    for data in dataset:
        cipherText = np.asarray(list(data) + ['_']*((3-len(data))%keyLength))
        data = np.asarray(list(data) + ['_']*((3-len(data))%keyLength))
        for i in range(keyLength):
            cipherText[key[i]::keyLength] = data[i::keyLength]

        yield ''.join(cipherText).replace('_', '')