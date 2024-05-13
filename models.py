# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:55:28 2024

@author: jacob
"""

def seqToClassLoad():
    MAX_LENGTH = 346
    hidden_size = 64
    SOS_token = 27
    batchsize = 30
    name = 'book64'
    midSize = 17264


    Y, X = generateData(device, alphaspacelower, [genericSubstitution], bookcorpus, length = 10000)
    trainData, testData = solverDataLoader(list(X[0]), list(Y), alphabetEmbedder, SOS_token, 28, 29, MAX_LENGTH, device)
    saveData(trainData, 'trainSolve' + name)
    saveData(testData, 'testSolve' + name)