import os, random

import librosa
import librosa.core
import librosa.feature
import librosa.display
import torch
import wavio as wv
from matplotlib import pyplot as plt

####PAIRWISEMIXING WAS DONE ONCE FOR THE DEVELOPMENT TRAINING DATASET AND ONCE FOR THE EVALUATION TRAINING DATASET SEPARATELY BY CHANGING DIRECTORIES AND ADJUSTING SAVING DIRECTORIES#####
###IN ORDER TO RE-RUN THE FILE FIRST CHANGE DIRECTORIES TO YOUR OWN ONES THEN RUN ONCE FOR THE DEV DATA THEN CHANGE DIRECTORIES AND RUN FOR EVAL DATA#########

global zero
zero = False
global i
i=0

#directory1 = 'C:\\Users\\Mahmoud\\Desktop\\Original data\\dev_data'
def __get_random_sample__(directory):
    #torch.randint(len(os.listdir(directory)), (1,)).item()
    newfilename=random.choice(os.listdir(directory))
    #print(newfilename)
    otherfilename = os.path.join(directory, newfilename)
    other_sample, sr = librosa.load(otherfilename, sr=16000)
    #print(otherfilename)
    return other_sample

def _pairwise_mixing__(original_sample,other_sample) -> dict:
    #mix factor in range between 0 inclusive and 1 exclusixe
    #it almost never outputs 0
    #print("AugMIX")
    sample = original_sample.copy()
    total_size=len(os.listdir(directory))
    other_sample = __get_random_sample__(directory)
    print(other_sample)
    mix_factor = torch.rand(1).item()
    if mix_factor == 0:
        zero= True
    else:
        zero= False
    print(mix_factor)
    sample = sample * mix_factor + other_sample * (1-mix_factor)
    #sample['mix_factor'] = mix_factor
    #sample['other_section'] = other_sample['section']
    return sample

import os
directory ="C:\\Users\\Mahmoud\\Desktop\\original data\\eval_data\\fan\\train"
#directory ="C:\\Users\\Mahmoud\\Desktop\\original data\\dev_data\\fan\\train" ### CHANGE WITH THIS FOR DEV###
for filename in os.listdir(directory):
    #print(filename)
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename, sr=16000)
    #print(ParseFilename)
    #print('---------------------------------------------------------------------------------------------------------------------------------')
    #print(sr)
    print(AUGSample)
    AUGSample = _pairwise_mixing__(AUGSample,directory)
    #print('@@@@@@@@@@@@@@@@@@@@@@')
    #print(AUGSample)
    if zero == True:
        print(zero)
        #C:\\Users\\Mahmoud\\Desktop\\PAIRWISEMIXINGDEV\\fan\\zero  ### CHANGE WITH THIS FOR DEV###
        wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\PAIRWISE_MIXINGEVAL\\fan\\zero' + filename + ".wav")
                , AUGSample, sr, sampwidth=1)
    else:
        #C:\\Users\\Mahmoud\\Desktop\\PAIRWISEMIXINGDEV\\fan\\  ### CHANGE WITH THIS FOR DEV###
        wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\PAIRWISE_MIXINGEVAL\\fan\\' + filename + ".wav")
                     , AUGSample, sr, sampwidth=1)
        #print(sr)
print('fan completed')

import os
directory ="C:\\Users\\Mahmoud\\Desktop\\original data\\eval_data\\gearbox\\train"
#directory ="C:\\Users\\Mahmoud\\Desktop\\original data\\dev_data\\gearbox\\train" ### CHANGE WITH THIS FOR DEV####
for filename in os.listdir(directory):
    #print(filename)
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename, sr=16000)
    #print(ParseFilename)
    #print('---------------------------------------------------------------------------------------------------------------------------------')
    #print(sr)
    print(AUGSample)
    AUGSample = _pairwise_mixing__(AUGSample,directory)
    #print('@@@@@@@@@@@@@@@@@@@@@@')
    #print(AUGSample)
    if zero == True:
        print(zero)
        # C:\\Users\\Mahmoud\\Desktop\\PAIRWISEMIXINGDEV\\gearbox\\zero  ### CHANGE WITH THIS FOR DEV###
        wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\PAIRWISE_MIXINGEVAL\\gearbox\\zero' + filename + ".wav")
                , AUGSample, sr, sampwidth=1)
    else:
        #C:\\Users\\Mahmoud\\Desktop\\PAIRWISEMIXINGDEV\\gearbox\\  ### CHANGE WITH THIS FOR DEV#######
        wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\PAIRWISE_MIXINGEVAL\\gearbox\\' + filename + ".wav")
                     , AUGSample, sr, sampwidth=1)
        #print(sr)
print('gearbox completed')

import os
directory ="C:\\Users\\Mahmoud\\Desktop\\original data\\eval_data\\pump\\train"
#directory ="C:\\Users\\Mahmoud\\Desktop\\original data\\dev_data\\pump\\train" ### CHANGE WITH THIS FOR DEV"""
for filename in os.listdir(directory):
    #print(filename)
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename, sr=16000)
    #print(ParseFilename)
    #print('---------------------------------------------------------------------------------------------------------------------------------')
    #print(sr)
    print(AUGSample)
    AUGSample = _pairwise_mixing__(AUGSample,directory)
    #print('@@@@@@@@@@@@@@@@@@@@@@')
    #print(AUGSample)
    if zero == True:
        print(zero)
        #C:\\Users\\Mahmoud\\Desktop\\PAIRWISEMIXINGDEV\\pump\\zero
        wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\PAIRWISE_MIXINGEVAL\\pump\\zero' + filename + ".wav")
                , AUGSample, sr, sampwidth=1)
    else:
        # C:\\Users\\Mahmoud\\Desktop\\PAIRWISEMIXINGDEV\\pump\\
        wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\PAIRWISE_MIXINGEVAL\\pump\\' + filename + ".wav")
                     , AUGSample, sr, sampwidth=1)
        #print(sr)
print('pump completed')

import os
directory ="C:\\Users\\Mahmoud\\Desktop\\original data\\eval_data\\slider\\train"
#directory ="C:\\Users\\Mahmoud\\Desktop\\original data\\dev_data\\slider\\train" ### CHANGE WITH THIS FOR DEV"""
for filename in os.listdir(directory):
    #print(filename)
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename, sr=16000)
    #print(ParseFilename)
    #print('---------------------------------------------------------------------------------------------------------------------------------')
    #print(sr)
    print(AUGSample)
    AUGSample = _pairwise_mixing__(AUGSample,directory)
    #print('@@@@@@@@@@@@@@@@@@@@@@')
    #print(AUGSample)
    if zero == True:
        print(zero)
        #C:\\Users\\Mahmoud\\Desktop\\PAIRWISEMIXINGDEV\\slider\\zero  ### CHANGE WITH THIS FOR DEV"""
        wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\PAIRWISE_MIXINGEVAL\\slider\\zero' + filename + ".wav")
                , AUGSample, sr, sampwidth=1)
    else:
        #C:\\Users\\Mahmoud\\Desktop\\PAIRWISEMIXINGDEV\\slider\\  ### CHANGE WITH THIS FOR DEV"""
        wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\PAIRWISE_MIXINGEVAL\\slider\\' + filename + ".wav")
                     , AUGSample, sr, sampwidth=1)
        #print(sr)
print('slider completed')

import os
directory ="C:\\Users\\Mahmoud\\Desktop\\original data\\eval_data\\ToyCar\\train"
#directory ="C:\\Users\\Mahmoud\\Desktop\\original data\\dev_data\\ToyCar\\train" ### CHANGE WITH THIS FOR DEV"""
for filename in os.listdir(directory):
    #print(filename)
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename, sr=16000)
    #print(ParseFilename)
    #print('---------------------------------------------------------------------------------------------------------------------------------')
    #print(sr)
    print(AUGSample)
    AUGSample = _pairwise_mixing__(AUGSample,directory)
    #print('@@@@@@@@@@@@@@@@@@@@@@')
    #print(AUGSample)
    if zero == True:
        print(zero)
        #C:\\Users\\Mahmoud\\Desktop\\PAIRWISEMIXINGDEV\\ToyCar\\zero ### CHANGE WITH THIS FOR DEV"""
        wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\PAIRWISE_MIXINGEVAL\\ToyCar\\zero' + filename + ".wav")
                , AUGSample, sr, sampwidth=1)
    else:
        #C:\\Users\\Mahmoud\\Desktop\\PAIRWISEMIXINGDEV\\ToyCar\\ ### CHANGE WITH THIS FOR DEV"""
        wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\PAIRWISE_MIXINGEVAL\\ToyCar\\' + filename + ".wav")
                     , AUGSample, sr, sampwidth=1)
        #print(sr)
print('ToyCar completed')

import os
directory ="C:\\Users\\Mahmoud\\Desktop\\original data\\eval_data\\ToyTrain\\train"
#directory ="C:\\Users\\Mahmoud\\Desktop\\original data\\dev_data\\ToyTrain\\train" ### CHANGE WITH THIS FOR DEV"""
for filename in os.listdir(directory):
    #print(filename)
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename, sr=16000)
    #print(ParseFilename)
    #print('---------------------------------------------------------------------------------------------------------------------------------')
    #print(sr)
    print(AUGSample)
    AUGSample = _pairwise_mixing__(AUGSample,directory)
    #print('@@@@@@@@@@@@@@@@@@@@@@')
    #print(AUGSample)
    if zero == True:
        print(zero)
        #C:\\Users\\Mahmoud\\Desktop\\PAIRWISEMIXINGDEV\\ToyTrain\\zero ### CHANGE WITH THIS FOR DEV"""
        wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\PAIRWISE_MIXINGEVAL\\ToyTrain\\zero' + filename + ".wav")
                , AUGSample, sr, sampwidth=1)
    else:
        #C:\\Users\\Mahmoud\\Desktop\\PAIRWISEMIXINGDEV\\ToyTrain\\ ### CHANGE WITH THIS FOR DEV"""
        wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\PAIRWISE_MIXINGEVAL\\ToyTrain\\' + filename + ".wav")
                     , AUGSample, sr, sampwidth=1)
        #print(sr)
print('ToyTrain completed')

import os
#directory ="C:\\Users\\Mahmoud\\Desktop\\original data\\dev_data\\valve\\train" ### CHANGE WITH THIS FOR DEV"""
directory ="C:\\Users\\Mahmoud\\Desktop\\original data\\eval_data\\valve\\train"
for filename in os.listdir(directory):
#if i < len(os.listdir(directory)):
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename, sr=16000)
    AUGSample = _pairwise_mixing__(AUGSample,directory)
    #C:\\Users\\Mahmoud\\Desktop\\PAIRWISEMIXINGDEV\\valve\\ ### CHANGE WITH THIS FOR DEV"""
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\PAIRWISE_MIXINGEVAL\\valve\\' + filename + ".wav")
                     , AUGSample, sr, sampwidth=1)
print('valve completed')