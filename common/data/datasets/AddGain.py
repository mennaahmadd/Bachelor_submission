import random

import librosa
import torch
import wavio as wv

####GAIN WAS DONE ONCE FOR THE DEVELOPMENT TRAINING DATASET AND ONCE FOR THE EVALUATION TRAINING DATASET SEPARATELY BY CHANGING DIRECTORIES AND ADJUSTING SAVING DIRECTORIES#####
###IN ORDER TO RE-RUN THE FILE FIRST CHANGE DIRECTORIES TO YOUR OWN ONES THEN RUN ONCE FOR THE DEV DATA THEN CHANGE DIRECTORIES AND RUN FOR EVAL DATA#########


def _augment_addgain_(sample, low: float = 1.25, high: float = 6.25) -> dict:
    #Applies a random gain between low and high #
    Gain = random.uniform(low,high)
    print(Gain)
    print(sample)
    sample = sample * Gain
    print('-------------------------------------')
    print(sample)
    return sample


import os
#directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\fan\\train'
directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\eval_data\\fan\\train'
for filename in os.listdir(directory):
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename, sr=16000)
    AUGSample = _augment_addgain_(AUGSample)
    #'C:\\Users\\Mahmoud\\Desktop\\GainDEV\\fan\\'
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\GainEVAL\\fan\\' + filename + ".wav")
             , AUGSample, sr, sampwidth=1)
import os
#directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\gearbox\\train'
directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\eval_data\\gearbox\\train'
for filename in os.listdir(directory):
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename, sr=16000)
    AUGSample = _augment_addgain_(AUGSample)
    # 'C:\\Users\\Mahmoud\\Desktop\\GainDEV\\gearbox\\'
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\GainEVAL\\gearbox\\' + filename + ".wav")
             , AUGSample, sr, sampwidth=1)

import os
#directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\pump\\train'
directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\eval_data\\pump\\train'
# iterate over files in
# that directory
for filename in os.listdir(directory):
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename, sr=16000)
    AUGSample = _augment_addgain_(AUGSample)
    # 'C:\\Users\\Mahmoud\\Desktop\\GainDEV\\pump\\'
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\GainEVAL\\pump\\' + filename + ".wav")
             , AUGSample, sr, sampwidth=1)

import os
#directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\slider\\train'
directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\eval_data\\slider\\train'
for filename in os.listdir(directory):
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename, sr=16000)
    AUGSample = _augment_addgain_(AUGSample)
    # 'C:\\Users\\Mahmoud\\Desktop\\GainDEV\\slider\\'
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\GainEVAL\\slider\\' + filename + ".wav")
             , AUGSample, sr, sampwidth=1)

import os
#directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\ToyCar\\train'
directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\eval_data\\ToyCar\\train'
for filename in os.listdir(directory):
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename, sr=16000)
    AUGSample = _augment_addgain_(AUGSample)
    # 'C:\\Users\\Mahmoud\\Desktop\\GainDEV\\ToyCar\\'
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\GainEVAL\\ToyCar\\' + filename + ".wav")
             , AUGSample, sr, sampwidth=1)

import os
#directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\ToyTrain\\train'
directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\eval_data\\ToyTrain\\train'
for filename in os.listdir(directory):
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename, sr=16000)
    AUGSample = _augment_addgain_(AUGSample)
    # 'C:\\Users\\Mahmoud\\Desktop\\GainDEV\\ToyTrain\\'
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\GainEVAL\\ToyTrain\\' + filename + ".wav")
             , AUGSample, sr, sampwidth=1)

import os
#directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\valve\\train'
directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\eval_data\\valve\\train'
for filename in os.listdir(directory):
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename, sr=16000)
    AUGSample = _augment_addgain_(AUGSample)
    # 'C:\\Users\\Mahmoud\\Desktop\\GainDEV\\valve\\'
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\GainEVAL\\valve\\' + filename + ".wav")
             , AUGSample, sr, sampwidth=1)


