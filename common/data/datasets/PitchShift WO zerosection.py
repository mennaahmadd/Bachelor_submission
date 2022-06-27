import random

import librosa
import librosa.core
import librosa.feature
import librosa.display
import torch
import wavio as wv
from matplotlib import pyplot as plt

####PITCH SHIFT WAS DONE ONCE FOR THE DEVELOPMENT TRAINING DATASET AND ONCE FOR THE EVALUATION TRAINING DATASET SEPARATELY BY CHANGING DIRECTORIES AND ADJUSTING SAVING DIRECTORIES#####
###IN ORDER TO RE-RUN THE FILE FIRST CHANGE DIRECTORIES TO YOUR OWN ONES THEN RUN ONCE FOR THE DEV DATA THEN CHANGE DIRECTORIES AND RUN FOR EVAL DATA#########


def _augment_pitch_shift1(sample,sr=16000):
    list = [-2, -1 , 1 , 2]
    n_steps = random.choice(list)
    print(n_steps)
    return librosa.effects.pitch_shift(sample, sr=sr, n_steps=n_steps)


import os
directory = 'C:\\Users\\Mahmoud\\shared\\DCASE2021\\task2\\dev_data\\fan\\train'
#directory = 'C:\\Users\\Mahmoud\\shared\\DCASE2021\\task2\\eval_data\\fan\\train'
for filename in os.listdir(directory):
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename, sr=16000)
    print(sr)
    #librosa.display.waveshow(AUGSample)
    #plt.show()
    print(ParseFilename)
    print(sr)
    AugmentedSample = _augment_pitch_shift1(AUGSample)
    #librosa.display.waveshow(AugmentedSample)
    #plt.show()
    #'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONEVAL\\fan\\'
    wv.write(os.path.join( 'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONDEV\\fan\\'+ filename + ".wav")
             , AugmentedSample, sr, sampwidth=1)

print("Fan completed")

import os
directory = 'C:\\Users\\Mahmoud\\shared\\DCASE2021\\task2\\dev_data\\gearbox\\train'
#directory = 'C:\\Users\\Mahmoud\\shared\\DCASE2021\\task2\\eval_data\\gearbox\\train'
for filename in os.listdir(directory):
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename,sr=16000)
    print(ParseFilename)
    print(sr)
    AugmentedSample = _augment_pitch_shift1(AUGSample)
     #'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONEVAL\\gearbox\\'
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONDEV\\gearbox\\' + filename + ".wav")
             , AugmentedSample, sr, sampwidth=1)
    print(sr)
print("gearbox completed")


import os
directory = 'C:\\Users\\Mahmoud\\shared\\DCASE2021\\task2\\dev_data\\pump\\train'
#directory = 'C:\\Users\\Mahmoud\\shared\\DCASE2021\\task2\\eval_data\\pump\\train'
for filename in os.listdir(directory):
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename, sr=16000)
    print(ParseFilename)
    print(sr)
    AugmentedSample = _augment_pitch_shift1(AUGSample)
    #'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONEVAL\\pump\\'
    wv.write(os.path.join( 'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONDEV\\pump\\'+ filename + ".wav")
             , AugmentedSample, sr, sampwidth=1)
    print(sr)
print('pump completed')


import os
directory = 'C:\\Users\\Mahmoud\\shared\\DCASE2021\\task2\\dev_data\\slider\\train'
#directory = 'C:\\Users\\Mahmoud\\shared\\DCASE2021\\task2\\eval_data\\slider\\train'
for filename in os.listdir(directory):
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename, sr=16000)
    print(ParseFilename)
    print(sr)
    AugmentedSample = _augment_pitch_shift1(AUGSample)
    #'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONEVAL\\slider\\'
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONDEV\\slider\\' + filename + ".wav")
             , AugmentedSample, sr, sampwidth=1)
    print(sr)
print('slider completed')


import os
directory = 'C:\\Users\\Mahmoud\\shared\\DCASE2021\\task2\\dev_data\\ToyCar\\train'
#directory = 'C:\\Users\\Mahmoud\\shared\\DCASE2021\\task2\\eval_data\\ToyCar\\train'
for filename in os.listdir(directory):
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename, sr=16000)
    print(ParseFilename)
    print(sr)
    AugmentedSample = _augment_pitch_shift1(AUGSample)
    #'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONEVAL\\ToyCar\\'
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONDEV\\ToyCar\\' + filename + ".wav")
             , AugmentedSample, sr, sampwidth=1)
    print(sr)
print('ToyCar completed')


import os
directory = 'C:\\Users\\Mahmoud\\shared\\DCASE2021\\task2\\dev_data\\ToyTrain\\train'
#directory = 'C:\\Users\\Mahmoud\\shared\\DCASE2021\\task2\\eval_data\\ToyTrain\\train'
for filename in os.listdir(directory):
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename, sr=16000)
    print(ParseFilename)
    print(sr)
    AugmentedSample = _augment_pitch_shift1(AUGSample)
    #'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONEVAL\\ToyTrain\\'
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONDEV\\ToyTrain\\' + filename + ".wav")
             , AugmentedSample, sr, sampwidth=1)
    print(sr)
print('ToyTrain completed')


import os
directory = 'C:\\Users\\Mahmoud\\shared\\DCASE2021\\task2\\dev_data\\valve\\train'
#directory = 'C:\\Users\\Mahmoud\\shared\\DCASE2021\\task2\\eval_data\\valve\\train'
for filename in os.listdir(directory):
    ParseFilename = os.path.join(directory, filename)
    AUGSample, sr = librosa.load(ParseFilename, sr=16000)
    print(ParseFilename)
    print(sr)
    AugmentedSample = _augment_pitch_shift1(AUGSample)
    #'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONEVAL\\valve\\'
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONDEV\\valve\\'+ filename + ".wav")
             , AugmentedSample, sr, sampwidth=1)
    print(sr)
print('valve completed')
