import librosa
import librosa.core
import librosa.feature
import librosa.display
import torch
import wavio as wv


###NOTE:###
###I DID EACH MACHINE SEPARATLY ONCE FOR DEVELOPMENT TRAINING DATASET AND ONCE FOR EVALUATION TRAINING DATASET,
# IN ORDER NOT TO GET CONFUSED AND TO ASSURE THAT EVERYTHING IS GOING WELL WITH PRINT STATEMENTS###
### FOR SUBMITTING THE CODE I CHANGED THE VARIABLES NAMES IN ORDER FOR ANYONE TO BE ABLE TO RUN THE PYTHON FILE AT ONCE####
### IF ANYONE IS GOING TO RUN THE CODE PLEASE MAKE SURE TO CHANGE THE DIRECTORIES ####
### AUGMENTED DATA WERE ADDED TO THE TRAINING OF THE MODEL BY RANDOMLY SELECTING FROM THE AUGMENTED DATA FOR TRAINING DIRECTORIES ###
### AUGMIX TAKES THE GAIN-AUGMENTED SAMPLE WITH THE POLARITY INVERSION SAMPLE AND PITCH SHIFT SAMPLE AND CONVEXLY COMBINE THEM AFTER ADDING WEIGHT TO EACH WITH THE ORIGINAL SAMPLE#####

global i1
i1= 0

global directoryGain
directoryGain1 = 'C:\\Users\\Mahmoud\\Desktop\\GainDEV\\fan'
global directoryPitch
directoryPitch1 = 'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONDEV\\fan'
global directoryPI
directoryPI1 = 'C:\\Users\\Mahmoud\\Desktop\\PolarityInversionDEV\\fan'

global GainSample1
global PitchSample1
global PISample1

def __get_Gain_sample__(directoryGain1):
    global i1
    global GainSample1
    if i1 < len(os.listdir(directoryGain1)):
        ParseFilenameGain1 = os.path.join(directoryGain1, os.listdir(directoryGain1)[i1])
        print('Gain:' + ParseFilenameGain1)
        GainSample1, sr = librosa.load(ParseFilenameGain1, sr=16000)
        weight1 = torch.rand(1).item()
        GainSample1 = GainSample1 * weight1
        print(weight1)
    return GainSample1

def __get_Pitch_sample__(directoryPitch1):
    global i1
    global PitchSample1
    if i1 < len(os.listdir(directoryPitch1)):
        ParseFilenamePitch1 = os.path.join(directoryPitch1, os.listdir(directoryPitch1)[i1])
        print('Pitch:' + ParseFilenamePitch1)
        PitchSample1, sr = librosa.load(ParseFilenamePitch1, sr=16000)
        weight2 = torch.rand(1).item()
        PitchSample1 = PitchSample1*weight2
        print(weight2)
    return PitchSample1

def __get_PI_sample__(directoryPI1):
    global i1
    global PISample1
    if i1 < len(os.listdir(directoryGain1)):
        ParseFilenamePI1 = os.path.join(directoryPI1, os.listdir(directoryPI1)[i1])
        print('PI:' + ParseFilenamePI1)
        PISample1, sr = librosa.load(ParseFilenamePI1, sr=16000)
        weight3 = torch.rand(1).item()
        PISample1 = PISample1*weight3
        print(weight3)
    return PISample1


def _aug_mix__(original_sample1, directoryPitch1, directoryGain1,directoryPI1):
    #mix factor in range between 0 inclusive and 1 exclusixe
    #it almost never outputs 0
    sample1 = original_sample1.copy()
    mix_factor = torch.rand(1).item()
    sampleaug1 = __get_Pitch_sample__(directoryPitch1) + __get_Gain_sample__(directoryGain1)+ __get_PI_sample__(directoryPI1)
    sample1 = (sample1*mix_factor) + ((1-mix_factor)*sampleaug1)
    print(mix_factor)
    print(sample1)
    global i1
    i1 = i1+1
    return sample1

import os
directory1 ='C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\fan\\train'
for filename1 in os.listdir(directory1):
    ParseFilename1 = os.path.join(directory1, filename1)
    print('ParseFilename:' + ParseFilename1)
    AUGSample1, sr = librosa.load(ParseFilename1, sr=16000)
    AUGSampleF1 = _aug_mix__(AUGSample1, directoryPitch1, directoryGain1, directoryPI1)
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\AUGMIXDEV\\fan\\' + filename1 + ".wav")
                     , AUGSampleF1, sr, sampwidth=1)
print('FAN COMPLETED')
#-----------------------------------------------------------------------------------------------------------------------------------------------------
global i2
i2=0
global directoryGain2
directoryGain2 = 'C:\\Users\\Mahmoud\\Desktop\\GainDEV\\gearbox'
global directoryPitch2
directoryPitch2 = 'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONDEV\\gearbox'
global directoryPI2
directoryPI2 = 'C:\\Users\\Mahmoud\\Desktop\\PolarityInversionDEV\\gearbox'

global GainSample2
global PitchSample2
global PISample2

def __get_Gain_sample__(directoryGain2):
    global i2
    global GainSample2
    if i2 < len(os.listdir(directoryGain2)):
        ParseFilenameGain2 = os.path.join(directoryGain2, os.listdir(directoryGain2)[i2])
        print('Gain:' + ParseFilenameGain2)
        GainSample2, sr = librosa.load(ParseFilenameGain2, sr=16000)
        weight12 = torch.rand(1).item()
        GainSample2 = GainSample2 * weight12
        print(weight12)
    return GainSample2

def __get_Pitch_sample__(directoryPitch2):
    global i2
    global PitchSample2
    if i2 < len(os.listdir(directoryPitch2)):
        ParseFilenamePitch2 = os.path.join(directoryPitch2, os.listdir(directoryPitch2)[i2])
        print('Pitch:' + ParseFilenamePitch2)
        PitchSample2, sr = librosa.load(ParseFilenamePitch2, sr=16000)
        weight22 = torch.rand(1).item()
        PitchSample2 = PitchSample2*weight22
        print(weight22)
    return PitchSample2

def __get_PI_sample__(directoryPI2):
    global i2
    global PISample2
    if i2 < len(os.listdir(directoryGain2)):
        ParseFilenamePI2 = os.path.join(directoryPI2, os.listdir(directoryPI2)[i2])
        print('PI:' + ParseFilenamePI2)
        PISample2, sr = librosa.load(ParseFilenamePI2, sr=16000)
        weight32 = torch.rand(1).item()
        PISample2 = PISample2*weight32
        print(weight32)
    return PISample2


def _aug_mix__(original_sample2, directoryPitch2, directoryGain2,directoryPI2):
    #mix factor in range between 0 inclusive and 1 exclusixe
    #it almost never outputs 0
    sample2 = original_sample2.copy()
    mix_factor = torch.rand(1).item()
    sampleaug2 = __get_Pitch_sample__(directoryPitch2) + __get_Gain_sample__(directoryGain2)+ __get_PI_sample__(directoryPI2)
    sample2 = (sample2*mix_factor) + ((1-mix_factor)*sampleaug2)
    print(mix_factor)
    print(sample2)
    global i2
    i2 = i2+1
    return sample2

import os
directory2 ='C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\gearbox\\train'
for filename2 in os.listdir(directory2):
    ParseFilename2 = os.path.join(directory2, filename2)
    print('ParseFilename:' + ParseFilename2)
    AUGSample2, sr = librosa.load(ParseFilename2, sr=16000)
    AUGSampleF2 = _aug_mix__(AUGSample2, directoryPitch2, directoryGain2, directoryPI2)
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\AUGMIXDEV\\gearbox\\' + filename2 + ".wav")
                     , AUGSampleF2, sr, sampwidth=1)
print('GEARBOX COMPLETED')
#-------------------------------------------------------------------------------------------------------------------------------------------------
global i3
i3=0

global directoryGain3
directoryGain3 = 'C:\\Users\\Mahmoud\\Desktop\\GainDEV\\pump'
global directoryPitch3
directoryPitch3 = 'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONDEV\\pump'
global directoryPI3
directoryPI3 = 'C:\\Users\\Mahmoud\\Desktop\\PolarityInversionDEV\\pump'

global GainSample3
global PitchSample3
global PISample3

def __get_Gain_sample__(directoryGain3):
    global i3
    global GainSample3
    if i3 < len(os.listdir(directoryGain3)):
        ParseFilenameGain3 = os.path.join(directoryGain3, os.listdir(directoryGain3)[i3])
        print('Gain:' + ParseFilenameGain3)
        GainSample3, sr = librosa.load(ParseFilenameGain3, sr=16000)
        weight13 = torch.rand(1).item()
        GainSample3 = GainSample3 * weight13
        print(weight13)
    return GainSample3

def __get_Pitch_sample__(directoryPitch3):
    global i3
    global PitchSample3
    if i3 < len(os.listdir(directoryPitch3)):
        ParseFilenamePitch3 = os.path.join(directoryPitch3, os.listdir(directoryPitch3)[i3])
        print('Pitch:' + ParseFilenamePitch3)
        PitchSample3, sr = librosa.load(ParseFilenamePitch3, sr=16000)
        weight23 = torch.rand(1).item()
        PitchSample3 = PitchSample3*weight23
        print(weight23)
    return PitchSample3

def __get_PI_sample__(directoryPI3):
    global i3
    global PISample3
    if i3 < len(os.listdir(directoryGain3)):
        ParseFilenamePI3 = os.path.join(directoryPI3, os.listdir(directoryPI3)[i3])
        print('PI:' + ParseFilenamePI3)
        PISample3, sr = librosa.load(ParseFilenamePI3, sr=16000)
        weight33 = torch.rand(1).item()
        PISample3 = PISample3*weight33
        print(weight33)
    return PISample3


def _aug_mix__(original_sample3, directoryPitch3, directoryGain3,directoryPI3):
    #mix factor in range between 0 inclusive and 1 exclusixe
    #it almost never outputs 0
    sample3 = original_sample3.copy()
    mix_factor = torch.rand(1).item()
    sampleaug3 = __get_Pitch_sample__(directoryPitch3) + __get_Gain_sample__(directoryGain3)+ __get_PI_sample__(directoryPI3)
    sample3 = (sample3*mix_factor) + ((1-mix_factor)*sampleaug3)
    print(mix_factor)
    print(sample3)
    global i3
    i3 = i3+1
    return sample3

import os
directory3 ='C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\pump\\train'
for filename3 in os.listdir(directory3):
    ParseFilename3 = os.path.join(directory3, filename3)
    print('ParseFilename:' + ParseFilename3)
    AUGSample3, sr = librosa.load(ParseFilename3, sr=16000)
    AUGSampleF3 = _aug_mix__(AUGSample3, directoryPitch3, directoryGain3, directoryPI3)
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\AUGMIXDEV\\pump\\' + filename3 + ".wav")
                     , AUGSampleF3, sr, sampwidth=1)
print('PUMP COMPLETED')
#-------------------------------------------------------------------------------------------------------------------------------------------------
global i4
i4=0

global directoryGain4
directoryGain4 = 'C:\\Users\\Mahmoud\\Desktop\\GainDEV\\slider'
global directoryPitch4
directoryPitch4 = 'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONDEV\\slider'
global directoryPI4
directoryPI4 = 'C:\\Users\\Mahmoud\\Desktop\\PolarityInversionDEV\\slider'

global GainSample4
global PitchSample4
global PISample4

def __get_Gain_sample__(directoryGain4):
    global i4
    global GainSample4
    if i4 < len(os.listdir(directoryGain4)):
        ParseFilenameGain4 = os.path.join(directoryGain4, os.listdir(directoryGain4)[i4])
        print('Gain:' + ParseFilenameGain4)
        GainSample4, sr = librosa.load(ParseFilenameGain4, sr=16000)
        weight14 = torch.rand(1).item()
        GainSample4 = GainSample4 * weight14
        print(weight14)
    return GainSample4

def __get_Pitch_sample__(directoryPitch4):
    global i4
    global PitchSample4
    if i4 < len(os.listdir(directoryPitch4)):
        ParseFilenamePitch4 = os.path.join(directoryPitch4, os.listdir(directoryPitch4)[i4])
        print('Pitch:' + ParseFilenamePitch4)
        PitchSample4, sr = librosa.load(ParseFilenamePitch4, sr=16000)
        weight24 = torch.rand(1).item()
        PitchSample4 = PitchSample4*weight24
        print(weight24)
    return PitchSample4

def __get_PI_sample__(directoryPI4):
    global i4
    global PISample4
    if i4 < len(os.listdir(directoryGain4)):
        ParseFilenamePI4 = os.path.join(directoryPI4, os.listdir(directoryPI4)[i4])
        print('PI:' + ParseFilenamePI4)
        PISample4, sr = librosa.load(ParseFilenamePI4, sr=16000)
        weight34 = torch.rand(1).item()
        PISample4 = PISample4*weight34
        print(weight34)
    return PISample4


def _aug_mix__(original_sample4, directoryPitch4, directoryGain4,directoryPI4):
    #mix factor in range between 0 inclusive and 1 exclusixe
    #it almost never outputs 0
    sample4 = original_sample4.copy()
    mix_factor = torch.rand(1).item()
    sampleaug4 = __get_Pitch_sample__(directoryPitch4) + __get_Gain_sample__(directoryGain4)+ __get_PI_sample__(directoryPI4)
    sample4 = (sample4*mix_factor) + ((1-mix_factor)*sampleaug4)
    print(mix_factor)
    print(sample4)
    global i4
    i4 = i4+1
    return sample4

import os
directory4 ='C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\slider\\train'
for filename4 in os.listdir(directory4):
    ParseFilename4 = os.path.join(directory4, filename4)
    print('ParseFilename:' + ParseFilename4)
    AUGSample4, sr = librosa.load(ParseFilename4, sr=16000)
    AUGSampleF4 = _aug_mix__(AUGSample4, directoryPitch4, directoryGain4, directoryPI4)
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\AUGMIXDEV\\slider\\' + filename4 + ".wav")
                     , AUGSampleF4, sr, sampwidth=1)
print('SLIDER COMPLETED')
#------------------------------------------------------------------------------------------------------------------------------
global i5
i5=0
global directoryGain5
directoryGain5 = 'C:\\Users\\Mahmoud\\Desktop\\GainDEV\\ToyCar'
global directoryPitch5
directoryPitch5 = 'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONDEV\\ToyCar'
global directoryPI5
directoryPI5 = 'C:\\Users\\Mahmoud\\Desktop\\PolarityInversionDEV\\ToyCar'

global GainSample5
global PitchSample5
global PISample5

def __get_Gain_sample__(directoryGain5):
    global i5
    global GainSample5
    if i5 < len(os.listdir(directoryGain5)):
        ParseFilenameGain5 = os.path.join(directoryGain5, os.listdir(directoryGain5)[i5])
        print('Gain:' + ParseFilenameGain5)
        GainSample5, sr = librosa.load(ParseFilenameGain5, sr=16000)
        weight15 = torch.rand(1).item()
        GainSample5 = GainSample5 * weight15
        print(weight15)
    return GainSample5

def __get_Pitch_sample__(directoryPitch5):
    global i5
    global PitchSample5
    if i5 < len(os.listdir(directoryPitch5)):
        ParseFilenamePitch5 = os.path.join(directoryPitch5, os.listdir(directoryPitch5)[i5])
        print('Pitch:' + ParseFilenamePitch5)
        PitchSample5, sr = librosa.load(ParseFilenamePitch5, sr=16000)
        weight25 = torch.rand(1).item()
        PitchSample5 = PitchSample5*weight25
        print(weight25)
    return PitchSample5

def __get_PI_sample__(directoryPI5):
    global i5
    global PISample5
    if i5 < len(os.listdir(directoryGain5)):
        ParseFilenamePI5 = os.path.join(directoryPI5, os.listdir(directoryPI5)[i5])
        print('PI:' + ParseFilenamePI5)
        PISample5, sr = librosa.load(ParseFilenamePI5, sr=16000)
        weight35 = torch.rand(1).item()
        PISample5 = PISample5*weight35
        print(weight35)
    return PISample5


def _aug_mix__(original_sample5, directoryPitch5, directoryGain5,directoryPI5):
    #mix factor in range between 0 inclusive and 1 exclusixe
    #it almost never outputs 0
    sample5 = original_sample5.copy()
    mix_factor = torch.rand(1).item()
    sampleaug5 = __get_Pitch_sample__(directoryPitch5) + __get_Gain_sample__(directoryGain5)+ __get_PI_sample__(directoryPI5)
    sample5 = (sample5*mix_factor) + ((1-mix_factor)*sampleaug5)
    print(mix_factor)
    print(sample5)
    global i5
    i5 = i5+1
    return sample5

import os
directory5 ='C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\ToyCar\\train'
for filename5 in os.listdir(directory5):
    ParseFilename5 = os.path.join(directory5, filename5)
    print('ParseFilename:' + ParseFilename5)
    AUGSample5, sr = librosa.load(ParseFilename5, sr=16000)
    AUGSampleF5 = _aug_mix__(AUGSample5, directoryPitch5, directoryGain5, directoryPI5)
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\AUGMIXDEV\\ToyCar\\' + filename5 + ".wav")
                     , AUGSampleF5, sr, sampwidth=1)
print('TOYCAR COMPLETED')
#---------------------------------------------------------------------------------------------------------------------------------------------------------
global i6
i6=0
global directoryGain6
directoryGain6 = 'C:\\Users\\Mahmoud\\Desktop\\GainDEV\\ToyTrain'
global directoryPitch6
directoryPitch6 = 'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONDEV\\ToyTrain'
global directoryPI6
directoryPI6 = 'C:\\Users\\Mahmoud\\Desktop\\PolarityInversionDEV\\ToyTrain'

global GainSample6
global PitchSample6
global PISample6

def __get_Gain_sample__(directoryGain6):
    global i6
    global GainSample6
    if i6 < len(os.listdir(directoryGain6)):
        ParseFilenameGain6 = os.path.join(directoryGain6, os.listdir(directoryGain6)[i6])
        print('Gain:' + ParseFilenameGain6)
        GainSample6, sr = librosa.load(ParseFilenameGain6, sr=16000)
        weight16 = torch.rand(1).item()
        GainSample6 = GainSample6 * weight16
        print(weight16)
    return GainSample6

def __get_Pitch_sample__(directoryPitch6):
    global i6
    global PitchSample6
    if i6 < len(os.listdir(directoryPitch6)):
        ParseFilenamePitch6 = os.path.join(directoryPitch6, os.listdir(directoryPitch6)[i6])
        print('Pitch:' + ParseFilenamePitch6)
        PitchSample6, sr = librosa.load(ParseFilenamePitch6, sr=16000)
        weight26 = torch.rand(1).item()
        PitchSample6 = PitchSample6*weight26
        print(weight26)
    return PitchSample6

def __get_PI_sample__(directoryPI6):
    global i6
    global PISample6
    if i6 < len(os.listdir(directoryGain6)):
        ParseFilenamePI6 = os.path.join(directoryPI6, os.listdir(directoryPI6)[i6])
        print('PI:' + ParseFilenamePI6)
        PISample6, sr = librosa.load(ParseFilenamePI6, sr=16000)
        weight36 = torch.rand(1).item()
        PISample6 = PISample6*weight36
        print(weight36)
    return PISample6


def _aug_mix__(original_sample6, directoryPitch6, directoryGain6,directoryPI6):
    #mix factor in range between 0 inclusive and 1 exclusixe
    #it almost never outputs 0
    sample6 = original_sample6.copy()
    mix_factor = torch.rand(1).item()
    sampleaug6 = __get_Pitch_sample__(directoryPitch6) + __get_Gain_sample__(directoryGain6)+ __get_PI_sample__(directoryPI6)
    sample6 = (sample6*mix_factor) + ((1-mix_factor)*sampleaug6)
    print(mix_factor)
    print(sample6)
    global i6
    i6 = i6+1
    return sample6

import os
directory6 ='C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\ToyTrain\\train'
for filename6 in os.listdir(directory6):
    ParseFilename6 = os.path.join(directory6, filename6)
    print('ParseFilename:' + ParseFilename6)
    AUGSample6, sr = librosa.load(ParseFilename6, sr=16000)
    AUGSampleF6 = _aug_mix__(AUGSample6, directoryPitch6, directoryGain6, directoryPI6)
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\AUGMIXDEV\\ToyTrain\\' + filename6 + ".wav")
                     , AUGSampleF6, sr, sampwidth=1)
print('TOYTRAIN COMPLETED')
#-------------------------------------------------------------------------------------------------------------------------------------------------------
global i7
i7=0
global directoryGain7
directoryGain7 = 'C:\\Users\\Mahmoud\\Desktop\\GainDEV\\valve'
global directoryPitch7
directoryPitch7 = 'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONDEV\\valve'
global directoryPI7
directoryPI7 = 'C:\\Users\\Mahmoud\\Desktop\\PolarityInversionDEV\\valve'

global GainSample7
global PitchSample7
global PISample7

def __get_Gain_sample__(directoryGain7):
    global i7
    global GainSample7
    if i7 < len(os.listdir(directoryGain7)):
        ParseFilenameGain7 = os.path.join(directoryGain7, os.listdir(directoryGain7)[i7])
        print('Gain:' + ParseFilenameGain7)
        GainSample7, sr = librosa.load(ParseFilenameGain7, sr=16000)
        weight17 = torch.rand(1).item()
        GainSample7 = GainSample7 * weight17
        print(weight17)
    return GainSample7

def __get_Pitch_sample__(directoryPitch7):
    global i7
    global PitchSample7
    if i7 < len(os.listdir(directoryPitch7)):
        ParseFilenamePitch7 = os.path.join(directoryPitch7, os.listdir(directoryPitch7)[i7])
        print('Pitch:' + ParseFilenamePitch7)
        PitchSample7, sr = librosa.load(ParseFilenamePitch7, sr=16000)
        weight27 = torch.rand(1).item()
        PitchSample7 = PitchSample7*weight27
        print(weight27)
    return PitchSample7

def __get_PI_sample__(directoryPI7):
    global i7
    global PISample7
    if i7 < len(os.listdir(directoryGain7)):
        ParseFilenamePI7 = os.path.join(directoryPI7, os.listdir(directoryPI7)[i7])
        print('PI:' + ParseFilenamePI7)
        PISample7, sr = librosa.load(ParseFilenamePI7, sr=16000)
        weight37 = torch.rand(1).item()
        PISample7 = PISample7*weight37
        print(weight37)
    return PISample7


def _aug_mix__(original_sample7, directoryPitch7, directoryGain7,directoryPI7):
    #mix factor in range between 0 inclusive and 1 exclusixe
    #it almost never outputs 0
    sample7 = original_sample7.copy()
    mix_factor = torch.rand(1).item()
    sampleaug7 = __get_Pitch_sample__(directoryPitch7) + __get_Gain_sample__(directoryGain7)+ __get_PI_sample__(directoryPI7)
    sample7 = (sample7*mix_factor) + ((1-mix_factor)*sampleaug7)
    print(mix_factor)
    print(sample7)
    global i7
    i7 = i7+1
    return sample7

import os
directory7 ='C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\valve\\train'
for filename7 in os.listdir(directory7):
    ParseFilename7 = os.path.join(directory7, filename7)
    print('ParseFilename:' + ParseFilename7)
    AUGSample7, sr = librosa.load(ParseFilename7, sr=16000)
    AUGSampleF7 = _aug_mix__(AUGSample7, directoryPitch7, directoryGain7, directoryPI7)
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\AUGMIXDEV\\valve\\' + filename7 + ".wav")
                     , AUGSampleF7, sr, sampwidth=1)
print('VALVE DONE')
#---------------------------------------------------------------------------------------------------------------------------------------
###EVAL TRAINING DATASET ----> WHICH IS THE ""ADDITIONAL DATASET""
#-----------------------------------------------------------------------------------------------------------------------------------------
global i71
i71=0

global directoryGain71
directoryGain71 = 'C:\\Users\\Mahmoud\\Desktop\\GainEVAL\\valve'
global directoryPitch71
directoryPitch71 = 'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONEVAL\\valve'
global directoryPI71
directoryPI71 = 'C:\\Users\\Mahmoud\\Desktop\\PolarityInversionEVAL\\valve'

global GainSample71
global PitchSample71
global PISample71

def __get_Gain_sample__(directoryGain71):
    global i71
    global GainSample71
    if i71 < len(os.listdir(directoryGain71)):
        ParseFilenameGain71 = os.path.join(directoryGain71, os.listdir(directoryGain71)[i71])
        print('Gain:' + ParseFilenameGain71)
        GainSample71, sr = librosa.load(ParseFilenameGain71, sr=16000)
        weight711 = torch.rand(1).item()
        GainSample71 = GainSample71 * weight711
        print(weight711)
    return GainSample71

def __get_Pitch_sample__(directoryPitch71):
    global i71
    global PitchSample71
    if i71 < len(os.listdir(directoryPitch71)):
        ParseFilenamePitch71 = os.path.join(directoryPitch71, os.listdir(directoryPitch71)[i71])
        print('Pitch:' + ParseFilenamePitch71)
        PitchSample71, sr = librosa.load(ParseFilenamePitch71, sr=16000)
        weight712 = torch.rand(1).item()
        PitchSample71 = PitchSample71*weight712
        print(weight712)
    return PitchSample71

def __get_PI_sample__(directoryPI71):
    global i71
    global PISample71
    if i71 < len(os.listdir(directoryGain71)):
        ParseFilenamePI71 = os.path.join(directoryPI71, os.listdir(directoryPI71)[i71])
        print('PI:' + ParseFilenamePI71)
        PISample71, sr = librosa.load(ParseFilenamePI71, sr=16000)
        weight713 = torch.rand(1).item()
        PISample71 = PISample71*weight713
        print(weight713)
    return PISample71


def _aug_mix__(original_sample71, directoryPitch71, directoryGain71,directoryPI71):
    #mix factor in range between 0 inclusive and 1 exclusixe
    #it almost never outputs 0
    sample71 = original_sample71.copy()
    mix_factor = torch.rand(1).item()
    sampleaug71 = __get_Pitch_sample__(directoryPitch71) + __get_Gain_sample__(directoryGain71)+ __get_PI_sample__(directoryPI71)
    sample71 = (sample71*mix_factor) + ((1-mix_factor)*sampleaug71)
    print(mix_factor)
    print(sample71)
    global i71
    i71 = i71+1
    return sample71

import os
directory71 ='C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\eval_data\\valve\\train'
for filename71 in os.listdir(directory71):
    ParseFilename71 = os.path.join(directory71, filename71)
    print('ParseFilename:' + ParseFilename71)
    AUGSample71, sr = librosa.load(ParseFilename71, sr=16000)
    AUGSampleF71 = _aug_mix__(AUGSample71, directoryPitch71, directoryGain71, directoryPI71)
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\AUGMIXEVAL\\valve\\' + filename71 + ".wav")
                     , AUGSampleF71, sr, sampwidth=1)
print('VALVE DONE')
#----------------------------------------------------------------------------------------------------------------------------------------
global i61
i61 = 0

global directoryGain61
directoryGain61 = 'C:\\Users\\Mahmoud\\Desktop\\GainEVAL\\ToyCar'
global directoryPitch61
directoryPitch61 = 'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONEVAL\\ToyCar'
global directoryP61
directoryPI61 = 'C:\\Users\\Mahmoud\\Desktop\\PolarityInversionEVAL\\ToyCar'

global GainSample61
global PitchSample61
global PISample61

def __get_Gain_sample__(directoryGain61):
    global i61
    global GainSample61
    if i61 < len(os.listdir(directoryGain61)):
        ParseFilenameGain61 = os.path.join(directoryGain61, os.listdir(directoryGain61)[i61])
        print('Gain:' + ParseFilenameGain61)
        GainSample61, sr = librosa.load(ParseFilenameGain61, sr=16000)
        weight611 = torch.rand(1).item()
        GainSample61 = GainSample61 * weight611
        print(weight611)
    return GainSample61

def __get_Pitch_sample__(directoryPitch61):
    global i61
    global PitchSample61
    if i61 < len(os.listdir(directoryPitch61)):
        ParseFilenamePitch61 = os.path.join(directoryPitch61, os.listdir(directoryPitch61)[i61])
        print('Pitch:' + ParseFilenamePitch61)
        PitchSample61, sr = librosa.load(ParseFilenamePitch61, sr=16000)
        weight612 = torch.rand(1).item()
        PitchSample61 = PitchSample61*weight612
        print(weight612)
    return PitchSample61

def __get_PI_sample__(directoryPI61):
    global i61
    global PISample61
    if i61 < len(os.listdir(directoryGain61)):
        ParseFilenamePI61 = os.path.join(directoryPI61, os.listdir(directoryPI61)[i61])
        print('PI:' + ParseFilenamePI61)
        PISample61, sr = librosa.load(ParseFilenamePI61, sr=16000)
        weight613 = torch.rand(1).item()
        PISample61 = PISample61*weight613
        print(weight613)
    return PISample61


def _aug_mix__(original_sample61, directoryPitch61, directoryGain61,directoryPI61):
    #mix factor in range between 0 inclusive and 1 exclusixe
    #it almost never outputs 0
    sample61 = original_sample61.copy()
    mix_factor = torch.rand(1).item()
    sampleaug61 = __get_Pitch_sample__(directoryPitch61) + __get_Gain_sample__(directoryGain61)+ __get_PI_sample__(directoryPI61)
    sample61 = (sample61*mix_factor) + ((1-mix_factor)*sampleaug61)
    print(mix_factor)
    print(sample61)
    global i61
    i61 = i61+1
    return sample61

import os
directory61 ='C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\eval_data\\ToyCar\\train'
for filename61 in os.listdir(directory61):
    ParseFilename61 = os.path.join(directory61, filename61)
    print('ParseFilename:' + ParseFilename61)
    AUGSample61, sr = librosa.load(ParseFilename61, sr=16000)
    AUGSampleF61 = _aug_mix__(AUGSample61, directoryPitch61, directoryGain61, directoryPI61)
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\AUGMIXEVAL\\ToyCar\\' + filename61 + ".wav")
                     , AUGSampleF61, sr, sampwidth=1)
print('TOYCAR DONE')
#-------------------------------------------------------------------------------------------------------------------------------------------------
global i51
i51 = 0
global directoryGain51
directoryGain51 = 'C:\\Users\\Mahmoud\\Desktop\\GainEVAL\\slider'
global directoryPitch51
directoryPitch51 = 'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONEVAL\\slider'
global directoryPI51
directoryPI51 = 'C:\\Users\\Mahmoud\\Desktop\\PolarityInversionEVAL\\slider'

global GainSample51
global PitchSample51
global PISample51

def __get_Gain_sample__(directoryGain51):
    global i51
    global GainSample51
    if i51 < len(os.listdir(directoryGain51)):
        ParseFilenameGain51 = os.path.join(directoryGain51, os.listdir(directoryGain51)[i51])
        print('Gain:' + ParseFilenameGain51)
        GainSample51, sr = librosa.load(ParseFilenameGain51, sr=16000)
        weight511 = torch.rand(1).item()
        GainSample51 = GainSample51 * weight511
        print(weight511)
    return GainSample51

def __get_Pitch_sample__(directoryPitch51):
    global i51
    global PitchSample51
    if i51 < len(os.listdir(directoryPitch51)):
        ParseFilenamePitch51 = os.path.join(directoryPitch51, os.listdir(directoryPitch51)[i51])
        print('Pitch:' + ParseFilenamePitch51)
        PitchSample51, sr = librosa.load(ParseFilenamePitch51, sr=16000)
        weight512 = torch.rand(1).item()
        PitchSample51 = PitchSample51*weight512
        print(weight512)
    return PitchSample51

def __get_PI_sample__(directoryPI51):
    global i51
    global PISample51
    if i51 < len(os.listdir(directoryGain51)):
        ParseFilenamePI51 = os.path.join(directoryPI51, os.listdir(directoryPI51)[i51])
        print('PI:' + ParseFilenamePI51)
        PISample51, sr = librosa.load(ParseFilenamePI51, sr=16000)
        weight513 = torch.rand(1).item()
        PISample51 = PISample51*weight513
        print(weight513)
    return PISample51


def _aug_mix__(original_sample51, directoryPitch51, directoryGain51,directoryPI51):
    #mix factor in range between 0 inclusive and 1 exclusixe
    #it almost never outputs 0
    sample51 = original_sample51.copy()
    mix_factor = torch.rand(1).item()
    sampleaug51 = __get_Pitch_sample__(directoryPitch51) + __get_Gain_sample__(directoryGain51)+ __get_PI_sample__(directoryPI51)
    sample51 = (sample51*mix_factor) + ((1-mix_factor)*sampleaug51)
    print(mix_factor)
    print(sample51)
    global i51
    i51 = i51+1
    return sample51

import os
directory51 ='C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\eval_data\\slider\\train'
for filename51 in os.listdir(directory51):
    ParseFilename51 = os.path.join(directory51, filename51)
    print('ParseFilename:' + ParseFilename51)
    AUGSample51, sr = librosa.load(ParseFilename51, sr=16000)
    AUGSampleF51 = _aug_mix__(AUGSample51, directoryPitch51, directoryGain51, directoryPI51)
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\AUGMIXEVAL\\slider\\' + filename51 + ".wav")
                     , AUGSampleF51, sr, sampwidth=1)
print('SLIDER DONE')
#------------------------------------------------------------------------------------------------------------------------------------------

global i41
i41=0

global directoryGain41
directoryGain41 = 'C:\\Users\\Mahmoud\\Desktop\\GainEVAL\\pump'
global directoryPitch41
directoryPitch41 = 'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONEVAL\\pump'
global directoryPI41
directoryPI41 = 'C:\\Users\\Mahmoud\\Desktop\\PolarityInversionEVAL\\pump'

global GainSample41
global PitchSample41
global PISample41

def __get_Gain_sample__(directoryGain41):
    global i41
    global GainSample41
    if i41 < len(os.listdir(directoryGain41)):
        ParseFilenameGain41 = os.path.join(directoryGain41, os.listdir(directoryGain41)[i41])
        print('Gain:' + ParseFilenameGain41)
        GainSample41, sr = librosa.load(ParseFilenameGain41, sr=16000)
        weight411 = torch.rand(1).item()
        GainSample41 = GainSample41 * weight411
        print(weight411)
    return GainSample41

def __get_Pitch_sample__(directoryPitch41):
    global i41
    global PitchSample41
    if i41 < len(os.listdir(directoryPitch41)):
        ParseFilenamePitch41 = os.path.join(directoryPitch41, os.listdir(directoryPitch41)[i41])
        print('Pitch:' + ParseFilenamePitch41)
        PitchSample41, sr = librosa.load(ParseFilenamePitch41, sr=16000)
        weight412 = torch.rand(1).item()
        PitchSample41 = PitchSample41*weight412
        print(weight412)
    return PitchSample41

def __get_PI_sample__(directoryPI41):
    global i41
    global PISample41
    if i41 < len(os.listdir(directoryGain41)):
        ParseFilenamePI41 = os.path.join(directoryPI41, os.listdir(directoryPI41)[i41])
        print('PI:' + ParseFilenamePI41)
        PISample41, sr = librosa.load(ParseFilenamePI41, sr=16000)
        weight413 = torch.rand(1).item()
        PISample41 = PISample41*weight413
        print(weight413)
    return PISample41


def _aug_mix__(original_sample41, directoryPitch41, directoryGain41,directoryPI41):
    #mix factor in range between 0 inclusive and 1 exclusixe
    #it almost never outputs 0
    sample41 = original_sample41.copy()
    mix_factor = torch.rand(1).item()
    sampleaug41 = __get_Pitch_sample__(directoryPitch41) + __get_Gain_sample__(directoryGain41)+ __get_PI_sample__(directoryPI41)
    sample41 = (sample41*mix_factor) + ((1-mix_factor)*sampleaug41)
    print(mix_factor)
    print(sample41)
    global i41
    i41 = i41+1
    return sample41

import os
directory41 ='C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\eval_data\\pump\\train'
for filename41 in os.listdir(directory41):
    ParseFilename41 = os.path.join(directory41, filename41)
    print('ParseFilename:' + ParseFilename41)
    AUGSample41, sr = librosa.load(ParseFilename41, sr=16000)
    AUGSampleF41 = _aug_mix__(AUGSample41, directoryPitch41, directoryGain41, directoryPI41)
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\AUGMIXEVAL\\pump\\' + filename41 + ".wav")
                     , AUGSampleF41, sr, sampwidth=1)
print('PUMP DONE')
#---------------------------------------------------------------------------------------------------------------------------------------------
global i31
i31=0

global directoryGain31
directoryGain31 = 'C:\\Users\\Mahmoud\\Desktop\\GainEVAL\\gearbox'
global directoryPitch31
directoryPitch31 = 'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONEVAL\\gearbox'
global directoryPI31
directoryPI31 = 'C:\\Users\\Mahmoud\\Desktop\\PolarityInversionEVAL\\gearbox'

global GainSample31
global PitchSample31
global PISample31

def __get_Gain_sample__(directoryGain31):
    global i31
    global GainSample31
    if i31 < len(os.listdir(directoryGain31)):
        ParseFilenameGain31 = os.path.join(directoryGain31, os.listdir(directoryGain31)[i31])
        print('Gain:' + ParseFilenameGain31)
        GainSample31, sr = librosa.load(ParseFilenameGain31, sr=16000)
        weight311 = torch.rand(1).item()
        GainSample31 = GainSample31 * weight311
        print(weight311)
    return GainSample31

def __get_Pitch_sample__(directoryPitch31):
    global i31
    global PitchSample31
    if i31 < len(os.listdir(directoryPitch31)):
        ParseFilenamePitch31 = os.path.join(directoryPitch31, os.listdir(directoryPitch31)[i31])
        print('Pitch:' + ParseFilenamePitch31)
        PitchSample31, sr = librosa.load(ParseFilenamePitch31, sr=16000)
        weight312 = torch.rand(1).item()
        PitchSample31 = PitchSample31*weight312
        print(weight312)
    return PitchSample31

def __get_PI_sample__(directoryPI31):
    global i31
    global PISample31
    if i31 < len(os.listdir(directoryGain31)):
        ParseFilenamePI31 = os.path.join(directoryPI31, os.listdir(directoryPI31)[i31])
        print('PI:' + ParseFilenamePI31)
        PISample31, sr = librosa.load(ParseFilenamePI31, sr=16000)
        weight313 = torch.rand(1).item()
        PISample31 = PISample31*weight313
        print(weight313)
    return PISample31


def _aug_mix__(original_sample31, directoryPitch31, directoryGain31,directoryPI31):
    #mix factor in range between 0 inclusive and 1 exclusixe
    #it almost never outputs 0
    sample31 = original_sample31.copy()
    mix_factor = torch.rand(1).item()
    sampleaug31 = __get_Pitch_sample__(directoryPitch31) + __get_Gain_sample__(directoryGain31)+ __get_PI_sample__(directoryPI31)
    sample31 = (sample31*mix_factor) + ((1-mix_factor)*sampleaug31)
    print(mix_factor)
    print(sample31)
    global i31
    i31 = i31+1
    return sample31

import os
directory31 ='C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\eval_data\\gearbox\\train'
for filename31 in os.listdir(directory31):
    ParseFilename31 = os.path.join(directory31, filename31)
    print('ParseFilename:' + ParseFilename31)
    AUGSample31, sr = librosa.load(ParseFilename31, sr=16000)
    AUGSampleF31 = _aug_mix__(AUGSample31, directoryPitch31, directoryGain31, directoryPI31)
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\AUGMIXEVAL\\gearbox\\' + filename31 + ".wav")
                     , AUGSampleF31, sr, sampwidth=1)
print('GEARBOX DONE')
#---------------------------------------------------------------------------------------------------------------------------------------------------
global i21
i21=0

global directoryGain21
directoryGain21 = 'C:\\Users\\Mahmoud\\Desktop\\GainEVAL\\fan'
global directoryPitch21
directoryPitch21 = 'C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONEVAL\\fan'
global directoryPI21
directoryPI21 = 'C:\\Users\\Mahmoud\\Desktop\\PolarityInversionEVAL\\fan'

global GainSample21
global PitchSample21
global PISample21

def __get_Gain_sample__(directoryGain21):
    global i21
    global GainSample21
    if i21 < len(os.listdir(directoryGain21)):
        ParseFilenameGain21 = os.path.join(directoryGain21, os.listdir(directoryGain21)[i21])
        print('Gain:' + ParseFilenameGain21)
        GainSample21, sr = librosa.load(ParseFilenameGain21, sr=16000)
        weight211 = torch.rand(1).item()
        GainSample21 = GainSample21 * weight211
        print(weight211)
    return GainSample21

def __get_Pitch_sample__(directoryPitch21):
    global i21
    global PitchSample21
    if i21 < len(os.listdir(directoryPitch21)):
        ParseFilenamePitch21 = os.path.join(directoryPitch21, os.listdir(directoryPitch21)[i21])
        print('Pitch:' + ParseFilenamePitch21)
        PitchSample21, sr = librosa.load(ParseFilenamePitch21, sr=16000)
        weight212 = torch.rand(1).item()
        PitchSample21 = PitchSample21*weight212
        print(weight212)
    return PitchSample21

def __get_PI_sample__(directoryPI21):
    global i21
    global PISample21
    if i21 < len(os.listdir(directoryGain21)):
        ParseFilenamePI21 = os.path.join(directoryPI21, os.listdir(directoryPI21)[i21])
        print('PI:' + ParseFilenamePI21)
        PISample21, sr = librosa.load(ParseFilenamePI21, sr=16000)
        weight213 = torch.rand(1).item()
        PISample21 = PISample21*weight213
        print(weight213)
    return PISample21


def _aug_mix__(original_sample21, directoryPitch21, directoryGain21,directoryPI21):
    #mix factor in range between 0 inclusive and 1 exclusixe
    #it almost never outputs 0
    sample21 = original_sample21.copy()
    mix_factor = torch.rand(1).item()
    sampleaug21 = __get_Pitch_sample__(directoryPitch21) + __get_Gain_sample__(directoryGain21)+ __get_PI_sample__(directoryPI21)
    sample21 = (sample21*mix_factor) + ((1-mix_factor)*sampleaug21)
    print(mix_factor)
    print(sample21)
    global i21
    i21 = i21+1
    return sample21

import os
directory21 ='C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\eval_data\\fan\\train'
for filename21 in os.listdir(directory21):
    ParseFilename21 = os.path.join(directory21, filename21)
    print('ParseFilename:' + ParseFilename21)
    AUGSample21, sr = librosa.load(ParseFilename21, sr=16000)
    AUGSampleF21 = _aug_mix__(AUGSample21, directoryPitch21, directoryGain21, directoryPI21)
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\AUGMIXEVAL\\fan\\' + filename21 + ".wav")
                     , AUGSampleF21, sr, sampwidth=1)
print('FAN DONE')