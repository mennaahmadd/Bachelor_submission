import os
import random

import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import pandas as pd
import torch
from torch_audiomentations.utils.io import Audio

#------------------------------------------------------------------------------------------------------------------------------
###THIS FILE IS FOR PLOTTING GAIN WAVE FORM, SPECTROGRAM, AND MEL-SPECTROGRAM OF BEFORE AND AFTER GAIN EFFECT#######
### FOR THE FIRST TRAINING SAMPLE IN THE DEV DATASET OF GEARBOX#######
#------------------------------------------------------------------------------------------------------------------------------
def _augment_Gain_Trial(sample, low: float = 1.25, high: float = 6.25):
    Gain = random.uniform(low, high)
    print(Gain)
    sample = sample * Gain
    print(sample)
    return sample
#----------------------------------------------------------------------------------------------------------------------------------------------------------------

# First training sample in gearbox
audio_dataGAIN = r"C:\Users\Mahmoud\shared\DCASE2021\task2\dev_data\gearbox\train\section_00_source_train_normal_0000_0_g_25_mm_2000_mV_none.wav"
G, sr = librosa.load(audio_dataGAIN, sr=16000, mono=True)
print(G.shape)
print (G)
print(sr)
G=G[10000:14000]
#plot time-domain waveform without augmentations
librosa.display.waveshow(G)
plt.title('Time-domain representation')
plt.ylabel('Amplitude')
plt.show()

#----------------------------------------------------------------------------

#plot frequency domain waveform without augmentations
G_FREQ = np.fft.fft(G)
G_FREQ_MAGNITUDE = np.absolute(G_FREQ)
Frequency=np.linspace(0,sr,len(G_FREQ_MAGNITUDE))
Part_of_frequency=Frequency[:int(len(Frequency)/2)]
Part_of_magnitude=G_FREQ_MAGNITUDE[:int(len(G_FREQ_MAGNITUDE)/2)]
plt.plot(Part_of_frequency,Part_of_magnitude)
plt.title('Discrete Fourier Transform')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.show()
#---------------------------------------------------------------------------------
#LOGMELSPEC before augmentation
n_fft=1024
hop_length=512
n_mels=128
MEL_Spec=librosa.feature.melspectrogram(G,sr=sr,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels)
LOGMEL_SPEC=librosa.power_to_db(MEL_Spec)
librosa.display.specshow(LOGMEL_SPEC,
                         x_axis='time',
                         y_axis='mel',
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.title('Log Mel-Spectrogram')
plt.show()
#-----------------------------------------------------------------------
#------------------------------------------------------------------------

#Applying Gain to the sample
AugmentedSampleTrialGain= _augment_Gain_Trial(G)

#--------------------------------------------------------------------------

#plot waveform after Gain
librosa.display.waveshow(AugmentedSampleTrialGain)
plt.title('Time-domain representation after gain')
plt.ylabel('Amplitude')
plt.show()
#--------------------------------------------------------------------------------------

#plot frequency domain waveform after gain augmentation
AugG_FREQ = np.fft.fft(AugmentedSampleTrialGain)
AugG_FREQ_MAGNITUDE = np.absolute(AugG_FREQ)
AugFrequency=np.linspace(0,sr,len(AugG_FREQ_MAGNITUDE))
AugPart_of_frequency=AugFrequency[:int(len(AugFrequency)/2)]
AugPart_of_magnitude=AugG_FREQ_MAGNITUDE[:int(len(AugG_FREQ_MAGNITUDE)/2)]
plt.plot(AugPart_of_frequency,AugPart_of_magnitude)
plt.title('Discrete Fourier Transform after gain')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.show()



#----------------------------------------------------------------------------------------
#LOGMELSPEC AFTER GAIN
n_fft=1024
hop_length=512
n_mels=128
MEL_SpecGAIN=librosa.feature.melspectrogram(y=AugmentedSampleTrialGain,sr=sr,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels)
LOGMEL_SPECGAIN=librosa.power_to_db(MEL_SpecGAIN)
librosa.display.specshow(LOGMEL_SPECGAIN,
                         x_axis='time',
                         y_axis='mel',
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.title('Log Mel-Spectrogram after gain')
plt.show()