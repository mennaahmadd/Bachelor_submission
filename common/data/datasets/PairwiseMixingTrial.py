import random

import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import torch
import torchaudio.transforms

#------------------------------------------------------------------------------------------------------------------------------
###THIS FILE IS FOR PLOTTING GAIN WAVE FORM, SPECTROGRAM, AND MEL-SPECTROGRAM OF BEFORE AND AFTER PAIRWISEMIXING EFFECT#######
### FOR THE FIRST TRAINING SAMPLE IN THE DEV DATASET OF GEARBOX#######
####There is no differenceee in the either in waveforms or in spectrograms before and after pairwisewise mixing
#------------------------------------------------------------------------------------------------------------------------------



global directory1
#random training audio from gearbox dev
directory1 = "C:\\Users\\Mahmoud\\shared\\DCASE2021\\task2\\dev_data\\gearbox\\train\\section_00_source_train_normal_0048_100_g_25_mm_1000_mV_none.wav"

def _pairwise_mixing__(original_sample,directory1) -> dict:
    #mix factor in range between 0 inclusive and 1 exclusixe
    #it almost never outputs 0
    sample = original_sample.copy()
    sound2,sr = librosa.load(directory1, sr=16000, mono=True)
    sound2=sound2[10000:14000]
    print(sound2)
    mix_factor = torch.rand(1).item()
    print(mix_factor)
    sample = sample * mix_factor + sound2 * (1-mix_factor)
    return sample


# First training sample in gearbox
audio_dataPairwise = r"C:\Users\Mahmoud\shared\DCASE2021\task2\dev_data\gearbox\train\section_00_source_train_normal_0000_0_g_25_mm_2000_mV_none.wav"
soundAUGMix, sr = librosa.load(audio_dataPairwise, sr=16000, mono=True)
soundAUGMix=soundAUGMix[10000:14000]
#-------------------------------------------------------------------------------------------------------------------------------------------------------

#Applying Pairwise-Mixing to the audio file
AugmentedSampleTrialPairwise= _pairwise_mixing__(soundAUGMix,directory1)

#Plot waveform
librosa.display.waveshow(AugmentedSampleTrialPairwise)
plt.title('Time-domain representation after pairwise-mixing')
plt.ylabel('Amplitude')
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------
#plot frequency domain waveform after gain augmentation
AugPW_FREQ = np.fft.fft(AugmentedSampleTrialPairwise)
AugPW_FREQ_MAGNITUDE = np.absolute(AugPW_FREQ)
AugPWFrequency=np.linspace(0,sr,len(AugPW_FREQ_MAGNITUDE))
AugPWPart_of_frequency=AugPWFrequency[:int(len(AugPWFrequency)/2)]
AugPWPart_of_magnitude=AugPW_FREQ_MAGNITUDE[:int(len(AugPW_FREQ_MAGNITUDE)/2)]
plt.plot(AugPWPart_of_frequency,AugPWPart_of_magnitude)
plt.title('Discrete Fourier Transform after pairwise-mixing')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

#LOG MELSPEC AFTER PARIWISE
n_fft=1024
hop_length=512
n_mels=128
MEL_SpecPAIR=librosa.feature.melspectrogram(y=AugmentedSampleTrialPairwise,sr=sr,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels)
LOGMEL_SPECPAIR=librosa.power_to_db(MEL_SpecPAIR)
librosa.display.specshow(LOGMEL_SPECPAIR,
                         x_axis='time',
                         y_axis='mel',
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.title('Log Mel-Spectrogram after pairwise-mixing')
plt.show()