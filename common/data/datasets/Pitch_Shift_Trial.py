import random

import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import torch
import torchaudio.transforms

#------------------------------------------------------------------------------------------------------------------------------
###THIS FILE IS FOR PLOTTING GAIN WAVE FORM, SPECTROGRAM, AND MEL-SPECTROGRAM OF BEFORE AND AFTER PITCH SHIFT EFFECT#######
### FOR THE FIRST TRAINING SAMPLE IN THE DEV DATASET OF GEARBOX#######
#------------------------------------------------------------------------------------------------------------------------------

def _augment_pitch_shiftTrial(sample):
    #list = [-2, -1 , 1 , 2]
    #n_steps = random.choice(list)
    n_steps = -2
    print(n_steps)
    return librosa.effects.pitch_shift(sample,sr=sr, n_steps=n_steps)


# First training sample in gearbox
audio_dataPS = r"C:\Users\Mahmoud\shared\DCASE2021\task2\dev_data\gearbox\train\section_00_source_train_normal_0000_0_g_25_mm_2000_mV_none.wav"
soundPS, sr = librosa.load(audio_dataPS, sr=16000, mono=True)
soundPS=soundPS[10000:14000]

#-----------------------------------------------------------------------------------------------------------------------------------

#Applying pitch shift to the audio file
AugmentedSamplePitch= _augment_pitch_shiftTrial(soundPS)

#Plot Time-representation waveform after pitch
librosa.display.waveshow(AugmentedSamplePitch)
plt.title('Time-domain representation after pitch-shift')
plt.ylabel('Amplitude')
plt.show()

#--------------------------------------------------------------------------------------
#plot frequency domain waveform after gain augmentation
AugP_FREQ = np.fft.fft(AugmentedSamplePitch)
AugP_FREQ_MAGNITUDE = np.absolute(AugP_FREQ)
AugPFrequency=np.linspace(0,sr,len(AugP_FREQ_MAGNITUDE))
AugPPart_of_magnitude=AugP_FREQ_MAGNITUDE[:int(len(AugP_FREQ_MAGNITUDE)/2)]
AugPPart_of_frequency=AugPFrequency[:int(len(AugPFrequency)/2)]
plt.plot(AugPPart_of_frequency,AugPPart_of_magnitude)
plt.title('Discrete Fourier Transform after Pitch-shift')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.show()
#------------------------------------------------------------------------
#plot log-mel-spectrogram after pitch
n_fft=1024
hop_length=512
n_mels=128
MEL_SpecPitch=librosa.feature.melspectrogram(y=AugmentedSamplePitch,sr=sr,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels)
LOGMEL_SPECPITCH=librosa.power_to_db(MEL_SpecPitch)
librosa.display.specshow(LOGMEL_SPECPITCH,
                         x_axis='time',
                         y_axis='mel',
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.title('Log Mel-Spectrogram after pitch-shift ')
plt.show()
