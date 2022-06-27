import librosa
import torch
import random
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import torch
#------------------------------------------------------------------------------------------------------------------------------
###THIS FILE IS FOR PLOTTING GAIN WAVE FORM, SPECTROGRAM, AND MEL-SPECTROGRAM OF BEFORE AND AFTER AUGMIX EFFECT#######
### FOR THE FIRST TRAINING SAMPLE IN THE DEV DATASET OF GEARBOX#######
#------------------------------------------------------------------------------------------------------------------------------

global directoryGain
directoryGain = r"C:\\Users\\Mahmoud\\Desktop\\GainDEV\\gearbox\\section_00_source_train_normal_0000_0_g_25_mm_2000_mV_none.wav.wav"
global directoryPitch
directoryPitch = r"C:\\Users\\Mahmoud\\Desktop\\NEWFOLDER PITCH SHIFT WITHOUT ZERO SECTIONDEV\\gearbox\\section_00_source_train_normal_0000_0_g_25_mm_2000_mV_none.wav.wav"
global directoryPI
directoryPI = r"C:\\Users\\Mahmoud\\Desktop\\PolarityInversionDEV\\gearbox\\section_00_source_train_normal_0000_0_g_25_mm_2000_mV_none.wav.wav"

global GainSample
global PitchSample
global PISample

def __get_Gain_sample__(directoryGain):
    global GainSample
    GainSample, sr = librosa.load(directoryGain, sr=16000)
    GainSample=GainSample[10000:14000]
    weight1 = torch.rand(1).item()
    GainSample = GainSample * weight1
    print('GAIN')
    print(weight1)
    return GainSample

def __get_Pitch_sample__(directoryPitch):
    global PitchSample
    PitchSample, sr = librosa.load(directoryPitch, sr=16000)
    PitchSample=PitchSample[10000:14000]
    weight2 = torch.rand(1).item()
    PitchSample = PitchSample*weight2
    print(weight2)
    return PitchSample

def __get_PI_sample__(directoryPI):
    global PISample
    PISample, sr = librosa.load(directoryPI, sr=16000)
    PISample=PISample[10000:14000]
    weight3 = torch.rand(1).item()
    PISample = PISample*weight3
    print('PISAMPLE')
    print(weight3)
    return PISample

def _aug_mix__(original_sample, directoryPitch, directoryGain,directoryPI):
    sample = original_sample.copy()
    mix_factor = torch.rand(1).item()
    print(mix_factor)
    sampleaug = __get_Pitch_sample__(directoryPitch) + __get_Gain_sample__(directoryGain)+ __get_PI_sample__(directoryPI)
    sample = (sample*mix_factor) + ((1-mix_factor)*sampleaug)
    return sample


audio_dataAUGMix = r"C:\Users\Mahmoud\shared\DCASE2021\task2\dev_data\gearbox\train\section_00_source_train_normal_0000_0_g_25_mm_2000_mV_none.wav"

AUGMIX1, sr = librosa.load(audio_dataAUGMix, sr=16000, mono=True)
AUGMIX1=AUGMIX1[10000:14000]
#load augmix sample
AUGSampleF = _aug_mix__(AUGMIX1, directoryPitch, directoryGain, directoryPI)


#augmix waveform
librosa.display.waveshow(AUGSampleF)
plt.title('Time-domain representation after AUGMix')
plt.ylabel('Amplitude')
plt.show()
#--------------------------------------------------------------------------------------
#plot frequency domain waveform after gain augmentation
AugMix_FREQ = np.fft.fft(AUGSampleF)
AugMix_FREQ_MAGNITUDE = np.absolute(AugMix_FREQ)
AugMixFrequency=np.linspace(0,sr,len(AugMix_FREQ_MAGNITUDE))
AugMixPart_of_magnitude=AugMix_FREQ_MAGNITUDE[:int(len(AugMix_FREQ_MAGNITUDE)/2)]
AugMixPart_of_frequency=AugMixFrequency[:int(len(AugMixFrequency)/2)]
plt.plot(AugMixPart_of_frequency,AugMixPart_of_magnitude)
plt.title('Discrete Fourier Transform after AUGMix')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.show()
#--------------------------------------------------------------------------
#LOGMELSPEC AFTER AUGMIX
n_fft=1024
hop_length=512
n_mels=128
MEL_SpeAUGMIX=librosa.feature.melspectrogram(y=AUGSampleF,sr=sr,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels)
LOGMEL_SPECAUGMIX=librosa.power_to_db(MEL_SpeAUGMIX)
librosa.display.specshow(LOGMEL_SPECAUGMIX,
                         x_axis='time',
                         y_axis='mel',
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.title('Log Mel-Spectrogram after AUGMix')
plt.show()
