import librosa
from matplotlib import pyplot as plt
import librosa.display

#---------------------------------------------------------------------------------------------------------------------------------------------
###THIS FILE IS TO PLOT A MEL SPECTROGRAM OF THE FIRST AUDIO IN THE DEV DATASET OF GEARBOX WIRH 32 MEL FILTERS INSTED OF 128 ###########
###------------------------------------------------------------------------------------------------------------------------------------------

audio_data32 = r"C:\Users\Mahmoud\shared\DCASE2021\task2\dev_data\gearbox\train\section_00_source_train_normal_0000_0_g_25_mm_2000_mV_none.wav"

sound32, sr = librosa.load(audio_data32, sr=16000, mono=True)
print(sr)
sound32=sound32[10000:14000]

#plotting the log mel spectrogram
n_fft=1024
hop_length=512
n_mels=32
MelSpecT128 = librosa.feature.melspectrogram(y=sound32, sr=sr, n_fft= n_fft, hop_length= hop_length, n_mels= n_mels)
log_mel_Spec128DB = librosa.power_to_db(MelSpecT128)
librosa.display.specshow(log_mel_Spec128DB,sr=sr, x_axis='time', y_axis='mel')
plt.colorbar()
plt.title('Log Mel-Spectrogram with 32 Mel-filters')
plt.show()