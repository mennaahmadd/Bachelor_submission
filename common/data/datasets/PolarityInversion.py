import librosa
import wavio as wv


####POLARITY INVERSION WAS DONE ONCE FOR THE DEVELOPMENT TRAINING DATASET AND ONCE FOR THE EVALUATION TRAINING DATASET SEPARATELY BY CHANGING DIRECTORIES AND ADJUSTING SAVING DIRECTORIES#####
###IN ORDER TO RE-RUN THE FILE FIRST CHANGE DIRECTORIES TO YOUR OWN ONES THEN RUN ONCE FOR THE DEV DATA THEN CHANGE DIRECTORIES AND RUN FOR EVAL DATA#########


def Polarity_Inversion(signal):
    return signal * -1


import os
#directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\fan\\train'
directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\eval_data\\fan\\train'
for file in os.listdir(directory):
    ParseFilenamePI = os.path.join(directory, file)
    SamplePI, sr = librosa.load(ParseFilenamePI, sr=16000)
    AugmentedSamplePI = Polarity_Inversion(SamplePI)
    #'C:\\Users\\Mahmoud\Desktop\\PolarityInversionDEV\\fan\\'
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\PolarityInversionEVAL\\fan\\' + file + ".wav")
             , AugmentedSamplePI, sr, sampwidth=1)
print("FAN DONE")


import os
#directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\gearbox\\train'
directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\eval_data\\gearbox\\train'
for file in os.listdir(directory):
    ParseFilenamePI = os.path.join(directory, file)
    SamplePI, sr = librosa.load(ParseFilenamePI, sr=16000)
    AugmentedSamplePI = Polarity_Inversion(SamplePI)
    #'C:\\Users\\Mahmoud\Desktop\\PolarityInversionDEV\\gearbox\\'
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\PolarityInversionEVAL\\gearbox\\' + file + ".wav")
             , AugmentedSamplePI, sr, sampwidth=1)
print("GEARBOX DONE")



import os
#directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\pump\\train'
directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\eval_data\\pump\\train'
for file in os.listdir(directory):
    ParseFilenamePI = os.path.join(directory, file)
    SamplePI, sr = librosa.load(ParseFilenamePI, sr=16000)
    AugmentedSamplePI = Polarity_Inversion(SamplePI)
    #'C:\\Users\\Mahmoud\Desktop\\PolarityInversionDEV\\pump\\'
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\PolarityInversionEVAL\\pump\\' + file + ".wav")
             , AugmentedSamplePI, sr, sampwidth=1)
print("PUMP DONE")



import os
#directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\slider\\train'
directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\eval_data\\slider\\train'
for file in os.listdir(directory):
    ParseFilenamePI = os.path.join(directory, file)
    SamplePI, sr = librosa.load(ParseFilenamePI, sr=16000)
    AugmentedSamplePI = Polarity_Inversion(SamplePI)
    #'C:\\Users\\Mahmoud\Desktop\\PolarityInversionDEV\\slider\\'
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\PolarityInversionEVAL\\slider\\' + file + ".wav")
             , AugmentedSamplePI, sr, sampwidth=1)
print("SLIDER DONE")



import os
#directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\ToyCar\\train'
directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\eval_data\\ToyCar\\train'
for file in os.listdir(directory):
    ParseFilenamePI = os.path.join(directory, file)
    SamplePI, sr = librosa.load(ParseFilenamePI, sr=16000)
    AugmentedSamplePI = Polarity_Inversion(SamplePI)
    #'C:\\Users\\Mahmoud\Desktop\\PolarityInversionDEV\\ToyCar\\'
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\PolarityInversionEVAL\\ToyCar\\' + file + ".wav")
             , AugmentedSamplePI, sr, sampwidth=1)
print("ToyCar DONE")



import os
#directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\ToyTrain\\train'
directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\eval_data\\ToyTrain\\train'
for file in os.listdir(directory):
    ParseFilenamePI = os.path.join(directory, file)
    SamplePI, sr = librosa.load(ParseFilenamePI, sr=16000)
    AugmentedSamplePI = Polarity_Inversion(SamplePI)
    #'C:\\Users\\Mahmoud\Desktop\\PolarityInversionDEV\\ToyTrain\\'
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\PolarityInversionEVAL\\ToyTrain\\' + file + ".wav")
             , AugmentedSamplePI, sr, sampwidth=1)
print("TOYTRAIN DONE")


import os
#directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\dev_data\\valve\\train'
directory = 'C:\\Users\\Mahmoud\\Desktop\\ORIGINALLLL\\task2\\eval_data\\valve\\train'
for file in os.listdir(directory):
    ParseFilenamePI = os.path.join(directory, file)
    SamplePI, sr = librosa.load(ParseFilenamePI, sr=16000)
    AugmentedSamplePI = Polarity_Inversion(SamplePI)
    #'C:\\Users\\Mahmoud\Desktop\\PolarityInversionDEV\\valve\\'
    wv.write(os.path.join('C:\\Users\\Mahmoud\\Desktop\\PolarityInversionEVAL\\valve\\' + file + ".wav")
             , AugmentedSamplePI, sr, sampwidth=1)
print("VALVE DONE")
