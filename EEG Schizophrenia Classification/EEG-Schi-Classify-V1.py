from glob import glob
import os
import mne
import numpy as np
import pandas
import matplotlib.pyplot as plt

all_file_path=glob('D:\Jupyter GR\EEG in Schizophrenia\*.edf')
print(len(all_file_path))

all_file_path[0]

healthy_file_path = [i for i in all_file_path if os.path.basename(i).startswith('h')]
patient_file_path = [i for i in all_file_path if not os.path.basename(i).startswith('h')]
print(len(healthy_file_path),len(patient_file_path))

def read_data(file_path):
  data=mne.io.read_raw_edf(file_path,preload=True)
  data.set_eeg_reference()
  data.filter(l_freq=0.5,h_freq=45)
  epochs=mne.make_fixed_length_epochs(data,duration=5,overlap=1)
  array=epochs.get_data()
  return array

sample_data=read_data(healthy_file_path[0])

sample_data.shape #no of epochs,channels,length of signal

control_epochs_array=[read_data(i) for i in healthy_file_path]
patient_epochs_array=[read_data(i) for i in patient_file_path]

control_epochs_array[0].shape

control_epochs_labels=[len(i)*[0] for i in control_epochs_array]
patient_epochs_labels=[len(i)*[1] for i in patient_epochs_array]
len(control_epochs_labels),len(patient_epochs_labels)

data_list=control_epochs_array+patient_epochs_array
label_list=control_epochs_labels+patient_epochs_labels

group_list=[[i]*len(j) for i,j in enumerate(data_list)]
len(group_list)

data_array=np.vstack(data_list)
label_array=np.hstack(label_list)
group_array=np.hstack(group_list)
print(data_array.shape,label_array.shape,group_array.shape)

#Feature Extraction
from scipy import stats
import numpy as np

def mean(x):
    return np.mean(x, axis=-1)

def std(x):
    return np.std(x, axis=-1)

def ptp(x):
    return np.ptp(x, axis=-1)

def var(x):
    return np.var(x, axis=-1)

def minin(x):
    return np.min(x, axis=-1)

def maxin(x):
    return np.max(x, axis=-1)

def argminin(x):
    return np.argmin(x, axis=-1)

def argmaxin(x):
    return np.argmax(x, axis=-1)

def rms(x):
    return np.sqrt(np.mean(x**2, axis=-1))

def abs_diff_signal(x):
    return np.sum(np.abs(np.diff(x, axis=-1)), axis=-1)

def skewness(x):
    return stats.skew(x, axis=-1)

def kurtosis(x):
    return stats.kurtosis(x, axis=-1)


def concatenate_feature(x):
    return np.concatenate([
        mean(x),
        std(x),
        ptp(x),
        var(x),
        minin(x),
        maxin(x),
        argminin(x),
        argmaxin(x),
        rms(x),
        abs_diff_signal(x),
        skewness(x),
        kurtosis(x)
    ], axis=1)


#by creating a loop to reshape the data
features = []
for d in data_array:
    d = d[np.newaxis, :, :]   # make it (1, 19, 1250)
    features.append(concatenate_feature(d))

feature_array = np.vstack(features)


# Model Implementation
#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV

# Model
clf = LogisticRegression(max_iter=1000)

# Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', clf)
])

# Parameters
param_grid = {
    'clf__C': [0.1, 0.5, 0.7, 1, 3, 5, 7]
}

# GroupKFold
gkf = GroupKFold(n_splits=5)

# Grid Search
gscv = GridSearchCV(pipe, param_grid, cv=gkf, n_jobs=-1)

# Fit
gscv.fit(feature_array, label_array, groups=group_array)

# Best score
print(gscv.best_score_)

#OutPut=0.6641522862011926

#Prediction
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

y_pred = cross_val_predict(pipe, feature_array, label_array, cv=gkf, groups=group_array)

print(classification_report(label_array, y_pred))
print(confusion_matrix(label_array, y_pred))
