import os
# import sys
import numpy as np
# import pandas as pd
# import pickle
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt, freqz
# from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
# from sklearn.metrics import average_precision_score
# from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve
# from sklearn.metrics import classification_report, confusion_matrix
import my_utils as gb_utl

# import preprocessing as eda_preprocessing
# import filtering as eda_filtering
# import cvxEDA as cvx 

# Low Pass Filter removes noise from the EDA data  https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def eda_lpf(order = 4, fs = 4, cutoff = 1):
    """
        The sampling frequency is 4 Hz and since we have 1 sample every second,
        the cutoff frequency is set to 1 Hz. Also, since the data is sampled
        at regular interval we set analog to false.

        https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units

        https://dsp.stackexchange.com/questions/19084/applying-filter-in-scipy-signal-use-lfilter-or-filtfilt
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a

def butter_lowpass_filter_eda(data):
    """
    Lowpass Butterworth filter for EDA data with cutoff frequency of 1 Hz and
    order 4
    """
    b, a = eda_lpf()
    y = filtfilt(b, a, data)
    return y

def plot_freq_response(fs, cutoff):
    b, a = eda_lpf()
    w, h = freqz(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()

# High Pass Filter is used to separate the SCL and SCR components from the EDA signal
def eda_hpf(order = 1, fs = 4, cutoff = 0.05):
    nyq = 0.5 * fs
    high = cutoff / nyq
    b, a = butter(order, high, btype='highpass')
    return b, a

def butter_highpass_filter_eda(data):
    b, a = eda_hpf()
    y = lfilter(b, a, data)
    return y


def combine_class_data(baseline_path, amusement_path, stressed_path, include_amusement = False):
    """
        Load the data for different stress class for the WESAD dataset and return X, Y. 
        If amusement is included into the baseline class, the labels assigned to amusement is 0 
        same as that of baseline class. 
            
        baseline_path (string): path to baseline data, 
        amusement_path (string): path to the amusement data,
        stressed_path (string): path to stressed data
        include_amusement (Boolean): whether to include amusement data into baseline or not. 
        By default amusement data is not included into baseline.
            
        X, Y : NumPy arrays.
    """
    stress_label = 1
    not_stress_label = 0
    
    # load the segments
    baseline_segments = gb_utl.read_data(baseline_path)
    stress_segments = gb_utl.read_data(stressed_path)
    
    # combine the baseline and stress segments
    X = np.concatenate([baseline_segments, stress_segments], axis = 0)
    Y = np.concatenate([np.zeros(baseline_segments.shape[0], dtype=int), 
                       np.ones(stress_segments.shape[0], dtype=int)
                       ])
    # include the amusement data is indicated
    if include_amusement:
        amusement_segments = gb_utl.read_data(amusement_path)
        X = np.concatenate([X, amusement_segments])
        Y = np.concatenate([Y, np.zeros(amusement_segments.shape[0], dtype=int)])
    
    return X, Y
    
    
def check_continuity(array):
    """
        Check whether the array contains continous values or not like 1, 2, 3, 4, ..
    """
    max_v = max(array)
    min_v = min(array)
    n = len(array)
#     print(n, min_v, max_v)
    if max_v - min_v + 1 == n:
#         print("Given array has continous values")
        return True
    else:
#         print("Given array is not continous")
        return False
        
