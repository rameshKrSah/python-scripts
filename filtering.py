# Importing necessary libraries
from scipy.signal import butter, filtfilt, lfilter
  
'''
Low pass filter to remove noise specially artifact noise
'''
def butter_lowpassfilter(data, cutoff, sample_rate, order=2):
  '''standard lowpass filter.
    Function that filters the data using standard Butterworth lowpass filter
	
    Parameters
    ----------
	data : 1-d array
        array containing the gsr data
    cutoff : int or float
        frequency in Hz that acts as cutoff for filter.
    sample_rate : int or float
        sample rate of the supplied signal
    order : int
        filter order, defines the strength of the roll-off
        around the cutoff frequency.
        default: 2
    
    Returns
    -------
    y : 1-d array
        filtered gsr data
  '''
  nyq = 0.5 * sample_rate
  normal_cutoff = cutoff/nyq
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  y = filtfilt(b, a, data)
  return y


# Low Pass Filter removes noise from the EDA data  https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def eda_lpf(order = 1, fs = 4, cutoff = 5):
    nyq = 0.5 * fs
    low = cutoff / nyq
    b, a = butter(order, low, btype='lowpass', analog=True)
    return b, a

def butter_lowpass_filter_eda(data):
    b, a = eda_lpf()
    y = lfilter(b, a, data)
    return y

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
