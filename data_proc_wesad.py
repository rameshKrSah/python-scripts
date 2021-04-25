
import os
import sys
import numpy as np
import pandas as pd
import pickle

import preprocessing as eda_preprocessing
import filtering as eda_filtering
import Utils as utl
import cvxEDA as cvx

# data directory
data_dir = "../Data/WESAD/

# folder path to save the data
data_save_folder = "../Processed Data/WESAD/"

# sampling frequency for labels
label_SF = 700

# all RespiBan sensors are sampled at 700 Hz
RESPI_BAN_SF = 700

# sampling frequency for E4
E4_ACC_SF = 32
E4_EDA_SF = 4
E4_TEMP_SF = 4
E4_BVP_SF = 64
E4_HR_SF = 1
ACC_CHANNEL = 3

EDA_CUTOFF_FREQ = 5.0/ E4_EDA_SF
print("EDA LPF Cut off frequency ", EDA_CUTOFF_FREQ)

# valid labels
baseline_label = 1
stress_label = 2
amusement_label = 3
meditation_label =  4
invliad_labels = [0, 5, 6, 7]

# keys in the dictionary
data_keys = ["signal", "label", "subject"]
data_keys_str_to_int = {"signal": 0, "label": 1, "subject_id": 2}

sensor_keys = ['wrist', 'chest']
sensor_keys_str_to_int = {"wrist":0, "chest":1}

respiBan_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
respiBan_keys_str_to_int = {"ACC": 0, "ECG": 1, "EDA":2, "EMG": 3, "Resp": 4, "Temp": 5}

E4_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
E4_keys_str_to_int = {"ACC":0, "BVP":1, "EDA":2, "TEMP":3}

def get_subject_data_e4(subject_data_file, debug=False):
    """
        @brief: Returns all E4 sensor data for one subject.
        @param: subject_data_file (string): path to the subject pickle file.
        
        @return: Returns the sensor data and labels. All numpy arrays. For E4 we have EDA, BVP, ACC, TEMP, 
        and labels.
    """
    
    # open the subject data file
    try:
        file = open(subject_data_file, "rb")
        data = pickle.load(file, encoding='latin1')
        file.close()
    except ValueError:
        print("Can't open file {}".format(subject_data_file))
        return
    
    """
        Now we need to syncronize the sensor data based on the labels, and labels are sampled at 700 Hz. 
        Hence, 700 entires of labels for each second of sensor values. Also, different sensors has 
        different sampling frequency. 
    """
    
    # get the labels
    data_label = data['label']
    
    # get the sensor data
    data = data['signal']['wrist']
    
    # get the EDA sensor values
    data_EDA = data['EDA']
    
    # get the ACC sensor values
    data_ACC = data["ACC"]
    
    # get the BVP values for Wrist E4 sensor
    data_BVP = data["BVP"]
    
    # get the temperature sensor 
    data_TEMP = data["TEMP"]
    
#     print(data_EDA.shape[0] / E4_EDA_SF)
#     print(sum(np.unique(data_label, return_counts=True)[1]) / label_SF)
    
    # we need to process the label, so that 700 entires for each second is merged as one label for each second.
    total_entries = len(data_label)
    processed_labels = []
    
    # data containers
    processed_EDA = []
    processed_ACC = []
    processed_BVP = []
    processed_TEMP = []
    
#     print(total_entries, total_entries // label_SF)
    if total_entries % label_SF != 0:
        # not able to divide the label entires
        print("Not able to divide the label entries by 700.")
    else:
        eda_start = 0
        eda_end = E4_EDA_SF
        
        acc_start = 0
        acc_end = E4_ACC_SF
        
        bvp_start = 0
        bvp_end = E4_BVP_SF

        temp_start = 0
        temp_end = E4_TEMP_SF
        
        l_start = 0
        l_end = label_SF
        
        for i in range(total_entries // label_SF):
            # get the labels
            l_entries = data_label[l_start:l_end]
            
            # check whether label entires have only one value or not
            labels = np.unique(l_entries)
            if len(labels) == 1:
                # get the label
                label = labels[0]
                
                # if the label is not transient or invalid states then save the sensor values
                if label not in invliad_labels:
                    # get the EDA
                    eda_entries = data_EDA[eda_start:eda_end]

                    # get the ACC
                    acc_entries = data_ACC[acc_start:acc_end]

                    # get the BVP 
                    bvp_entries = data_BVP[bvp_start:bvp_end]

                    # get the TEMP
                    temp_entries = data_TEMP[temp_start:temp_end]

                    # save the sensor values and label for this one second second
                    processed_labels.append(label)
                    processed_EDA.append(eda_entries)
                    processed_BVP.append(bvp_entries)
                    processed_ACC.append(acc_entries)
                    processed_TEMP.append(temp_entries)
            
            # no matter whether we saved the sensor values or not, we increment the index pointers
            # increment the position values
            l_start = l_end
            l_end = l_end + label_SF
            
            eda_start = eda_end
            eda_end = eda_end + E4_EDA_SF
            
            acc_start = acc_end
            acc_end = acc_end + E4_ACC_SF
         
            bvp_start = bvp_end
            bvp_end = bvp_end + E4_BVP_SF
            
            temp_start = temp_end
            temp_end = temp_end + E4_TEMP_SF
            
    # filter and normalize the EDA data
    eda_length = len(processed_EDA)
    processed_EDA = eda_filtering.butter_lowpassfilter(np.array(processed_EDA).ravel(), 
                                                       EDA_CUTOFF_FREQ, E4_EDA_SF, 
                                                       order=2)
    processed_EDA = eda_preprocessing.normalization(processed_EDA)
    
    # create numpy arrays from the sensor data and labels with proper shape
    processed_ACC = np.array(processed_ACC).reshape(-1, ACC_CHANNEL, E4_ACC_SF)
    processed_BVP = np.array(processed_BVP).reshape(-1, E4_BVP_SF)
    processed_EDA = np.array(processed_EDA).reshape(eda_length, E4_EDA_SF)
    processed_TEMP = np.array(processed_TEMP).reshape(-1, E4_TEMP_SF)
    processed_labels = np.array(processed_labels, dtype = int)
    
    return processed_EDA, processed_BVP, processed_ACC, processed_TEMP, processed_labels


def get_all_data_e4(save_folder):
    """
        @brief: Extract all E4 data stream: EDA, TEMP, ACC, and BVP. We filter the EDA data
        with a low pass filter and also normalize the data in the range (0, 1). Each sensor
        modality is separated into different class: baseline, amusement, and stress. We 
        save each individual subject data and also the total combined data.

        @param: save_folder: Path to the folder to store the data
    """
    # the meditation class is not continous, 
    # hence we will not consider data from the meditation class
    total_baseline_eda = []
    total_baseline_acc = []
    total_baseline_temp = []
    total_baseline_bvp = []

    total_stress_eda = []
    total_stress_acc = []
    total_stress_temp = []
    total_stress_bvp = []

    total_amusement_eda = []
    total_amusement_acc = []
    total_amusement_temp = []
    total_amusement_bvp = []

    sensor_name = "_E4_"

    # for each subject, get the sensor values for each class and store it
    print("the data directory is ", data_dir)
    for dir_ in os.listdir(data_dir):
        subject_name = dir_
        subject_file_path = data_dir+dir_+"/"+dir_+".pkl"

        if "pdf" not in subject_file_path:
            print("Subject name {}, \n Path {}".format(subject_name, subject_file_path))

            # get the sensor values and labels
            eda_values, bvp_values, acc_values, temp_values, labels = \
            get_subject_data_e4(subject_file_path)

            print(eda_values.shape, len(labels))

            # store sensor values for each subject separately for each class
            baseline_index = np.where(labels == baseline_label)[0]
            stress_index = np.where(labels == stress_label)[0]
            amusement_index = np.where(labels == amusement_label)[0]
            meditation_index = np.where(labels == meditation_label)[0]

            baseline_eda = eda_values[baseline_index].ravel()
            stress_eda = eda_values[stress_index].ravel()
            amusement_eda = eda_values[amusement_index].ravel()

            baseline_acc = acc_values[baseline_index]
            stress_acc = acc_values[stress_index]
            amusement_acc = acc_values[amusement_index]

            baseline_temp = temp_values[baseline_index].ravel()
            stress_temp = temp_values[stress_index].ravel()
            amusement_temp = temp_values[amusement_index].ravel()

            baseline_bvp = bvp_values[baseline_index].ravel()
            stress_bvp =  bvp_values[stress_index].ravel()
            amusement_bvp = bvp_values[amusement_index].ravel()

            # save the data for this subject
            uti.save_data(save_folder+subject_name+sensor_name+"baseline_eda",
                               baseline_eda)
            uti.save_data(save_folder+subject_name+sensor_name+"baseline_temp",
                               baseline_temp)
            uti.save_data(save_folder+subject_name+sensor_name+"baseline_acc",
                               baseline_acc)
            uti.save_data(save_folder+subject_name+sensor_name+"baseline_bvp",
                               baseline_bvp)


            uti.save_data(save_folder+subject_name+sensor_name+"amusement_eda",
                               amusement_eda)
            uti.save_data(save_folder+subject_name+sensor_name+"amusement_temp",
                               amusement_temp)
            uti.save_data(save_folder+subject_name+sensor_name+"amusement_acc",
                               amusement_acc)
            uti.save_data(save_folder+subject_name+sensor_name+"amusement_bvp",
                               amusement_bvp)


            uti.save_data(save_folder+subject_name+sensor_name+"stress_eda",
                               stress_eda)
            uti.save_data(save_folder+subject_name+sensor_name+"stress_temp",
                               stress_temp)
            uti.save_data(save_folder+subject_name+sensor_name+"stress_acc",
                               stress_acc)
            uti.save_data(save_folder+subject_name+sensor_name+"stress_bvp",
                               stress_bvp)


            # store the values
            total_baseline_eda.append(baseline_eda)
            total_baseline_acc.append(baseline_acc)
            total_baseline_temp.append(baseline_temp)
            total_baseline_bvp.append(baseline_bvp)

            total_stress_eda.append(stress_eda)
            total_stress_acc.append(stress_acc)
            total_stress_temp.append(stress_temp)
            total_stress_bvp.append(stress_bvp)

            total_amusement_eda.append(amusement_eda)
            total_amusement_acc.append(amusement_acc)
            total_amusement_temp.append(amusement_temp)
            total_amusement_bvp.append(amusement_bvp)

    # save the total data for all subjects
    total_baseline_eda = np.array(total_baseline_eda)
    total_baseline_temp = np.array(total_baseline_temp)
    total_baseline_acc = np.array(total_baseline_acc)
    total_baseline_bvp = np.array(total_baseline_bvp)

    total_amusement_eda = np.array(total_amusement_eda)
    total_amusement_temp = np.array(total_amusement_temp)
    total_amusement_acc = np.array(total_amusement_acc)
    total_amusement_bvp = np.array(total_amusement_bvp)

    total_stress_eda = np.array(total_stress_eda)
    total_stress_temp = np.array(total_stress_temp)
    total_stress_acc = np.array(total_stress_acc)
    total_stress_bvp = np.array(total_stress_bvp)

    uti.save_data(save_folder+sensor_name[1:]+"baseline_eda", 
                        total_baseline_eda)
    uti.save_data(save_folder+sensor_name[1:]+"baseline_temp", 
                        total_baseline_temp)
    uti.save_data(save_folder+sensor_name[1:]+"baseline_acc", 
                        total_baseline_acc)
    uti.save_data(save_folder+sensor_name[1:]+"baseline_bvp", 
                        total_baseline_bvp)

    uti.save_data(save_folder+sensor_name[1:]+"amusement_eda", 
                        total_amusement_eda)
    uti.save_data(save_folder+sensor_name[1:]+"amusement_temp", 
                        total_amusement_temp)
    uti.save_data(save_folder+sensor_name[1:]+"amusement_acc", 
                        total_amusement_acc)
    uti.save_data(save_folder+sensor_name[1:]+"amusement_bvp", 
                        total_amusement_bvp)

    uti.save_data(save_folder+sensor_name[1:]+"stress_eda", 
                        total_stress_eda)
    uti.save_data(save_folder+sensor_name[1:]+"stress_temp", 
                        total_stress_temp)
    uti.save_data(save_folder+sensor_name[1:]+"stress_acc", 
                        total_stress_acc)
    uti.save_data(save_folder+sensor_name[1:]+"stress_bvp", 
                        total_stress_bvp)

def get_phasic_tonic_components(processed_eda, sample_rate, save=True, name="", save_folder=""):
    """
        @brief: Separate the phasic and tonic component of the given preprocessed EDA.
        @param: processed_eda: filtered and normalized eda values for a class, for example
            baseline. We expect the processed_eda to have n arrays. 
        @param: sample_rate: sample rate for the EDA in Hz
        @param: save: A Boolean indicating whether to save the data or not.
        @param: name: A string. Possible values: baseline, amusement, stress
        @param save_folder: Path to a folder to store the data

        @return: phasic_component, tonic_component, and low pass filtered_phasic_component
        as a numpy array.
    """
    if save:
        if len(name) == 0 or len(save_folder) == 0:
            raise ValueError("Expected name to save the data")
        elif name not in ["baseline", "amusement", 'stress']:
            raise ValueError("Name must be one of these: baseline, amusement, or stress")
    
    phasic_component = []
    tonic_component = []
    phasic_filtered = []
    
    for eda in processed_eda:
        [phasic_gsr, p, tonic_gsr, l, d, e, obj] = cvx.cvxEDA(eda, 1./sample_rate)
        filtered_phasic_gsr = eda_filtering.butter_lowpassfilter(phasic_gsr, 
                                                                 5.0/sample_rate, 
                                                                 sample_rate, order=4)
        phasic_component.append(phasic_gsr)
        tonic_component.append(tonic_gsr)
        phasic_filtered.append(filtered_phasic_gsr)
    
    # get numpy arrays
    phasic_component = np.array(phasic_component)
    tonic_component = np.array(tonic_component)
    phasic_filtered = np.array(phasic_filtered)
    
    # save the data
    utl.save_data(save_folder+"E4_"+name+"_phasic_eda.pickle", phasic_component)
    utl.save_data(save_folder+"E4_"+name+"_tonic_eda.pickle", tonic_component)
    utl.save_data(save_folder+"E4_"+name+"_phasic_filtered_eda.pickle", 
                  phasic_filtered)
    
    return phasic_component, tonic_component, phasic_filtered


def segment_sensor_data(data_path, sample_rate, window_duration, overlap_percent,
                        save=True, name="", save_folder=""):
    """
        @brief: Given data values run sliding window segmentation.
        @param: data_path: A String. Sensor data path 
        @param: sample_rate: sample rate for the sensor in Hz
        @param: window_duration: int, window duration in seconds
        @param: overlap_percentage: (float) Overlap between consequtive window segments
        @param: save: A Boolean indicating whether to save the data or not.
        @param: name: A string. The file name that is used to save the processed window segments. 
            Use some descriptive name such as E4_Baseline_EDA or RespiBan_Stress_Phasic
        @param: save_folder: A string. Path to the folder in which the window segments wiil be 
            stored.
        
        @return: window segments, a numpy array.
    """
    if len(data_path) == 0:
        raise ValueError("Invalid data path. Current path ", data_path)

    try:
        data = utl.read_data(data_path)
    except:
        raise IOError("Cannot load data from path ", data_path)
    
    if save:
        if len(name) == 0 or len(save_folder) == 0:
            raise ValueError("Expected name and save folder path to save the data")
    
    window_segments = np.zeros((1, sample_rate * window_duration))
    
    # get the window segments
    for dt_arr in data:
#         print("Current data array length ", len(dt_arr))
        segments = utl.segment_sensor_reading(dt_arr, window_duration, overlap_percent, 
                                             sample_rate)
        window_segments = np.concatenate([window_segments, segments])
        
    
    # the first entry is all zeros. Discard it.
    window_segments = window_segments[1:, ]
    
    # save the data
    utl.save_data(save_folder+name+"_segments.pickle", window_segments)
    
    return window_segments