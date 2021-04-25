#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import csv
import pickle


from argparse import ArgumentParser

data_folder = "../Data/Wearable Devices Study Data/"
output_folder = "../Processed Data/24 seconds window ADARP/"

participants_folder = ['Part 101C',
 'Part 102C',
 'Part 104C',
 'Part 105C',
 'Part 106C',
 'Part 107C',
 'Part 108C',
 'Part 109C',
 'Part 110C',
 'Part 111C',
 'Part 112C']


# # Details
# The sampling rate for
# - Galvanic Skin Response is 4hz
# - Skin Temperature is 4hz
# - Blood Volume Pulse is 64hz
# - Acceleration is 32hz
# - Heart rate is 1hz
#
# For each sensor csv file, the first row is the timestamp for recording start time
# and in some sensor files the second row is sampling frequency. We can use the
# timestamp to put time values next to each row of values in all sensor files,
# and use this timestamp to extract the window around the tag timestamps.
#
# For window size we can experiment with different values, and we will start
# with 25 seconds window and go upto 10 minutes.
#
# We also need to apply filtering on sensor values. For EDA values,
# 1. First-order BW LPF cut-off frequency 5 Hz to remove noise.
# 2. First-order BW HPF cut-off frequency 0.05 Hz to separate SCR and SCL
#
# and for skin temperature
# 1. Second-order BW LPF frequency of 1 Hz
# 2. Second-order BW HPF frequency of 0.1 Hz
#
#
# Unix time is a system for describing a point in time, and is the number of
# seconds that have elapsed since the Unix epoch, minus leap seconds; the Unix
# epoch is 00:00:00 UTC on 1 January 1970.
#
# Every file except the IBI file has sampling frequency in the second row.
# All files have staring time in UNIX timestamp in the first row.

# Constants
E4_EDA_SF = 4
E4_ACC_SF = 32
E4_BVP_SF = 64
E4_HR_SF = 1
E4_TEMP_SF = 4

def save_data(path, data):
    """
    Given a path and data value, write the data value to the path as a pickle file.
    :param path: file path with .pkl extension
    :param data: data values
    """
    f = open(path, "wb")
    pickle.dump(data, f)
    f.close()

def read_data(path):
    """
    Given a path, read the file and return the contents.
    :param path: File path with .pkl extension
    """
    f = open(path, "rb")
    d = pickle.load(f)
    return d

def get_sensor_data(file_path):
    """
    Load data from a text file located at file_path.
    :param file_path: path to the text file

    """
    data = []
    try:
        data = np.genfromtxt(file_path, delimiter=',')
    except:
        print("Error reading the file {}".format(file_path))

    return data


def get_tag_timestamps(file_directory):
    """
    Given the data directory, open the tags file and retun the tag timestamps as an array.
    :param file_directory: Path to the folder containing the tags file.

    """
    file_path = file_directory + "/tags.csv"
    tag_timestamps = []

    try:
        with open(file_path, "r") as read_file:
            csv_reader = csv.reader(read_file)
            for row in csv_reader:
                unix_time = float(row[0])
                tag_timestamps.append(unix_time)
    except:
        print("No Tags file in " + file_directory)

    return tag_timestamps


def extract_segments_around_tags(data, tags, window_size):
    """
    Ginen data array, tags array and window size extract window size segments from the data array abound the tags.

    :param data: Data array
    :param tags: An array with tag event times
    :param window size: Window size in seconds

    """
    # return array
    segments = []

    # get the start time, sampling frequency and the sensor data
    start_time = data[0]
    sampling_freq = data[1]

    try:
        if len(start_time):
            start_time = start_time[0]
    except:
        start_time = start_time

    try:
        if len(sampling_freq):
            sampling_freq = sampling_freq[0]
    except:
        sampling_freq = sampling_freq

    sensor_data = data[2:]
    data_length = len(sensor_data)

    # now create the timestamps for each row of data
    sensor_time = [start_time]

    # Get the time for each obsevation in the data array
    delta_time = 1 / sampling_freq
#     print("For sampling freq {}, delta time: {}".format(sampling_freq, delta_time))

    for d in sensor_data:
        start_time = start_time + delta_time
        sensor_time.append(start_time)

    sensor_time_length = len(sensor_time)
    start_time = sensor_time[0]
    end_time = sensor_time[-1]

    # for each time stamp in tags
    for timestamp in tags:
        # if the timestamp is within the sensor time array
        if (timestamp > start_time) & (timestamp < end_time):
            # how far is the timestamp from the start time.
            difference = int(timestamp - start_time)

            # get the index in the sensor data array, based on the difference of tag timestamp
            position = int(difference * sampling_freq)

            # if the position is within the sensor time
            if position < data_length:
                # divide the window size by 2, equal length before and after the timestamp
                n_observation = int((window_size / 2) * sampling_freq)

                # window segment position in the data array
                from_ = position - n_observation
                to_ = position + n_observation

                # get the window segment
                if (to_ < data_length) & (from_ > 0):
                    segments.append(sensor_data[from_:to_])

    return segments


def get_eda_data_around_tags(data_folder, tag_timestamps, window_size):
    """
    Get EDA segments from the EDA CSV file in data_folder with tag_timestamps for window length of window_size

    :param data_folder: Path to the folder containing the EDA file
    :param tag_timestamps: An array containing the tag event markers.
    :param window_size: Window size in seconds.

    """

    # load the data from EDA.csv
    file_path = data_folder + "/EDA.csv"
    sensor_data = get_sensor_data(file_path)

    # remove noise
#     sensor_data = butter_lowpass_filter_eda(sensor_data)

    # get the SCR
#     sensor_data = butter_highpass_filter_eda(sensor_data)

    return extract_segments_around_tags(sensor_data, tag_timestamps, window_size)


def get_hr_data_around_tags(data_folder, tag_timestamps, window_size):
    """
    Get HR segments from the HR CSV file in data_folder with tag_timestamps for window length of window_size

    :param data_folder: Path to the folder containing the HR file
    :param tag_timestamps: An array containing the tag event markers.
    :param window_size: Window size in seconds.

    """

    # load the data from EDA.csv
    file_path = data_folder + "/HR.csv"
    sensor_data = get_sensor_data(file_path)

    return extract_segments_around_tags(sensor_data, tag_timestamps, window_size)


def get_temp_data_around_tags(data_folder, tag_timestamps, window_size):
    """
    Get TEMP segments from the TEMP CSV file in data_folder with tag_timestamps for window length of window_size

    :param data_folder: Path to the folder containing the TEMP file
    :param tag_timestamps: An array containing the tag event markers.
    :param window_size: Window size in seconds.

    """

    # load the data from EDA.csv
    file_path = data_folder + "/TEMP.csv"
    sensor_data = get_sensor_data(file_path)

    return extract_segments_around_tags(sensor_data, tag_timestamps, window_size)


def get_bvp_data_around_tags(data_folder, tag_timestamps, window_size):
    """
    Get BVP segments from the BVP CSV file in data_folder with tag_timestamps for window length of window_size

    :param data_folder: Path to the folder containing the BVP file
    :param tag_timestamps: An array containing the tag event markers.
    :param window_size: Window size in seconds.

    """

    # load the data from EDA.csv
    file_path = data_folder + "/BVP.csv"
    sensor_data = get_sensor_data(file_path)

    return extract_segments_around_tags(sensor_data, tag_timestamps, window_size)


def get_acc_data_around_tags(data_folder, tag_timestamps, window_size):
    """
    Get ACC segments from the ACC CSV file in data_folder with tag_timestamps for window length of window_size

    :param data_folder: Path to the folder containing the ACC file
    :param tag_timestamps: An array containing the tag event markers.
    :param window_size: Window size in seconds.

    """
    # load the data from EDA.csv
    file_path = data_folder + "/ACC.csv"
    sensor_data = get_sensor_data(file_path)

    return extract_segments_around_tags(sensor_data, tag_timestamps, window_size)


def extract_data_around_tags(window_length_seconds):
    """
    From the ADARP data folder extract the sensor segments for length of window_length_seconds. EDA, HR, ACC, BVP, and TEMP segemnts
    are extracted around tag event markers.

    :param window_length_seconds: Size of the window in seconds.

    """
    eda_data = []
    hr_data = []
    acc_data = []
    bvp_data = []
    temp_data= []

    # for each participants
    for p in participants_folder:
        part_eda_data = []
        part_hr_data = []
        part_acc_data = []
        part_bvp_data = []
        part_temp_data = []

#         print("Extracting data for participants: {}".format(p))
        participants_folder_path = data_folder + p + "/"
        subfolders = os.listdir(participants_folder_path)

        # for each sub-folder in the participants folder
        for sub in subfolders:
            path = participants_folder_path + sub
#             print("For subfolder: {}".format(path))

            # get the tag events in this folder
            tag_timestamps = get_tag_timestamps(path)

            # if there are tag events, get the sensor values
            if len(tag_timestamps):
                # first EDA
                values = get_eda_data_around_tags(path, tag_timestamps, window_length_seconds)
                if len(values):
                    eda_data.extend(values)
                    part_eda_data.extend(values)

                # second temperature
                values = get_temp_data_around_tags(path, tag_timestamps, window_length_seconds)
                if len(values):
                    temp_data.extend(values)
                    part_temp_data.extend(values)

                # third bvp
                values = get_bvp_data_around_tags(path, tag_timestamps, window_length_seconds)
                if len(values):
                    bvp_data.extend(values)
                    part_bvp_data.extend(values)

                # fourth hr
                values = get_hr_data_around_tags(path, tag_timestamps, window_length_seconds)
                if len(values):
                    hr_data.extend(values)
                    part_hr_data.extend(values)

                # fifth acc
                values = get_acc_data_around_tags(path, tag_timestamps, window_length_seconds)
                if len(values):
                    acc_data.extend(values)
                    part_acc_data.extend(values)

        # save the participants data
        print("Saving data of participants " + p)
        save_data(output_folder + p + "_EDA_TAG.pkl", np.array(part_eda_data))
        save_data(output_folder + p + "_TEMP_TAG.pkl", np.array(part_temp_data))
        save_data(output_folder + p + "_HR_TAG.pkl", np.array(part_hr_data))
        save_data(output_folder + p + "_BVP_TAG.pkl", np.array(part_bvp_data))
        save_data(output_folder + p + "_ACC_TAG.pkl", np.array(part_acc_data))

    return np.array(eda_data), np.array(hr_data), np.array(acc_data), np.array(bvp_data), np.array(temp_data)

def get_window_segment(sensor_data, window_length, sampling_freq):
    """
    Segments the data in sensor_data array sampled at frequency of sampling_freq for window size of window_length.

    :param sensor_data: Data array
    :param window_length: Window size in seconds
    :param sampling_freq: Sampling frequency for the given data array.

    """
    segments = []

    # get the segment length
    segment_length = window_length * sampling_freq
    sensor_data_length = len(sensor_data)

    number_of_segments = sensor_data_length // segment_length
#     print(sensor_data_length, number_of_segments)
    from_ = 0
    to_ = segment_length
    for i in range(number_of_segments):
        seg = sensor_data[from_:to_]
        segments.append(seg)
        from_ = to_
        to_ = to_ + segment_length

    return segments


def extract_data_without_tags(window_length_seconds):

    """
    Extract sensor segments from the ADARP data folder for observation folder that has empty tags file. EDA, HR, ACC, BVP, and TEMP
    segments are extracted from observation folder that has empty tags file.

    :param window_length_seconds: Window size in seconds

    """
    eda_data = []
    hr_data = []
    acc_data = []
    bvp_data = []
    temp_data= []

    # for each participants
    for p in participants_folder:
        part_eda_data = []
        part_hr_data = []
        part_acc_data = []
        part_bvp_data = []
        part_temp_data = []

#         print("Extracting data for participants {}".format(p))
        participants_folder_path = data_folder + p + "/"
        subfolders = os.listdir(participants_folder_path)

        # for each subfolders in the participant folder
        for sub in subfolders:
            path = participants_folder_path + sub

            # get tag timestamps
            tag_timestamps = get_tag_timestamps(path)

            # if there are no tags, get sensor values
            if len(tag_timestamps) == 0:
                # EDA Segments
                data = get_sensor_data(path+"/EDA.csv")
                # first entry is the start time and the second row is the sampling frequency
                segments = get_window_segment(data[2:], window_length_seconds, E4_EDA_SF)
                if len(segments):
                    eda_data.extend(segments)
                    part_eda_data.extend(segments)

                # HR Segments
                data = get_sensor_data(path+"/HR.csv")
                segments = get_window_segment(data[2:], window_length_seconds, E4_HR_SF)
                if len(segments):
                    hr_data.extend(segments)
                    part_hr_data.extend(segments)

                # TEMP Segments
                data = get_sensor_data(path+"/TEMP.csv")
                segments = get_window_segment(data[2:], window_length_seconds, E4_TEMP_SF)
                if len(segments):
                    temp_data.extend(segments)
                    part_temp_data.extend(segments)

                # BVP Segments
                data = get_sensor_data(path+"/BVP.csv")
                segments = get_window_segment(data[2:], window_length_seconds, E4_BVP_SF)
                if len(segments):
                    bvp_data.extend(segments)
                    part_bvp_data.extend(segments)

                # ACC Segments
                data = get_sensor_data(path+"/ACC.csv")
                segments = get_window_segment(data[2:], window_length_seconds, E4_ACC_SF)
                if len(segments):
                    acc_data.extend(segments)
                    part_acc_data.extend(segments)

        # save the participants data
        print("Saving data of participants " + p)
        save_data(output_folder + p + "_EDA_NO_TAG.pkl", np.array(part_eda_data))
        save_data(output_folder + p + "_TEMP_NO_TAG.pkl", np.array(part_temp_data))
        save_data(output_folder + p + "_HR_NO_TAG.pkl", np.array(part_hr_data))
        save_data(output_folder + p + "_BVP_NO_TAG.pkl", np.array(part_bvp_data))
        save_data(output_folder + p + "_ACC_NO_TAG.pkl", np.array(part_acc_data))

    return np.array(eda_data), np.array(hr_data), np.array(temp_data), np.array(bvp_data), np.array(acc_data)


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


if __name__ == '__main__':
    parser = ArgumentParser("Data processing ADARP")
    parser.add_argument(
        "-i",
        "--input_directory",
        type=str,
        required=True,
        help="Directory that contains the subject data for ADARP"
    )

    parser.add_argument(
        "-o",
        "--output_directory",
        type=str,
        required=True,
        help="Directory to store the processed data"
    )

    parser.add_argument(
        "-w",
        "--window_size",
        type=int,
        required=True,
        help="window size in seconds"
    )

    args = parser.parse_args()
    # args.input_directory
    # args.output_directory
    # args.window_size