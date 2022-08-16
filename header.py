import sys
import numpy as np
from scipy import signal

data_header = ['Time', 'Des_X_Pos', 'Des_Y_Pos', 'X_Pos', 'Y_Pos', 'OptoForce_X', 'OptoForce_Y', 'OptoForce_Z',
               'OptoForce_Z_Torque', 'Theta_1', 'Theta_2', 'Fxy_Mag', 'Fxy_Angle', 'CorrForce_X', 'Corr_Force_Y',
               'Target_Num', 'X_Vel', 'Y_Vel', 'Vxy_Mag', 'Vxy_Angle', 'Dist_From_Target', 'Disp_Btw_Pts', 'Est_Vel',
               'To_From_Home', 'Num_Prev_Targets', 'Resistance']


def load_npz(npz_file):
    # This function loads npz file, and reconstructs a ragged list of numpy arrays given data and counter.
    # Returns ragged_list, data, and counter.
    # Assume that each of the npz files contains two pieces of information: the data to be unpacked and the unpacking indices.
    my_list = npz_file.files
    data = npz_file[my_list[0]]
    counter = npz_file[my_list[1]]
    sep_row = np.cumsum(counter)
    # Define an object that the arrays can be loaded into, noting that they are ragged because the number of rows may differ each time.
    ragged_list = [data[0:sep_row[0], :]]  # Append the first set of data.
    for j in range(len(counter)-1):
        ragged_list.append(data[sep_row[j]:sep_row[j+1], :])
        # Since the list values for each array is calculated, the last row in the array is included in the calculation
        # and a separate line to append the last piece of data from a start to end point is not needed.
    return ragged_list, data, counter


def get_task_number(file_name):
    if "T1" in file_name:
        task = 1
    elif "T2" in file_name:
        task = 2
    elif "T3" in file_name:
        task = 3
    elif "T4" in file_name:
        task = 4
    else:
        my_str = "Task name wasn't found, file_name: " + file_name + ", exiting program."
        sys.exit(my_str)
    return task


def delete_multiple_element(list_object, indices):
    # This code was obtained from https://thispointer.com/python-remove-elements-from-list-by-index/
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)


def butterworth_filter(data):
    # 2nd order 50 Hz Butterworth filter applied along the time series dimension of as many columns are passed to it.
    # Will likely just be used to filter the velocity data, since a second order 20Hz Butterworth filter has already
    # been applied to the force data.
    sos = signal.butter(2, 50, 'lowpass', fs=1000, output='sos')
    # Frequency was selected to smooth the signal so there are no accidental spikes in data when a person has actually
    # stopped, but not so that it changes the shape of the signal, which if too low makes it difficult to
    # identify stopping regions.
    # Since fs is specified, set the cutoff filter to 20 Hz.
    filtered_data = signal.sosfiltfilt(sos, data, axis=0)
    return filtered_data
