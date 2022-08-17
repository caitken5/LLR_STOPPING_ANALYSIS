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


def calculate_stops(v, vel_limit, stop_limit):
    # In this function, I want to return the list of arrays that contain the indices of when the user is below the
    # selected velocity limit and is therefore considered stopped.
    temp = []
    stop_list = []
    stops = np.asarray(np.nonzero((v < vel_limit) & (v > -vel_limit)))[0]  # This corresponds to the
    # indices of elements where a stop occurs.
    if stops.shape[0] != 0:
        for j, k in enumerate(stops):  # Append all the values as new arrays to the stop_list.
            # Does not yet deal if arrays are below a certain size.
            temp.append(k)
            if k == stops[-1]:
                arr = np.asarray(temp)
                stop_list.append(arr)
            elif (stops[j + 1] - k) > 1:
                arr = np.asarray(temp)
                stop_list.append(arr)
                temp = []
        # As long as stops has some length, the above code will result in the stop_list containing some information.
        # Now remove arrays from stop_list that are not greater than number of samples specified in stop_limit.
        remove_list = []  # Used to append values to then remove values from stop_list all at once.
        for j in range(len(stop_list)):
            len_stop = stop_list[j].shape[0]
            if len_stop < stop_limit:
                # Add element to remove_list.
                remove_list.append(j)
        if len(remove_list) > 0:
            # If remove_list is empty, this below function would probably remove the final element, or some kind of error.
            delete_multiple_element(stop_list, remove_list)
    return stop_list


def time_stopped(stop_list, dt):
    # This function calculates the total amount of time spent stopped during the reaching motion.
    total_time = 0
    for i in range(len(stop_list)):
        start_row = stop_list[i][0]
        end_row = stop_list[i][-1]
        num_rows = end_row - start_row
        total_time += num_rows*dt
    return total_time


def reaction_time(dist_from_target):
    # This function calculates the reaction time of the user in this particular reach. It returns the row number that
    # corresponds to when the reaction occurs.
    react_time = None
    for j in range(10, dist_from_target.shape[0]):
        row_start = dist_from_target[j] - dist_from_target[j-10]
        if row_start <= -1:
            react_time = j
            break
    return react_time


def num_stops(stop_list):
    # This function calculates the number of times that the user stopped on individual occasions.
    return len(stop_list)


def avg_dist_stopped(stop_list, dist_from_target):
    # This function calculates the average distance after the reaction has occurred at which a user is stopped.
    my_list = []
    for i in range(len(stop_list)):
        for j in range(stop_list[i].shape[0]):
            my_list.append(stop_list[i][j])
    # Use the total list of indices after reaction time to get values from dist_from_target.
    avg_dist_after_list = dist_from_target[my_list]
    avg_dist_after = np.mean(avg_dist_after_list)
    return avg_dist_after


def stopped_before_after_reaction(stop_list, reaction_row):
    # Function that returns the lists of arrays that correspond to time stopped before reaction time, and time
    # stopped after reaction time.
    # Code assumes stop_list is not empty, and therefore, if the last value of array in stop_list is not before
    # reaction_row, then it must occur after.
    stop_before = []
    stop_after = []
    for i in range(len(stop_list)):
        if stop_list[i][-1] < reaction_row:
            stop_before.append(stop_list[i])
        else:
            stop_after.append(stop_list[i])
    return stop_before, stop_after


def stopped_within_target(stop_list, dist_from_target, dist_limit):
    # This function returns the array of samples of distances if they occur within a desigated stoppe distance.
    stopped_within = []
    for i in range(len(stop_list)):
        avg_dist = np.mean(dist_from_target[stop_list[i]])
        if avg_dist < dist_limit:
            stopped_within.append(stop_list[i])
    return stopped_within
