import os
import numpy as np
import matplotlib.pyplot as plt
import gc

import header as h


source_folder = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/" \
                "4_LLR_DATA_SEGMENTATION/NPZ_FILES_BY_TARGET"
storage_name = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/7_LLR_STOPPING_GRAPHS/" \
               "STOPPING_REGIONS_BLACK_AND_WHITE"

testing = False
make_graphs = True
vel_limit = 0.0002  # Chosen to account for minor noise in signal but also to select regions of data in which a user
# may have stopped during a movement.
# Pretty sure this vel_limit might be okay though.
stop_limit = 20  # The least number of acceptable absolute values below vel_limit in the time series "dd" that
# represents a stop. Used to exclude regions where one slows to immediately start another motion.
dt = 0.01  # The period between each sample obtained from the robot.
dist_limit = 10  # This represents the number of mm in radius around the target in which a pause occurs.

# Create some variables for controlling the font size.
SMALLER_SIZE = 18
SMALL_SIZE = 24

# Change the sizes for specific features of plots accordingly.
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)


if __name__ == '__main__':
    print("Running stopping analysis...")
    for file in os.listdir(source_folder):
        if file.endswith('.npz'):
            task_number = h.get_task_number(file)
            if (("T1" in file) or ("T2" in file)) & ("V0" in file) & ("SCS_33_V0_T1_U_L" in file):
                # Here if file includes task 1 and 2, and if it also is a V0 file, as I'm not studying other pieces
                # of information.
                print(file)
                source_file = source_folder + '/' + file
                # The source file is a npz file. So I need to load in the data and unpack.
                data = np.load(source_file, allow_pickle=True)
                # Unpack the data.
                ragged_list, target_list, target_i = h.load_npz(data)
                # Retrieving of data here.
                for i in range(len(ragged_list)):
                    stuff = ragged_list[i]
                    t = stuff[:, h.data_header.index("Time")]
                    v_unfilt = stuff[:, h.data_header.index("Vxy_Mag")]
                    v = h.butterworth_filter(v_unfilt)
                    dv = np.gradient(v)
                    d = stuff[:, h.data_header.index("Dist_From_Target")]
                    f = stuff[:, h.data_header.index("Fxy_Mag")]
                    # Plotting of data here
                    fig = plt.figure(num=1, dpi=100, facecolor='w', edgecolor='w')
                    fig.set_size_inches(15, 8)
                    ax1 = fig.add_subplot(111)
                    ax1.grid(visible=True)
                    ax1.plot(t, f, 'k--', label="Force (N)")
                    ax1.plot(t, d/10, 'k-', label="Distance from Target (cm)")
                    # ax1.plot(t, dv*10, label="Acceleration of Robot / 10 (cm/$s^2$)")
                    ax1.plot(t, v*100, 'k:', label="Absolute Velocity (cm/s)")
                    # Identify points of the velocity of distance from target (DfT) below selected threshold.
                    stop_list = h.calculate_stops(v, vel_limit, stop_limit)
                    total_time_stopped = 0
                    num_stops = 0
                    if len(stop_list) > 0:
                        total_time_stopped = h.time_stopped(stop_list, dt)  # Returns total amount of time
                        # stopped in seconds.
                        num_stops = h.num_stops(stop_list)
                    reaction_row = h.reaction_time(d)  # Indicates the row in the series that corresponds to the
                    # beginning of the reaction time.
                    stop_before = []
                    stop_after = []  # Need to ensure these elements are empty arrays so that if they are referenced
                    # later I don't run into errors where the don't exist.
                    stop_time_before = None
                    stop_time_after = None
                    avg_dist_after = None
                    if (reaction_row is not None) & (num_stops != 0):
                        stop_before, stop_after = h.stopped_before_after_reaction(stop_list, reaction_row)
                        if len(stop_before) > 0:
                            stop_time_before = h.time_stopped(stop_before, dt)
                        if len(stop_after) > 0:
                            stop_time_after = h.time_stopped(stop_after, dt)
                            avg_dist_after = h.avg_dist_stopped(stop_after, d)
                    # Here is the code for calculating if a person stopped within the selected target range,
                    # how much time was spent there, and how many stops occurred within that region.
                    time_stopped_within = None
                    times_stopped_within = None
                    if num_stops != 0:
                        stopped_within = h.stopped_within_target(stop_list, d, dist_limit)
                        time_stopped_within = h.time_stopped(stopped_within, dt)
                        times_stopped_within = h.num_stops(stopped_within)
                    if len(stop_list) > 0:
                        for j in range(len(stop_list)):
                            t1 = t[stop_list[j][0]]
                            t2 = t[stop_list[j][-1]]
                            if j == 1:
                                ax1.axvspan(t1, t2, color='grey', alpha=0.5, label="Stopping Region")
                            else:
                                ax1.axvspan(t1, t2, color='grey', alpha=0.5)
                    # Set some labels.
                    file_name = file.split('.')[0]
                    ax1.set_xlabel("Time (s)")
                    ax1.set_ylabel("Magnitude")
                    # ax1.set_title("Identifying Potential Stopping Points for Sample " + file_name + "_" + str(i) +
                    #              " \n Using Absolute Distance from Target, Velocity, and It's Derivatives")
                    plt.legend(loc="upper right")
                    save_str = storage_name + '/' + file_name + "_" + str(i)
                    if testing & make_graphs:
                        plt.show()
                        plt.close()
                    elif make_graphs:
                        plt.savefig(fname=save_str)
                        fig.clf()
                        gc.collect()
