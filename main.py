import os
import numpy as np
import matplotlib.pyplot as plt
import gc

import header as h


source_folder = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/" \
                "4_LLR_DATA_SEGMENTATION/NPZ_FILES_BY_TARGET"
storage_name = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/7_LLR_STOPPING_GRAPHS/" \
               "STOPPING_REGIONS"

testing = False
make_graphs = False
# TODO: Play with vel_limit and stop_limit to confirm these are good indicators of a person stopping movement.
#  I think the threshold is a little low right now.
vel_limit = 0.00025  # Chosen to account for minor noise in signal but also to select regions of data in which a user
# may have stopped during a movement.
stop_limit = 10  # The least number of acceptable absolute values below vel_limit in the time series "dd" that
# represents a stop. Used to exclude regions where one slows to immediately start another motion.
dt = 0.01  # The period between each sample obtained from the robot.
dist_limit = 10  # This represents the number of mm in radius around the target in which a pause occurs.

if __name__ == '__main__':
    print("Running frequency analysis...")
    for file in os.listdir(source_folder):
        if file.endswith('.npz'):
            task_number = h.get_task_number(file)
            if (("T1" in file) or ("T2" in file)) & ("V0" in file):
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
                    # Plotting of data here
                    fig = plt.figure(num=1, dpi=100, facecolor='w', edgecolor='w')
                    fig.set_size_inches(25, 8)
                    ax1 = fig.add_subplot(111)
                    ax1.grid(visible=True)
                    ax1.plot(t, v_unfilt*100, label="Unfiltered Absolute Velocity of Robot [mm/s]*100")
                    ax1.plot(t, d/10, label="Distance from Target (DfT) [cm]")
                    ax1.plot(t, dv*10, label="Acceleration of Robot [mm/s]*10")
                    ax1.plot(t, v*100, label="Absolute Velocity of Robot [mm/s]*100")
                    # Identify points of the velocity of distance from target (DfT) below selected threshold.
                    stop_list = h.calculate_stops(v, vel_limit, stop_limit)
                    total_time_stopped = 0
                    num_stops = 0
                    if len(stop_list) > 0:
                        total_time_stopped = h.total_time_stopped(stop_list, dt)  # Returns total amount of time
                        # stopped in seconds.
                        num_stops = h.num_stops(stop_list)
                    # TODO: Generate function to calculate time spent paused within reach of target.
                    reaction_row = h.reaction_time(d)  # Indicates the row in the series that corresponds to the
                    # beginning of the reaction time.
                    # TODO: Design function that, if a reaction time exists, and so does stopped_time, returns a value
                    #  that corresponds to the amount of time stopped before the reaction occurs, and amount of time
                    #  stopped after the reaction occurs. Returns a None value if doesn't occur before or after.
                    if (reaction_row is not None) & (num_stops != 0):
                        stop_before, stop_after = h.stopped_before_after_reaction(stop_list, reaction_row)
                    if len(stop_list) > 0:
                        for j in range(len(stop_list)):
                            t1 = t[stop_list[j][0]]
                            t2 = t[stop_list[j][-1]]
                            ax1.axvspan(t1, t2, color='orange', alpha=0.5)
                    # Set some labels.
                    file_name = file.split('.')[0]
                    ax1.set_xlabel("Time [s]")
                    ax1.set_ylabel("Magnitude")
                    ax1.set_title("Identifying Potential Stopping Points for Sample " + file_name + "_" + str(i) +
                                  " \n Using Absolute Distance from Target, Velocity, and It's Derivatives")
                    plt.legend()
                    save_str = storage_name + '/' + file_name + "_" + str(i)
                    if testing & make_graphs:
                        plt.show()
                        plt.close()
                    elif make_graphs:
                        plt.savefig(fname=save_str)
                        fig.clf()
                        gc.collect()
