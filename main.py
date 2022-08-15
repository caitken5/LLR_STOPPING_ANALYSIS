import os
import numpy as np
import matplotlib.pyplot as plt
import gc

import header as h



source_folder = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/4_LLR_DATA_SEGMENTATION/" \
                "NPZ_FILES_BY_TARGET"
storage_name = "D:/PD_Participant_Data/LLR_DATA_ANALYSIS_CLEANED/LLR_DATA_PROCESSING_PIPELINE/7_LLR_STOPPING_GRAPHS"

testing = True
vel_limit = 0.0025  # Chosen to account for minor noise in signal but also to select regions of data in which a user
# may have paused during a movement.

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
                    d = stuff[:, h.data_header.index("Dist_From_Target")]
                    dd = np.gradient(d)
                    ddd = np.gradient(dd)
                    v = stuff[:, h.data_header.index("Vxy_Mag")]
                    vx = stuff[:, h.data_header.index("X_Vel")]
                    vy = stuff[:, h.data_header.index("Y_Vel")]

                    # Plotting of data here
                    fig = plt.figure(num=1, dpi=100, facecolor='w', edgecolor='w')
                    fig.set_size_inches(25, 8)
                    ax1 = fig.add_subplot(111)
                    ax1.grid(visible=True)
                    ax1.plot(t, d/10, label="Distance from Target (DfT) [cm]")
                    ax1.plot(t, dd*10, label="Velocity of DfT [mm/s]*10")
                    ax1.plot(t, ddd*10, label="Acceleration of DfT [mm/s^2]*10")
                    ax1.plot(t, v*100, label="Absolute Velocity of Robot [mm/s]*100")
                    # Identify points of the velocity of distance from target (DfT) below selected threshold.
                    pauses = np.asarray(np.nonzero((dd < vel_limit) & (dd > -vel_limit)))[0]  # This corresponds to the indices of elements where pause
                    # occurs.
                    # Return the elements of pauses where the row elements of pauses is non-sequential.
                    pause_rows = np.diff(pauses)
                    pause_row_ends = np.nonzero(pause_rows > 1)
                    start = 0
                    pause_list = []
                    for j in pause_row_ends:
                        # Using the number of times where the index changes, retrieve the indices in pauses that correlate to indices where a pause occurred. Then, calculate the amount of time spent in each pause state, and how many pauses occurred.
                        val = int(j[0]) + 1
                        temp = pauses[start:val]  # We want to ensure we sample the entire point up to the end of the pause.

                        if temp.shape[0] > 10:
                            temp = temp[:, np.newaxis]
                            pause_list.append(temp)
                        start = val
                    pause_list.append(pauses[start:])

                    for j in range(len(pause_list)):
                        t1 = t[pause_list[j][0]]
                        t2 = t[pause_list[j][-1]]
                        ax1.axvspan(t1, t2, color='orange', alpha=0.5)
                    # Set some labels.
                    file_name = file.split('.')[0]
                    ax1.set_xlabel("Time [s]")
                    ax1.set_ylabel("Magnitude")
                    ax1.set_title("Identifying Potential Stopping Points for Sample " + file_name + "_" + str(i) +
                                  " \n Using Absolute Distance from Target, Velocity, and It's Derivatives")
                    plt.legend()
                    save_str = storage_name + '/' + file_name + "_" + str(i)
                    if testing:
                        plt.show()
                        plt.close()
                    else:
                        plt.savefig(fname=save_str)
                        fig.clf()
                        gc.collect()

