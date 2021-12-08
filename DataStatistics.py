"""
[ELEN0016-2]
François Lievens
Julien Hubar
Matthias Pirlet
December 2020

This code was used to perform a simple statistical
analysis of the input annotated data.
Results from this code was used to help us to determine
some of the thresholds that we used in our final
implementation.
"""
import pandas as pd
from utils import prepare_annotations
import config
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def data_statistics(data_path):

    # Get the list of input videos
    file_lst = os.listdir('{}/images'.format(data_path))

    # Get all annotations files path
    annot_path = []
    for i in range(0, len(file_lst)):
        gp = int(file_lst[i][-2:])
        annot_path.append('{}/annotations/{}/{}.csv'.format(data_path, file_lst[i], file_lst[i]))

    # Array to store stats
    dist_lst = []
    nb_cells_lst = []
    drp_width_lst = []
    nb_frames_lst = []
    tracker_lst = []
    cells_by_tracker = {}


    for z in tqdm(range(0, len(annot_path))):

        ant = prepare_annotations(annot_path[z])
        nb_frames_lst.append(len(ant))

        prev = None  # Precedent frame

        # Read all frame index
        for i in range(0, len(ant)):

            # If no droplet on the frame
            if ant[i] is None:
                continue

            for key in list(ant[i].keys()):
                if not key in tracker_lst:
                    tracker_lst.append(key)
                    cells_by_tracker[key] = []
                # Get the number of cells in the droplet
                nb_cells_lst.append(ant[i][key][1])
                cells_by_tracker[key].append(ant[i][key][1])

                # Now the distance with the previous position of the droplet
                dist = compare_distance(prev, ant[i], key)
                if dist is not None:
                    dist_lst.append(dist)

                # The width of the droplet
                drp_width_lst.append(ant[i][key][0][3])

            prev = ant[i]

    # Get the average and std number of detecte cells by tracker idx
    avg_cell_bytrk = []
    std_cell_bytrk = []
    max_cell_bytrk = []
    for key in cells_by_tracker.keys():
        avg_cell_bytrk.append(np.mean(cells_by_tracker[key]))
        std_cell_bytrk.append(np.std(cells_by_tracker[key]))
        max_cell_bytrk.append(np.max(cells_by_tracker[key]))

    # And standard deviation

    # Plot results
    print('# ================================================================= #')
    print('#                Statistical analysis of the dataset                #')
    print('# ================================================================= #')

    print(' ')
    print('Total number of frames: {}'.format(np.sum(np.asarray(nb_frames_lst))))
    print('Total number of droplets tracker: {}'.format(len(tracker_lst)))


    # Plot histogram of distances run by droplet between two frames
    n, bins, patches = plt.hist(dist_lst, 50, density=True, facecolor='g', alpha=0.75)
    plt.title('Moving distances of droplets between two frames')
    plt.grid(True)
    plt.xlim(0, 700)
    plt.savefig('Figs/stat_movig_dist_histo.png')
    plt.show()

    # Plot histogram of droplet width
    n, bins, patches = plt.hist(drp_width_lst, 50, density=True, facecolor='g', alpha=0.75)
    plt.title('Droplets width histogram')
    plt.grid(True)
    plt.savefig('Figs/stat_drp_width_histo.png')
    plt.show()

    # Plot histogram of the number of cells per droplets
    nb_cell_data = []
    for itm in avg_cell_bytrk:
        nb_cell_data.append(np.round(itm, 0))
    tot_droplets = np.sum(nb_cell_data)

    n, bins, patches = plt.hist(nb_cell_data, density=True, facecolor='g', alpha=0.75)
    plt.title('Distribution of the number of cells by droplets')
    plt.grid(True)
    plt.xlabel('Number of cells')
    plt.savefig('Figs/stat_cell_in_drp_histo.png')
    plt.show()


def compare_distance(prev, actual, track):

    if prev is None:
        return None
    if not str(track) in prev.keys():
        return None

    actual_pos = actual[str(track)][0][2]
    prev_pos = prev[str(track)][0][2]

    return actual_pos - prev_pos




if __name__ == '__main__':

    data_statistics(config.DATA_PATH)