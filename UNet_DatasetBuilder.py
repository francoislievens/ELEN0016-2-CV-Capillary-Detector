"""
[ELEN0016-2]
François Lievens
Julien Hubar
Matthias Pirlet
December 2020

This file contain the code that we build to
generate a dataset used during the training
of the UNet
"""
import numpy as np
import cv2
import torch
import os
import sys
import time
import config
from utils import prepare_annotations

DEVICE = 'gpu'

MIN_DRP_WIDTH = 250

def create_UNet_dataset(vid_path, annot_path, start_idx, output_path):

    # Prepare folders
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists('{}/Annotations'.format(output_path)):
        os.mkdir('{}/Annotations'.format(output_path))
    if not os.path.exists('{}/Images'.format(output_path)):
        os.mkdir('{}/Images'.format(output_path))

    # Load the video reader
    vid = cv2.VideoCapture(vid_path)

    # Import annotations and tranform in an array:
        # Indexed by frame
        # For each frame a dict
            # Keys are trakers
            # Values are [[drop coord], nb_cells, [[cells coord]]]
                # Drop coord: [x_min, x_max, x_center, width]
                # Cell coord: [x_center, y_center, width, height]
    annot = prepare_annotations(annot_path)

    # Reading loop
    output_idx = start_idx
    frame_idx = 0
    while True:

        # Get the frame
        ret, frame = vid.read()
        if not ret or frame_idx >= len(annot):
            break

        # To grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get annotations
        data = annot[frame_idx]

        if data is None:    # No droplet in the frame
            frame_idx += 1
            continue

        # Plot all droplets
        for i in range(0, len(data.keys())):

            # Create an image with just the droplet
            kys = list(data.keys())
            drp_coord = data[kys[i]][0]

            # Don't take parts of droplets
            if drp_coord[1] - drp_coord[0] < MIN_DRP_WIDTH:
                break

            sub_frame = frame[0:240, drp_coord[0]:drp_coord[1]]
            sub_frame = cv2.resize(sub_frame, (240, 240), cv2.INTER_AREA)

            # Write the file
            cv2.imwrite('{}/Images/{}.jpg'.format(output_path, output_idx), sub_frame)

            # If cells in it: add coordinates in an annotation file
            if data[kys[i]][1] > 0:
                cells_coord = data[kys[i]][2]
                f = open('{}/Annotations/{}.txt'.format(output_path, output_idx), 'w')
                for k in range(0, len(cells_coord)):
                    # First we need relative coordinates to the cell
                    x_cell = (cells_coord[k][0] - drp_coord[0]) / drp_coord[3]  # divided by the cell width
                    width_cell = cells_coord[k][2] / drp_coord[3]
                    f.write('0 {} {} {} {}\n'.format(x_cell,
                                                     cells_coord[k][1] / 240,
                                                     width_cell,
                                                     cells_coord[k][3] / 240))
                f.close()
            output_idx += 1

        frame_idx += 1

    return output_idx

if __name__ == '__main__':

    # Get the list of all video files:
    file_lst = os.listdir('{}/images'.format(config.DATA_PATH))

    start_idx = 0
    for i in range(0, len(file_lst)):
        # Get paths
        gp_str = file_lst[i][-2:]
        gp = int(gp_str)
        vid_path = '{}/images/{}/group{}.mp4'.format(config.DATA_PATH, file_lst[i], gp)
        print(vid_path)
        annot_path = '{}/annotations/{}/{}.csv'.format(config.DATA_PATH, file_lst[i], file_lst[i])

        # Extract droplets images
        start_idx = create_UNet_dataset(vid_path, annot_path, start_idx, config.UNET_DATA_PATH)

