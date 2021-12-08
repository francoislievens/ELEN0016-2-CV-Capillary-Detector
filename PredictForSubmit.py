"""
[ELEN0016-2]
François Lievens
Julien Hubar
Matthias Pirlet
December 2020

This file implement a more simple version of our algorithm
(no multi-thread) that we used during the challenges.
The particularity of this implementation is that we detect
all droplets for each frames and all are feed to our UNet.
By this way we can output all detected elements of all frames
of interest.

This implementation don't perform the counting of detected
objects.
"""
import cv2
import numpy as np
import torch
from UNet import UNet
from utils import get_droplet_coordinates
from utils import count_peaks_2d
import sys

# Change the following path to the target video
INPUT_PATH = 'Original_Dataset/images/CV2021_GROUP02/group2.mp4'
UNET_PATH = 'Model/UNet_A/model_weights.pt'
OUTPUT_PATH = 'Output_Results'

DEVICE = 'cuda'
SHOW = True
SAVE = True

if __name__ == '__main__':

    # Hyper parameters
    col_sum_out_thresh = 1000  # Minimal threshold to enter a droplet
    col_sum_in_thresh = 300  # Minimal threshold to stay in a droplet
    dist_thresh = 750  # Maximal distance that a droplet can do between two frames
    droplet_min_width = 250  # Width threshold to only consider full droplets
    offset = 10

    # Prepare output files
    if SAVE:
        f = open('{}/results_frame_interest.csv'.format(OUTPUT_PATH), 'w')
        f.close()


    # Instanciate the UNet
    unet = UNet(input_filters=1).to(DEVICE)
    try:
        unet.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
    except:
        print('ERROR: Impossible to load the state dict')
        sys.exit(-1)

    vid = cv2.VideoCapture(INPUT_PATH)
    cuda_stream = None
    MOG = cv2.cuda.createBackgroundSubtractorMOG(history=50, nmixtures=2, backgroundRatio=0.02)


    ret, frame = vid.read()

    idx = 0
    while ret:
        #if idx % 50 == 0:

        print('Frame: {}'.format(idx))
        if SHOW:
            show_frame = np.copy(frame)
        frame = cv2.cuda_GpuMat(frame)
        frame = cv2.cuda.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask = MOG.apply(frame, 0.1, cuda_stream)
        mask = mask.download()

        col_sum = np.sum(mask, axis=0)

        # Get droplet coordinates
        drop_coord = get_droplet_coordinates(col_sum,
                                             col_sum_out_thresh,
                                             col_sum_in_thresh,
                                             droplet_min_width)

        # Get inputs for the UNet
        un_frames = []
        if SAVE:
            f = open('{}/results_frame_interest.csv'.format(OUTPUT_PATH), 'a')
        for i in range(len(drop_coord)):
            x_start = int(drop_coord[i][0])
            x_end = int(drop_coord[i][1])
            if SAVE:
                f.write('frame_{},{},{},{},{},droplet\n'.format(idx,
                                                                x_start, 0, x_end, 240))
            if SHOW:
                show_frame = cv2.rectangle(show_frame,
                                           (x_start, 0),
                                           (x_end, 240),
                                           (240, 240, 240), 3)
            sub_frame = frame.colRange(x_start, x_end)
            sub_frame = cv2.cuda.resize(sub_frame, (240, 240))
            sub_frame = sub_frame.download()
            un_frames.append(((x_start, x_end), sub_frame))
        if SAVE:
            f.close()


        # Predict cells
        cell_coord = []
        for itm in un_frames:
            drp_coord, drp_img = itm
            drp_tensor = torch.tensor(drp_img).reshape((-1, 240, 240)).to(DEVICE)
            unet.eval()
            with torch.no_grad():
                preds = unet(drp_tensor)
            preds = preds.reshape(240, 240)
            preds = preds.cpu().numpy()

            # detect peaks in the mask
            cnt, positions = count_peaks_2d(preds)

            for pos in positions:
                # Get coordinates relative to the whole frame
                pos[0] = int((pos[0] / 240) * (drp_coord[1] - drp_coord[0]))
                pos[0] = pos[0] + drp_coord[0]
                pos[1] = int(pos[1])

                if SAVE:
                    f = open('{}/results_frame_interest.csv'.format(OUTPUT_PATH), 'a')
                    f.write('frame_{},{},{},{},{},cell\n'.format(idx,
                                                                 pos[0] - 10,
                                                                 pos[1] - 10,
                                                                 pos[0] + 10,
                                                                 pos[1] + 10))
                    f.close()
                if SHOW:
                    show_frame = cv2.rectangle(show_frame,
                                               (pos[0] - 10, pos[1] - 10),
                                               (pos[0] + 10, pos[1] + 10),
                                               (0, 0, 0), 3)
        if SHOW:
            cv2.imshow('output', show_frame)
            cv2.waitKey()

        ret, frame = vid.read()
        idx += 1