import time
import cv2
from threading import Thread
from queue import Queue
from utils import get_droplet_coordinates
from utils import check_new_droplets
from utils import check_new_droplets_no_prev
from utils import plot_droplet_detection
import numpy as np


OUTPUT_DIR = 'Intermediate_Results/Droplet_Detector'


class DropletDetector():
    """
    In this thread, we compute the position of each droplet,
    we count each droplet who pass through the image by checking
    if not already pass and we fill a buffer who contain droplets
    images to count the number of cells
    """

    def __init__(self,
                 input_buffer,
                 buff_size=500,
                 device='cuda',
                 avg_kernel_size=40,
                 debug=False):

        # Hyper parameters:
        #self.col_sum_thresh = 1000  # Droplet detection if sum of mask pxl of the column above 1000
        self.col_sum_out_thresh = 1000  # Minimal threshold to enter a droplet
        self.col_sum_in_thresh = 300  # Minimal threshold to stay in a droplet
        self.dist_thresh = 750  # Maximal distance that a droplet can do between two frames
        self.droplet_min_width = 250  # Width threshold to only consider full droplets
        self.show_graph = debug  # To show the graph of the column sum vector
        # Store the input buffer
        self.input = input_buffer
        # A kernel for the average convolution
        self.avg_ker = np.ones(avg_kernel_size) / avg_kernel_size

        # The output buffer
        self.buffer = Queue(maxsize=buff_size)
        self.device = device
        self.cuda = False
        if self.device == 'cuda':
            self.cuda = True

        # Thread management
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.end = False
        self.pred_end = False

        # Store last frame droplets
        self.last_frame_drop = []
        # An array who store droplets find at previous iter (to count cells)
        self.last_frame_detect = []
        # Count new droplets
        self.drop_counter = 0
        self.drop_prev_counter = 0      # To keep the index of previously detected droplet for indexing UNet inputs

        # Store final results
        self.final_results = []

    def start(self):

        self.thread.start()
        return self

    def update(self):

        while self.input.more():

            # Perform iterations to fill the buffer
            if not self.buffer.full():

                # =============================================================== #
                #                  Droplet Detection and counting                 #
                # =============================================================== #

                # Get a frame
                frame, mask, frame_idx = self.input.read()
                # Download on cpu
                if self.cuda:
                    mask = mask.download()

                # Makes the sum of elements in a column of the mask
                col_sum = np.sum(mask, axis=0)
                # Apply an average convolution (too computational expensive
                # col_sum = np.convolve(col_sum, self.avg_ker, mode='valid')

                # Use this method from utils with njit to detect cells coordinates
                drop_coord = get_droplet_coordinates(col_sum,
                                                     self.col_sum_out_thresh,
                                                     self.col_sum_in_thresh,
                                                     self.droplet_min_width)
                # returned coord: (x_start, x_end, center_x, width)

                if drop_coord.shape[0] == 0:
                    find_idx = []
                    prev_find = []
                else:
                    # Check if no new droplets
                    if len(self.last_frame_drop) > 0:  # Avoid empty array for njit
                        drp_counter, find_idx, prev_find = check_new_droplets(drop_coord,
                                                                              self.last_frame_drop,
                                                                              self.last_frame_detect,
                                                                              self.dist_thresh,
                                                                              self.drop_counter)
                    else:
                        drp_counter, find_idx, prev_find = check_new_droplets_no_prev(drop_coord,
                                                                                      self.drop_counter)
                    self.drop_prev_counter = self.drop_counter
                    self.drop_counter = drp_counter

                # =============================================================== #
                #               Extracting a view of each droplet                 #
                #               to feed into the UNet and counting                #
                #                       and counting cells                        #
                # =============================================================== #

                # Do this only if previously detected droplet:
                crop_cell = []
                if True in prev_find:

                    # For each droplet who appear for the second time, extract the droplet image
                    for i in range(0, len(prev_find)):
                        if prev_find[i]:
                            x_start = int(drop_coord[i][0])
                            x_end = int(drop_coord[i][1])

                            # Crop the frame and resize it for the UNet
                            if self.cuda:
                                sub_frame = frame.colRange(x_start, x_end)
                                sub_frame = cv2.cuda.resize(sub_frame, (240, 240))
                                sub_frame = sub_frame.download()
                            else:
                                sub_frame = frame[:, x_start:x_end]
                                sub_frame = cv2.resize(sub_frame, (240, 240), interpolation=cv2.INTER_AREA)

                            # Put the cropped cell into an array with the index of the detected droplet
                            crop_cell.append((sub_frame, self.drop_prev_counter - len(crop_cell)))

                            # Append the sub frame and corresponding index in the buffer
                            self.buffer.put((
                                sub_frame,
                                self.drop_prev_counter - len(crop_cell),
                                frame_idx,
                                x_end - x_start
                            ))
                            self.final_results.append((frame_idx, x_start, x_end))




                # If we want to show the graph of the vector values
                if self.show_graph:
                    plot_droplet_detection(frame=frame,
                                           mask=mask,
                                           col_sum=col_sum,
                                           frame_idx=frame_idx,
                                           drop_coord=drop_coord,
                                           find_idx=find_idx,
                                           path=OUTPUT_DIR,
                                           device=self.device)

                # Update last droplet coord
                self.last_frame_drop = drop_coord
                self.last_frame_detect = find_idx

            else:
                # If the buffer is already full, wait 0.1 sec
                time.sleep(0.01)

        # When the previous algorithm end
        self.end = True

    def read(self):
        """
        :return: The oldest image in the buffer
        """
        return self.buffer.get()


    def more(self):
        """
        If the buffer is empty but not already set as
        end of video, we continue 20 times to wait a
        potential refilling
        :return: True if there are elements after waiting
        """

        while self.buffer.qsize() == 0 and not self.end:
            time.sleep(0.01)

        buff_fill = False
        if self.buffer.qsize() > 0:
            buff_fill = True

        return buff_fill

    def running(self):

        if self.end:
            return False
        else:
            return self.more()

    def stop(self):

        self.end = True
        self.thread.join()
