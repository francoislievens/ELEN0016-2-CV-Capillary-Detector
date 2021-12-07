import time
import cv2
from threading import Thread
from queue import Queue
import numpy as np


class MOG_filter():
    """
    This class is used to apply the Mixture of gaussian cuda
    implementation of open-cv in a thread and store the obtained
    mask + the original frame in a buffer.
    """

    def __init__(self,
                 input_buffer,
                 buff_size=500,
                 device='cuda',
                 ):

        # Store the input buffer
        self.input = input_buffer
        self.device = device
        self.cuda = False
        if device == 'cuda':
            self.cuda = True

        # The MOG filter
        self.MOG = None
        self.cuda_stream = None
        self.closing_filter = None

        if self.cuda:
            self.MOG = cv2.cuda.createBackgroundSubtractorMOG(history=50, nmixtures=2, backgroundRatio=0.02)
        else:
            self.MOG = cv2.bgsegm.createBackgroundSubtractorMOG(history=100)

        # The output buffer
        self.buffer = Queue(maxsize=buff_size)
        self.device = device

        # Thread management
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.end = False
        self.pred_end = False

    def start(self):

        self.thread.start()
        return self

    def update(self):

        while self.input.more():

            # Perform iterations to fill the buffer
            if not self.buffer.full():

                # Get a frame
                frame, frame_idx = self.input.read()
                # Apply MOG and morphological transformations
                if self.cuda:
                    mask = self.MOG.apply(frame, 0.1, self.cuda_stream)

                else:
                    mask = self.MOG.apply(frame)
                    # Closing
                    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.opening_kernel)
                # Append in the buffer
                self.buffer.put((frame, mask, frame_idx))

            else:
                # If the buffer is already full, wait 0.01 sec
                time.sleep(0.01)

        # When the previous algorithm end
        self.end = True

    def read(self):
        """
        :return: The oldest tuple in the buffer:
        (<original frame>, <the MOG mask>, <the frame index>)
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