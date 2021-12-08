"""
[ELEN0016-2]
François Lievens
Julien Hubar
Matthias Pirlet
December 2020

This file implement the first thread of our algorithm who
have have the role of reading successive frames from the
input video by using OpenCV.
All succesives frames and corresponding indexes are stored
in a buffer who will be access by the next threads.
"""
import time
import cv2
from threading import Thread
from queue import Queue


class ReadingBuffer():
    """
    Implementation of a video reading buffer inspired by
    https://www.simonwenkel.com/2020/02/13/opencv_cuda_for_background_subtraction.html
    The general frame of this thread was keep to implement our others threads.
    """

    def __init__(self,
                 path_list,
                 buff_size=500,
                 device='cuda'):

        # Index in the path list
        self.path_idx = 0
        self.path_list = path_list

        # The video reading object
        self.vid = cv2.VideoCapture(path_list[0])
        # The buffer
        self.buffer = Queue(maxsize=buff_size)
        self.device = device

        # Thread management
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.end = False

        # Count the frames
        self.frame_idx = 0

    def start(self):

        self.thread.start()
        return self

    def update(self):

        while not self.end:

            # Fill the buffer
            if not self.buffer.full():
                # Get the frame
                ret, frame = self.vid.read()

                if not ret:
                    # End of the files to read
                    if self.path_idx + 1 >= len(self.path_list):
                        self.end = True
                        break
                    # If there are remaining files to read
                    else:
                        self.path_idx += 1
                        self.vid.release()
                        self.vid = cv2.VideoCapture(self.path_list[self.path_idx])
                        continue

                # If cuda available, put on the gpu memory
                if self.device == 'cuda':
                    frame = cv2.cuda_GpuMat(frame)
                    # Convert to RGB
                    frame = cv2.cuda.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Append in the buffer
                self.buffer.put((frame, self.frame_idx))
                self.frame_idx += 1

            else:
                # If the buffer is already full, wait 0.1 sec
                time.sleep(0.1)

        # close the video reader
        self.vid.release()

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