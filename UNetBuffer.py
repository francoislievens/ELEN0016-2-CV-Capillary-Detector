"""
[ELEN0016-2]
François Lievens
Julien Hubar
Matthias Pirlet
December 2020

This thread takes outputs frames from the droplet detector thread,
who consist in cropped images around droplets.
This images are tensorized and stored into a buffer who can
be accessed by the UNet Thread to count cells.
"""
import time
from threading import Thread
from queue import Queue
import torch

class UNetBuffer():

    def __init__(self,
                 input_buffer,
                 buff_size=500,
                 device='cuda',
                 display=False):

        # Store the input buffer
        self.input = input_buffer
        self.display = display

        # The output buffer
        self.buffer = Queue(maxsize=buff_size)
        self.device = device
        self.cuda = False
        if device == 'cuda':
            self.cuda = True

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

                # Get data
                frame, droplet_idx, frame_idx, width = self.input.read()

                # Tensorise:
                frame = torch.tensor(frame).to(self.device)

                # Append to the buffer
                self.buffer.put((frame, droplet_idx, frame_idx, width))
            else:
                # If the buffer is already full, wait 0.1 sec
                time.sleep(0.01)

        # When the previous algorithm end
        self.end = True

    def read(self, batch_size):
        """
        :return: The oldest image in the buffer
        """
        i = 0
        outputs = []
        outputs_idx = []
        outputs_frame_idx = []
        outputs_width = []
        while self.buffer.qsize() > 0 and len(outputs) < batch_size:
            frame, drp_idx, frame_idx, width = self.buffer.get()
            outputs.append(frame)
            outputs_idx.append(drp_idx)
            outputs_frame_idx.append(frame_idx)
            outputs_width.append(width)

        # concatenate tensors
        return torch.cat(outputs, dim=0), outputs_idx, outputs_frame_idx, outputs_width

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