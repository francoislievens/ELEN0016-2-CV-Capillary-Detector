import time
import cv2
from threading import Thread
from queue import Queue
import torch
from UNet import UNet
import numpy as np
import sys
from utils import count_peaks_2d


class UNetThread():

    def __init__(self,
                 input_buffer,
                 model_path='Model',
                 model_name='UNet_A',
                 device='cuda',
                 batch_size=10,
                 debug=False):

        self.debug = debug

        # Instanciate the model
        self.model = UNet(input_filters=1).to(device)
        self.batch_size = batch_size
        self.device = device
        self.cuda = False
        self.offset = 10
        if self.device == 'cuda':
            self.cuda = True
        # Load model weights
        try:
            self.model.load_state_dict(
                torch.load('{}/{}/model_weights.pt'.format(model_path, model_name), map_location=self.device))
            print('Unet sucessfully loaded')
        except:
            print('ERROR during the model loading: can not restore weights')
            sys.exit(1)

        # Store the number of cells
        self.total_cell = 0
        self.droplet_per_cell = []
        # And final results per droplet index:
        self.final_results = []
        self.histogram = np.zeros(10)

        # Thread management
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.end = False
        self.pred_end = False

        # Store the input buffer
        self.input = input_buffer

        self.global_idx = 0

    def start(self):

        self.thread.start()
        return self

    def update(self):

        while self.input.more():

            # Load a batch of data
            droplets, droplets_idx, frames_idx, droplets_width = self.input.read(batch_size=self.batch_size)
            droplets = droplets.reshape(-1, 240, 240)
            # Makes predictions
            self.model.eval()
            with torch.no_grad():
                preds = self.model(droplets).reshape(-1, 240, 240)

            preds = preds.cpu().numpy()

            for i in range(0, preds.shape[0]):
                if self.debug:
                    dbg_frame = droplets[i, :, :].cpu().numpy()
                    dbg_mask = preds[i, :, :]

                cnt, positions = count_peaks_2d(preds[i])
                if cnt > 0:
                    for pos in positions:
                        # Get coordinates for the frame without resizing
                        abs_x = (pos[0] / 240) * droplets_width[i]
                        self.final_results.append((frames_idx[i], droplets_idx[i], cnt, abs_x, pos[1]))

                        if self.debug:
                            dbg_frame = cv2.rectangle(dbg_frame,
                                                      (pos[0] - self.offset, pos[1] - self.offset),
                                                      (pos[0] + self.offset, pos[1] + self.offset),
                                                      (125, 125, 125), 2)
                            dbg_mask = cv2.rectangle(dbg_mask,
                                                     (pos[0] + self.offset, pos[1] + self.offset),
                                                     (pos[0] - self.offset, pos[1] - self.offset),
                                                     (125, 125, 125), 2)

                    self.droplet_per_cell.append(cnt)
                    self.total_cell += cnt
                    self.histogram[cnt] += 1

                    if self.debug:
                        cv2.imwrite('Intermediate_Results/Droplet_Detector/{}_d.jpg'.format(frames_idx[i]), dbg_frame)
                        cv2.imwrite('Intermediate_Results/Droplet_Detector/{}_e.jpg'.format(frames_idx[i]), dbg_mask)


            """
            if self.display:
                for j in range(0, preds.shape[0]):
                    disp_frame = preds[j].reshape(240, 240).cpu().numpy()
                    print('droplet idx: {} - nb cell: {}'.format(self.global_idx, count[j] / 100))
                    drp = droplets[j].reshape(240, 240).cpu().numpy()
                    cv2.imshow('droplet', drp)
                    cv2.imshow('cell_msk', disp_frame)
                    print('Peaks: {}'.format(peaks[j].cpu().numpy()))
                    cv2.waitKey()
            """
            self.global_idx += 1

        # End of the program
        self.end = True

    def running(self):

        if self.end:
            return False
        else:
            return True

    def stop(self):

        self.end = True
        self.thread.join()
