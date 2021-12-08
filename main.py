import numpy as np
import time
import matplotlib.pyplot as plt
import os
from ReadingBuffer import ReadingBuffer
from MOG_Filter import MOG_filter
from DropletDetector import DropletDetector
from UNetBuffer import UNetBuffer
from UNetThread import UNetThread


DATA_PATH = 'Original_Dataset'
UNET_PATH = 'Model'
UNET_NAME = 'UNet_A'
EVAL_BUFFER_LOAD = False

DEVICE = 'cuda'

BUFFER_SIZE = 500
DEBUG = False


def counter(video_list):

    # Get all path to target videos
    target_vid = []
    for itm in video_list:
        # Get groupe number
        gp_str = itm[-2:]
        gp = int(gp_str)
        tmp = '{}/images/{}/group{}.mp4'.format(DATA_PATH, itm, gp)
        target_vid.append(tmp)

    # Build the reading buffer
    buff_read = ReadingBuffer(path_list=target_vid,
                              buff_size=BUFFER_SIZE,
                              device=DEVICE)
    # The MOG filter thread
    buff_mog = MOG_filter(input_buffer=buff_read,
                          buff_size=BUFFER_SIZE,
                          device=DEVICE)
    # The Droplet detector/counter
    buff_droplet = DropletDetector(input_buffer=buff_mog,
                                   buff_size=BUFFER_SIZE,
                                   device=DEVICE,
                                   debug=DEBUG)

    # A buffer for the UNet: do the tensorisation step
    buff_UNet = UNetBuffer(input_buffer=buff_droplet,
                           buff_size=BUFFER_SIZE,
                           device=DEVICE,
                           display=False)

    # The UNet thread to count cells
    UNet_thread = UNetThread(input_buffer=buff_UNet,
                             model_path=UNET_PATH,
                             model_name=UNET_NAME,
                             device=DEVICE,
                             batch_size=10,
                             debug=DEBUG)
    start_time = time.time()
    buff_read.start()
    buff_mog.start()
    buff_droplet.start()
    buff_UNet.start()
    UNet_thread.start()


    # Filling track
    fill_reader = []
    fill_mog = []
    fill_drp_counter = []
    fill_UNet_buff = []
    while not UNet_thread.end:
    #for i in range(200):

        if EVAL_BUFFER_LOAD:
            fill_reader.append(buff_read.buffer.qsize())
            fill_mog.append(buff_mog.buffer.qsize())
            fill_drp_counter.append(buff_droplet.buffer.qsize())
            fill_UNet_buff.append(buff_UNet.buffer.qsize())

        time.sleep(0.1)

    end_time = time.time()
    if EVAL_BUFFER_LOAD:
        x_axis = np.arange(len(fill_reader))
        plt.plot(x_axis, fill_reader, label='Reader Buffer', color='Green', linewidth=0.5)
        plt.plot(x_axis, fill_mog, label='MOG buffer', color='Blue', linewidth=0.5)
        plt.plot(x_axis, fill_drp_counter, label='Droplet counter buffer', color='Red', linewidth=0.5)
        plt.plot(x_axis, fill_UNet_buff, label='UNet buffer', color='Purple', linewidth=0.5)
        plt.legend()
        plt.yscale('log')
        plt.title('Buffer load')
        plt.show()
        plt.close()

    print('Counted droplets: {}'.format(buff_droplet.drop_counter))
    print('Counted cells: {}'.format(UNet_thread.total_cell))
    print('Computing Time: {}'.format(end_time - start_time))
    total_frame_idx = buff_read.frame_idx
    print('Readed frames: {}'.format(total_frame_idx))
    print('FPS: {}'.format(total_frame_idx / (end_time - start_time)))

    # Histogram part
    nb_zero_cell = buff_droplet.drop_counter - np.sum(UNet_thread.histogram)
    UNet_thread.histogram[0] = nb_zero_cell
    histo = []
    for itm in UNet_thread.histogram.tolist():
        histo.append(str(int(itm)))
    file = open('Output_Results/results.csv', 'w')
    file.write(','.join(histo))
    file.write('\n')
    file.close()

    # Write final results
    results_drp = buff_droplet.final_results
    results_cell = UNet_thread.final_results

    to_print = []
    for i in range(0, total_frame_idx):
        # To compute absolute coordinates of the cell later
        x_start = 0
        for j in range(0, len(results_drp)):
            if results_drp[j][0] == i:
                to_print.append('frame_{},{},0,{},240,droplet'.format(i,
                                                                        results_drp[j][1],
                                                                        results_drp[j][2]))
                x_start = results_drp[j][1]
        for j in range(0, len(results_cell)):
            if results_cell[j][0] == i:
                to_print.append('frame_{},{},{},{},{},cell'.format(i,
                                                                     results_cell[j][3]-20+x_start,
                                                                     results_cell[j][4]-20,
                                                                     results_cell[j][3]+20+x_start,
                                                                     results_cell[j][4]+20))
    file = open('Output_results/results.csv', 'a')
    file.write('\n'.join(to_print))
    file.close()









if __name__ == '__main__':

    #video_list = ['CV2021_GROUP02', 'CV2021_GROUP03', 'CV2021_GROUP04']
    #video_list = ['CV2021_GROUP02']
    #video_list = ['CV2021_GROUP01','CV2021_GROUP02', 'CV2021_GROUP03', 'CV2021_GROUP04', 'CV2021_GROUP05', 'CV2021_GROUP06', 'CV2021_GROUP07',
    #              'CV2021_GROUP08','CV2021_GROUP09', 'CV2021_GROUP10', 'CV2021_GROUP11', 'CV2021_GROUP12', 'CV2021_GROUP13', 'CV2021_GROUP14']

    #video_list = os.listdir('Original_Dataset/images')
    video_list = ['CV2021_GROUP21']
    print('* ------------------------------------------------ *')
    print('* Input video list:')
    for itm in video_list:
        print('* {}'.format(itm))
    print('* ------------------------------------------------ *')

    counter(video_list)