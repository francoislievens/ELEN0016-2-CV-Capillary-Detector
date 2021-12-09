# ELEN0016-2-CV-Capillary-Detector

This repository contains the implementation of our project in the context of
the 2021 Computer Vision lecture.

The goal of this project was to provide a model who can count accurately biological
cells who pass inside droplets in a microscopical capillary. The input data are sample
by a high frequency camera (we assume the real time to be 300 frames per seconds).

With this code, we are providing a model who can detect efficiently droplets and cells
and returning:
1. The total number of frames read
2. The total number of different droplets who pass inside the capillary
3. The total number of different cells
4. The coordinates of each detected cells and droplets (only for one of the
multiple frames in which they appear during her travelling inside the capillary).
5. The histogram who describe the number of droplets who contain each possible 
number of cells.


At the end of the day, our model can perform up to 546 frames per second on our
computer of reference (AMD Ryzen 7300X (8 cores / 16 threads), Nvidia GTX1070). 

### Preliminary Notes

In order to realize the best performances, our code use cuda to paralelize deep learning and computer vision operation. </br>

This imply that if the parameter 'cuda' is given for the variable DEVICE, this code will be using the CUDA implementation of some operations of the library OpenCV. So, OpenCV cuda must be previously compilled and installed in the environment that you are using. </br>

If a gpu compatible with cuda or the cuda-OpenCV library is not available, this code can be run by using the value 'cpu' for the following variable DEVICE.

Our code need the following requirements:

    albumentations==1.1.0
    matplotlib==3.5.0
    numba==0.54.1
    numpy==1.20.0
    pandas==1.3.4
    Pillow==8.4.0
    scikit-learn==1.0.1
    Shapely==1.8.0
    tensorboard==2.7.0
    torch==1.10.0+cu102
    torchvision==0.11.1+cu102
    tqdm==4.62.3
    + OpenCV 4.5.4 compilled with cuda and contribs or OpenCV python + contribs if used on cpu.

We would like to point out that installing the necessary environment for this project can be complex:
The installations of some libraries such that "Albumentation" or "Jupyter Notebook" can 
install the "OpenCV" installation by the "OpenCV Headless" version who will break the actual "OpenCV"
version. 
In the case of Albumentation, it's possible to install it safely by using the pip command
"pip install -U albumentations --no-binary qudida,albumentations"

### How to excute the code 
The easier way to run this code is to run the [NOTEBOOK](https://github.com/francoislievens/ELEN0016-2-CV-Capillary-Detector/blob/main/NOTEBOOK.ipynb)
This noteook contains the following steps:

    1. Data Downloading from Cytomine platform
    2. Statistical Analysis of the downloaded annotations
    3. Dataset creation for the UNet Training
    4. UNet Training
    5. Plot training curves
    6. Full Video Analysis: The full execution of the algorithm over the whole given dataset. This execution
    generate the file 'Output_Results/results.csv' who contain the histogram of the number
    droplets for each possible number of cells that it can contain and the coordinates of each
    droplets and cells at the frame of detection
    7. Evaluation step: A new prediction is performed over the whole dataset and outputs
    are compare with the ground truth (the given annotations).

### Others informations over the code

The Notebook was provided to provide a more general and easy way to analyze the
implementation. This repository contains also all our sub-procedures:

1. The different threads:
   1. [ReadingBuffer.py](https://github.com/francoislievens/ELEN0016-2-CV-Capillary-Detector/blob/main/ReadingBuffer.py):
   This part of the project is responsible of the reading of the input video on the disk and storing
   extracted frames in a buffer who can be access by the next threads.
   2. [MOG_Filter.py](https://github.com/francoislievens/ELEN0016-2-CV-Capillary-Detector/blob/main/MOG_Filter.py):
   This thread applies the Mixture of Gaussian filter on the frames. Each frames and mask are stored into
   his buffer to be accessed by the next thread.
   3. [DropletDetector.py](https://github.com/francoislievens/ELEN0016-2-CV-Capillary-Detector/blob/main/DropletDetector.py):
   This thread applies our strategy to detect and count droplets.
   For each detected droplets, a sub-frame is cropped around the droplet and resized to be later
   analyze in order to count potential cells in it.
   4. [UNetBuffer.py](https://github.com/francoislievens/ELEN0016-2-CV-Capillary-Detector/blob/main/UNetBuffer.py):
   This thread tensorize and store all droplets images provided by the previous thread to keep it available
   for the UNet.
   5. [UNetThread.py](https://github.com/francoislievens/ELEN0016-2-CV-Capillary-Detector/blob/main/UNetThread.py):
   This thread implement our cell's detection strategy. It starts by using our previously trained UNet to generate
   a mask and then our 2 dimensional preaks detection strategy search coordinates of droplets.

2. Two Version of our algorithm are implemented and can be executed:
   1. [main.py](https://github.com/francoislievens/ELEN0016-2-CV-Capillary-Detector/blob/main/main.py):
   It contain our final implementation of the algorithm and is the same as in the Notebook. By default, it will
   execute an analysis of all the input Cytomine video dowloaded in "Original_Dataset". It's also possible to change
   the input video list to target new specific videos. The video names and folder have to follow the same logic
   that the original dataset.
   2. [PredictForSubmit.py](https://github.com/francoislievens/ELEN0016-2-CV-Capillary-Detector/blob/main/PredictForSubmit.py):
   It contain a simpler version of our algorithm (no multi-thread) who will detect all objects for each frames
   without counting. All detected objects are store into "Output_Results/results_frame_interest.csv". If the variable
   SHOW is set to True, OpenCV will show a window of the video with bounding boxes drawn for each objects.

3. Files who manages the UNet:
   1. [UNet.py](https://github.com/francoislievens/ELEN0016-2-CV-Capillary-Detector/blob/main/UNet.py):
   Contain the implementation of the UNet that we are using. This implementation is not from us but from
   [NeuroSYS-pl](https://github.com/NeuroSYS-pl/objects_counting_dmap/).
   2. [UNet_DatasetBuilder.py](https://github.com/francoislievens/ELEN0016-2-CV-Capillary-Detector/blob/main/UNet_DatasetBuilder.py):
   This code use the original dataset from Cytomine and his annotations to generate a dataset of cropped droplets
   frames that we used for the UNet training.
   3. [UNet_DataHandler.py](https://github.com/francoislievens/ELEN0016-2-CV-Capillary-Detector/blob/main/UNet_DataHandler.py):
   Contain the implementation of our data handler to manage data from UNet_DatasetBuilder, and during the training:
   ballancing the dataset, applying data augmentation and creating the masks who will be used as target
   for the UNet.
   4. [UNet_Trainer](https://github.com/francoislievens/ELEN0016-2-CV-Capillary-Detector/blob/main/UNet_Trainer.py):
   This code implement the training and evaluation procedures of our UNet.
   5. [Model](https://github.com/francoislievens/ELEN0016-2-CV-Capillary-Detector/tree/main/Model/):
   This folder contain the weights and trainings logs of our UNet model. The final weights that we have chosen
   are UNet_A.

4. Some other files who contain sub procedures:
   1. [utils.py](https://github.com/francoislievens/ELEN0016-2-CV-Capillary-Detector/blob/main/utils.py).
   2. [CytomineDownloader.py](https://github.com/francoislievens/ELEN0016-2-CV-Capillary-Detector/blob/main/CytomineDownloader.py).
   3. [DrawOutputs.py](https://github.com/francoislievens/ELEN0016-2-CV-Capillary-Detector/blob/main/DrawOutputs.py): Can be 
   used to see a video with bounding boxes drawn on objects by using outputs of the main algorithms.
   4. [DataStatistics.py](https://github.com/francoislievens/ELEN0016-2-CV-Capillary-Detector/blob/main/DataStatistics.py):
   Who perform som statistical analysis of the original dataset in order to better set our different thresholds
   during our detections procedures.
   





