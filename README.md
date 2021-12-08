# ELEN0016-2-CV-Capillary-Detector

### How to excute the code 
 • The easier way to run this code is to run the [NOTEBOOK](https://github.com/francoislievens/ELEN0016-2-CV-Capillary-Detector/blob/main/NOTEBOOK.ipynb)
 • This noteook contains the following steps:
    - Data Downloading from Cytomine platform
    - Statistical Analysis of the downloaded data
    - Dataset creation for the UNet Training
    - UNet Training
    - Plot training curves
    - Video Analysis
    - Load all components
    - Prediction Step
    - Model Evaluation

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
