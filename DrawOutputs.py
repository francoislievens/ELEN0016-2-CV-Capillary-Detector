import cv2
import pandas as pd
import numpy as np
import sys




if __name__ == '__main__':

    df = pd.read_csv('Output_Results/results.csv', sep=',', header=None)

    vid = cv2.VideoCapture('Original_Dataset/images/CV2021_GROUP02/group2.mp4')

    data = []
    for i in range(0, df.shape[0]):
        frame_idx = int(df.iloc[i][0].replace('frame_', ''))
        data.append((frame_idx, df.iloc[i][1],
                     df.iloc[i][2],
                     df.iloc[i][3],
                     df.iloc[i][4]))
    ret = True
    frame_idx = 0

    while ret:

        ret, frame = vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for i in range(0, len(data)):
            if data[i][0] == frame_idx:
                frame = cv2.rectangle(frame, (data[i][1], data[i][2]), (data[i][3], data[i][4]),
                                      (125, 125, 125), 8)
        cv2.imshow('window', frame)
        cv2.waitKey()






        frame_idx += 1
