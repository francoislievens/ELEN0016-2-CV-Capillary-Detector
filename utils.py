import cv2
import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
import math


def shuffle_lst(lst, seed=1):
    """
    Simple code to shuffle a list
    with a fixed seed
    """
    np.random.seed(seed)
    idx_lst = np.arange(len(lst))
    np.random.shuffle(idx_lst)

    return [lst[i] for i in idx_lst]


def heat_map_gen(coordinates, img_shape):
    """
    This code generate a 240 x 240 px heat-
    map to produce target images during the
    Unet training process.
    Heat maps contain a 2D gaussian
    centered on each cell center coordinates
    such that the sum of the values in the
    heatmap is equal to 100 * the number of
    cells in the image.
    """

    # Init the array
    label = np.zeros(img_shape, dtype=np.float32)

    # If no annotations (so no cell)
    if len(coordinates) == 0:
        return label

    # Map the center of each object
    for x, y in coordinates:
        if x >= 240:
            x = 239
        if y >= 240:
            y = 239
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        label[int(y), int(x)] = 100

    # Apply a gaussian filter convolution
    label = gaussian_filter(label, sigma=(1, 1), order=0)

    return label


@njit
def get_droplet_coordinates(col_sum, col_sum_out_tsh, col_sum_in_tsh, droplet_min_width):
    """
    The goal of this function is to detect coordinates
    of droplets by travelling the values of the sum of
    columns in the mask provided by the MOG filter.
    This code use "col_sum_in_tsh" as a threshold to detect
    that we enter a droplet while travelling the X axis and
    "col_sum_out_tsh" to detect when we out a droplet.
    To avoid some false droplet out when the sum of values
    """
    # Init an array for coordinates
    array_end_idx = 0
    drop_coord = np.empty((4, 4), dtype=np.int32)     # Note: we assume no more than 4 droplet in a frame
    start_idx = 0
    in_droplet = False
    for i in range(1, len(col_sum)-1):
        mean_val = np.mean(col_sum[i - 1:i + 1])
        if in_droplet and mean_val < col_sum_in_tsh:

            # We had to add an average value with next columns to avoid false positive
            end_avg_idx = i + 20
            if end_avg_idx >= len(col_sum):
                end_avg_idx = len(col_sum) - 1
            # If false exit:
            if np.mean(col_sum[i:end_avg_idx] > col_sum_in_tsh):
                continue

            in_droplet = False
            # Don't take if to small
            if i - start_idx < droplet_min_width:
                continue
            drop_coord[array_end_idx, :] = [start_idx, i, int((start_idx + i) / 2), i - start_idx]
            array_end_idx += 1
            continue
        elif not in_droplet and mean_val >= col_sum_out_tsh:
            start_idx = i
            in_droplet = True
    # If a droplet reach the right end:
    if in_droplet:
        drop_coord[array_end_idx, :] = [start_idx, len(col_sum), int((start_idx + len(col_sum)) / 2), len(col_sum) - start_idx]
        array_end_idx += 1
    # returned coord: (x_start, x_end, center_x, width)
    return drop_coord[0:array_end_idx, :]

@njit
def check_new_droplets(drop_coord, last_frame_drop, last_frame_detect, dist_thresh, drop_counter):

    # Check if no new droplets
    find_idx = np.empty(len(drop_coord))
    prev_find = np.empty(len(drop_coord))
    for i in range(len(drop_coord)):
        find_idx[i] = False
        prev_find[i] = False

    for i in range(0, drop_coord.shape[0]):
        # Search correspondances in droplets from previous frame
        find = False
        for j in range(0, len(last_frame_drop)):
            move_dst = drop_coord[i][2] - last_frame_drop[j][2]
            if move_dst > 0 and move_dst < dist_thresh:
                find = True
                if last_frame_detect[j]:
                    prev_find[i] = True
                break
        if not find:
            find_idx[i] = True
            drop_counter += 1

    return drop_counter, find_idx, prev_find

@njit
def check_new_droplets_no_prev(drop_coord, drop_counter):

    # Check if no new droplets
    find_idx = np.empty(len(drop_coord))
    prev_find = np.empty(len(drop_coord))
    for i in range(len(drop_coord)):
        find_idx[i] = False
        prev_find[i] = False
    for i in range(0, drop_coord.shape[0]):
        drop_counter += 1
        find_idx[i] = True

    return drop_counter, find_idx, prev_find

def count_peaks_2d(matrix,
                   pk_min_thresh=5):
    counter = 0
    results = []
    debug = False

    # Makes the sum over columns
    col_sum = np.sum(matrix, axis=0)

    # Find peaks index
    peaks, _ = find_peaks(col_sum, distance=5)

    # Check if multiple cell in this peak
    for pk in peaks:
        if debug:
            print('------------')
            print('col_sum peaks: {}'.format(pk))
        # Check if the peak is greater than the threshold
        if col_sum[pk] >= pk_min_thresh:
            sub_peaks, _ = find_peaks(matrix[:, pk], distance=5)
            for sub_pk in sub_peaks:
                if matrix[sub_pk, pk] > 1.5:
                    counter += 1
                    results.append([pk, sub_pk])
                    if debug:
                        plt.plot(matrix[:, pk], label='Y val for x={}'.format(pk))


    if debug and counter > 0:
        plt.plot(col_sum, label='Column sum')
        plt.title('Cell detection')
        plt.legend()
        plt.show()
        plt.close()
    return counter, results

def count_peaks_2d_bis(matrix,
                   pk_min_thresh=5):
    debug = False
    # Horizontal prediction:
    counter_h = 0
    results_h = []

    # Makes the sum over columns
    col_sum = np.sum(matrix, axis=0)

    # Find peaks index
    peaks, _ = find_peaks(col_sum, distance=5)

    # Check if multiple cell in this peak
    for pk in peaks:
        if debug:
            print('------------')
            print('col_sum peaks: {}'.format(pk))
        # Check if the peak is greater than the threshold
        if col_sum[pk] >= pk_min_thresh:
            sub_peaks, _ = find_peaks(matrix[:, pk], distance=5)
            for sub_pk in sub_peaks:
                if matrix[sub_pk, pk] > 1.5:
                    counter_h += 1
                    results_h.append([pk, sub_pk])
            if debug:
                plt.plot(matrix[:, pk], color='red')
                plt.title('Sub peak {}'.format(pk))
                plt.show()

    # Vertical prediction:
    counter_v = 0
    results_v = []
    line_sum = np.sum(matrix, axis=1)
    peaks, _ = find_peaks(line_sum, distance=5)
    for pk in peaks:
        # Check if the peak is greater than the threshold
        if line_sum[pk] >= pk_min_thresh:
            sub_peaks, _ = find_peaks(matrix[pk, :], distance=5)
            for sub_pk in sub_peaks:
                if matrix[pk, sub_pk] > 1.5:
                    counter_v += 1
                    results_v.append([sub_pk, pk])

    if debug:
        plt.plot(col_sum)
        plt.show()
        plt.close()
    if counter_h >= counter_v:
        return counter_h, results_h
    return counter_v, results_v



def plot_boxes(frame, bboxes, find_idx=None):

    if bboxes is not None and len(bboxes) > 0:
        idx = 0
        for box in bboxes:
            start_point = [int(box[0]), 0]
            end_point = [int(box[1]), 240]
            frame = cv2.rectangle(frame, start_point, end_point, (125, 125, 125), 5)
            if find_idx is not None:
                if find_idx[idx]:
                    start_point = [int((box[0] + box[1])/2), 110]
                    end_point = [int((box[0] + box[1]) / 2), 130]
                    frame = cv2.rectangle(frame, start_point, end_point, (125, 125, 125), 10)
            idx += 1
    return frame

def plot_droplet_detection(frame, mask, col_sum, frame_idx, drop_coord, find_idx, path, device='cpu'):

    if device == 'cuda':
        frame = frame.download()

    # Plot the column sum graph
    plt.plot(np.arange(len(col_sum)), col_sum)
    plt.ylim([0, 40000])
    plt.title('Frame {}'.format(frame_idx))
    plt.savefig('{}/{}_c.jpg'.format(path, frame_idx))
    plt.close()

    # Draw bounding boxes
    frame = plot_boxes(frame, drop_coord, find_idx)
    mask = plot_boxes(mask, drop_coord, find_idx)
    # Draw frames and mask
    cv2.imwrite('{}/{}_a.jpg'.format(path, frame_idx), frame)
    cv2.imwrite('{}/{}_b.jpg'.format(path, frame_idx), mask)
    print('Export frame {}'.format(frame_idx))


def prepare_annotations(path):
    """
    Load the annotation csv and sort it
    """
    annot = pd.read_csv(path, sep=';')

    # First we sort by frame index (slice)
    annot = annot.sort_values(by=['Slice'], ascending=True, ignore_index=True)

    # Get the max slice index
    max_slice = annot['Slice'].max()

    # Build an array indexed by slices and who condain all bboxes
    data = []
    for i in range(0, max_slice):
        # Select elements with this slice
        df = annot[annot['Slice'] == i]

        # If no box in this slice
        if df.shape[0] == 0:
            data.append(None)
            continue

        # A dictionary: Keys = tracker of the droplet, Values = [droplet_coord, nb_cells, [cells coord]]
        sub_data = {}
        # Read all droplets in this frame
        for j in range(0, df.shape[0]):

            # Don't care about cells in this loop
            if not 'Droplet' in df.iloc[j]['Terms']:
                continue
            if 'POINT' in df.iloc[j]['Geometry']:
                continue

            # Get Tracker index
            try:
                tracker = int(str(df.iloc[j]['Track']).replace('[', '').replace(']', ''))
            except:
                # In some cases we have a droplet without tracker
                continue

            # Get droplet coordinates and clean it
            coord = df.iloc[j]['Geometry']
            coord = coord.replace('POLYGON ((', '')
            coord = coord.replace('))', '')
            coord = coord.replace(',', '')
            coord = coord.split(' ')

            tl_x = float(coord[0])  # Top left x
            tl_y = float(coord[1])  # Top left y
            tr_x = float(coord[2])  # Top right x
            tr_y = float(coord[3])  # Top right y
            br_x = float(coord[4])  # Bottom right x
            br_y = float(coord[5])  # ...
            bl_x = float(coord[6])
            bl_y = float(coord[7])

            x_l = min(tl_x, tr_x, br_x, bl_x)
            x_r = max(tl_x, tr_x, br_x, bl_x)

            # Our general box format for droplets: (start_x, end_x, center_x, width)
            center_x = int((x_r + x_l) / 2)
            width = int(x_r - x_l)
            drp_coord = [int(x_l), int(x_r), center_x, width]
            if drp_coord[0] < 0:
                drp_coord[0] = 0
            if drp_coord[1] >= 1600:
                drp_coord[1] = 1599
            sub_data[str(tracker)] = [drp_coord, 0, []]

        # Now we look at cell's
        for j in range(0, df.shape[0]):

            # Don't care about Droplets in this loop
            if not 'Cell' in df.iloc[j]['Terms']:
                continue

            # Get Cell coordinates and clean it
            coord = df.iloc[j]['Geometry']
            coord = coord.replace('POLYGON ((', '')
            coord = coord.replace('))', '')
            coord = coord.replace(',', '')
            coord = coord.split(' ')
            tl_x = float(coord[0])  # Top left x
            tl_y = float(coord[1])  # Top left y
            tr_x = float(coord[2])  # Top right x
            tr_y = float(coord[3])  # Top right y
            br_x = float(coord[4])  # Bottom right x
            br_y = float(coord[5])  # ...
            bl_x = float(coord[6])
            bl_y = float(coord[7])

            x_l = min(tl_x, tr_x, br_x, bl_x)
            x_r = max(tl_x, tr_x, br_x, bl_x)
            y_b = max(tl_y, tr_y, br_y, bl_y)
            y_t = min(tl_y, tr_y, br_y, bl_y)

            # General cell box format: (x_center, y_center, width, height)
            x_center = int((x_r + x_l) / 2)
            y_center = int((y_b + y_t) / 2)
            width = int(x_r - x_l)
            height = int(y_b - y_t)
            crd = [x_center, y_center, width, height]

            # Check in which droplet it come
            done = False
            for key in sub_data.keys():
                if x_center < sub_data[key][0][1] and x_center > sub_data[key][0][0]:
                    done = True
                    sub_data[key][1] += 1
                    sub_data[key][2].append(crd)
                    break
            if not done:
                #print('WARNING: can not find a droplet for slice {}, cell {}'.format(i, crd))
                pass

        data.append(sub_data)



    return data


def avg_smoothing(seq, window_size):

    start_idx = 0
    end_idx = window_size
    target_idx = int(window_size / 2)
    output = np.zeros(len(seq) + window_size)
    for i in range(int(window_size / 2)):
        output[i] = seq[0]
        output[-i] = seq[-1]
    output[int(window_size / 2):int(window_size / 2) + len(seq)] = seq

    while end_idx < len(output):
        output[target_idx] = np.mean(seq[start_idx:end_idx])
        start_idx += 1
        end_idx += 1
        target_idx += 1

    return output[int(window_size / 2): int(window_size / 2) + len(seq)]

def convert_annot(df):
    """
    Output format: (frame index, type, track, x0, y0, x1, y1)
    """
    output = []
    for i in range(df.shape[0]):
        if 'POINT' in df.iloc[i]['Geometry']:
            continue
        tmp = [int(df.iloc[i]['Slice'])]
        if 'Droplet' in df.iloc[i]['Terms']:
            tmp.append('droplet')
            # Catch the fact that some droplets have no tracker
            try:
                track = int(str(df.iloc[i]['Track']).replace('[', '').replace(']', ''))
                tmp.append(track)
            except:
                continue
            # Check if cells in this droplet
        if 'Cell' in df.iloc[i]['Terms']:
            tmp.append('cell')
            tmp.append(0)   # No tracking for cells

        # Get coordinates and clean it
        coord = df.iloc[i]['Geometry']
        coord = coord.replace('POLYGON ((', '')
        coord = coord.replace('))', '')
        coord = coord.replace(',', '')
        coord = coord.split(' ')

        x_coord = [float(coord[0]), float(coord[2]), float(coord[4]), float(coord[6])]
        y_coord = [float(coord[1]), float(coord[3]), float(coord[5]), float(coord[7])]

        tmp.append(np.min(x_coord))
        tmp.append(np.min(y_coord))
        tmp.append(np.max(x_coord))
        tmp.append(np.max(y_coord))

        for i in range(3, len(tmp)):
            if tmp[i] < 0:
                tmp[i] = 0

        output.append(tmp)

    return output

def convert_pred(df):
    """
    Get the same format than converted annotations
    (frame index, type, track, x0, y0, x1, y1)
    with 0 for slice
    """
    output = []
    for i in range(0, df.shape[0]):
        tmp = []
        tmp.append(int(df.iloc[i][0].replace('frame_', '')))
        tmp.append(df.iloc[i][5])
        tmp.append(0)
        tmp.append(df.iloc[i][1])
        tmp.append(df.iloc[i][2])
        tmp.append(df.iloc[i][3])
        tmp.append(df.iloc[i][4])
        output.append(tmp)
        for i in range(3, len(tmp)):
            if tmp[i] < 0:
                tmp[i] = 0
    return output

def get_result_idx(frame_idx, annot):

    opt = -1
    idx = 0
    for itm in annot:
        if itm[0] == frame_idx:
            opt = idx
            break
        idx += 1
    return opt

def intersectionOverUnion(pred, target):
    """
    Inspired by Aladdin Persson
    source:  https://github.com/aladdinpersson/Machine-Learning-Collection/
    """
    x1 = np.max([pred[0], target[0]])
    y1 = np.max([pred[1], target[1]])
    x2 = np.min([pred[2], target[2]])
    y2 = np.min([pred[3], target[3]])

    a = (x2 - x1)
    b = (y2 - y1)
    if a < 0:
        a = 0

    if b < 0:
        b = 0
    intersect = a * b

    area_1 = abs((pred[2] - pred[0]) * (pred[3] - pred[1]))
    area_2 = abs((target[2] - target[0]) * (target[3] - target[1]))

    return intersect / (area_1 + area_2 - intersect + 1e-6)

def euclidDist(boxA, boxB):
    #print('box_a: {} box_b: {}'.format(boxA, boxB))
    center_A_x = (boxA[0] + boxA[2]) / 2
    center_A_y = (boxA[1] + boxA[3]) / 2
    center_B_x = (boxB[0] + boxB[2]) / 2
    center_B_y = (boxB[1] + boxB[3]) / 2

    dist = math.sqrt((center_A_x - center_B_x)**2 + (center_A_y - center_B_y)**2)
    return dist

def gound_truth_compare(pred_lst, trg_lst, iou_thresh=0.5):

    find_pred = [False] * len(pred_lst)
    find_trg = [False] * len(trg_lst)
    dist = []
    ious = []

    for i in range(len(pred_lst)):

        for j in range(len(trg_lst)):

            if find_trg[j]:
                continue

            pred = pred_lst[i][3:]
            trg = trg_lst[j][3:]

            iou = intersectionOverUnion(pred, trg)
            if iou > iou_thresh:
                ious.append(iou)
                dist.append(euclidDist(pred, trg))
                find_pred[i] = True
                find_trg[j] = True


    TP = 0
    FP = 0
    FN = 0
    for itm in find_pred:
        if itm:
            TP += 1
        else:
            FP += 1
    for itm in find_trg:
        if not itm:
            FN += 1

    return TP, FP, FN, ious, dist

















