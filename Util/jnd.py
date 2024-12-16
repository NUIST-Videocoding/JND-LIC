import os
import re
import cv2
import numpy as np



def max(array):
    a = 0
    shape = array.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if array[i, j, k] > a:
                    a = array[i, j, k]
    return a


def JND_cal(input):
    mask_h = np.array([[1 / 3, 1 / 3, 1 / 3],
                       [0, 0, 0],
                       [-1 / 3, -1 / 3, -1 / 3]])
    mask_v = np.array([[1 / 3, 0, -1 / 3],
                       [1 / 3, 0, -1 / 3],
                       [1 / 3, 0, -1 / 3]])
    mask_me = np.array([[1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
                        [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
                        [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
                        [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
                        [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25]])

    (height, width, channel) = input.shape
    Gh = cv2.filter2D(input, -1, mask_h) / 1.0
    Gv = cv2.filter2D(input, -1, mask_v) / 1.0
    Me = cv2.filter2D(input, -1, mask_me) / 1.0
    # print(Me)

    Lc = np.sqrt((np.power(Gh, 2) + np.power(Gv, 2)) / 2.0)
    N = 7

    Vm = (1.84 * np.power(Lc, 2.4)) / (np.power(Lc, 2) + 26 * 26) * (0.3 * np.power(N, 2.7)) / (np.power(N, 2) + 1)
    La = np.zeros_like(input)

    for i in range(channel):
        for j in range(height):
            for k in range(width):
                if Me[j, k, i] <= 127:
                    La[j, k, i] = 17 * (1 - np.sqrt(Me[j, k, i] / 127.0))
                else:
                    La[j, k, i] = (3.0 / 128) * (Me[j, k, i] - 127) + 3

    JND = La + Vm - 0.3 * np.minimum(La, Vm)
    return JND


if __name__ == '__main__':
    root_dir = "D:/data/kodak/Kodak_dataset"
    save_dir = "D:/data/kodak/jnd_png"
    list1 = os.listdir(root_dir)
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(list1)):
        list1[i] = list1[i].strip().replace('\n', '')
        root1 = os.path.join(root_dir + "/" + list1[i])
        split_strings = re.split(r'\.', list1[i])
        save_root1 = save_dir + "/" + split_strings[0] + ".npz"
        img = cv2.imread(root1)
        jnd = JND_cal(img)
        print(save_root1)
        np.savez(save_root1, jnd)

