import cv2
import numpy as np
import argparse
import os
import tqdm

from orbslam import ORBSLAM

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('path', metavar='path', type=str, help='data')
    args = parser.parse_args()

    path = args.path
    image_names = sorted(os.listdir(path))

    camera_matrix = np.array([
            [458.654, 0, 367.215],
            [0, 457.296, 248.375],
            [0, 0, 1]
    ])
    distortion_coefficients = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])

    slam = ORBSLAM(camera_matrix)

    t = tqdm.tqdm(image_names, total=len(image_names))
    for name in t:
        img = cv2.imread(path + '/' + name, cv2.IMREAD_GRAYSCALE)
        img = cv2.undistort(img, camera_matrix, distortion_coefficients)
        slam.update(img)
