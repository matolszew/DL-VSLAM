import numpy as np
from matplotlib import pyplot as plt
import cv2
import argparse
import os

from slam import SLAM

import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('path', metavar='path', type=str, help='data')
    args = parser.parse_args()

    path = args.path

    image_names = sorted(os.listdir(path))

    w = 1280
    h = 1024
    calibration_matrix = np.array([
            [0.535719308086809*w, 0, 0.493248545285398*w],
            [0, 0.669566858850269*h, 0.500408664348414*h],
            [0, 0, 1]
    ])
    sigma = 0.897966326944875

    slam = SLAM(width=w, height=h, calibration_matrix=calibration_matrix)

    t = tqdm.tqdm(image_names, total=len(image_names))

    for name in t:
        #print(name)
        #fig = plt.figure()
        img = cv2.imread(path + '/' + name, cv2.IMREAD_GRAYSCALE)
        #plt.imshow(img)
        #plt.show()
        img2 = cv2.undistort(img, calibration_matrix, sigma)
        # plt.imshow(img2)
        # plt.show()
        slam.run(img)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(slam.position[:,0], slam.position[:,1], slam.position[:,2], '-')
    #ax.scatter(slam.map[:,0], slam.map[:,1], slam.map[:,2])
    #ax.set_xlim3d(-20,20)
    #ax.set_ylim3d(-20,20)
    #ax.set_zlim3d(-20,20)
    plt.show()

    #print(dir(key_point[0]))
    # print('angle: ', key_point[1].angle)
    # print('class_id: ', key_point[1].class_id)
    # print('octave: ', key_point[1].octave)
    # print('pt: ', key_point[1].pt)
    # print('response: ', key_point[1].response)
    # print('size: ', key_point[1].size)
