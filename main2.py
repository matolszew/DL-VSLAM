import cv2
import numpy as np
import argparse
import os
import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d

from orbslam import ORBSLAM

def animate(i, slam, path):
    name = path + '/' + i
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    img = cv2.undistort(img, camera_matrix, distortion_coefficients)
    slam.update(img)

    t = slam._last_key_frame.camera_position
    r = slam._last_key_frame.camera_rotaion_matrix
    vx = r[:,0]
    vy = r[:,1]
    vz = r[:,2]
    d = 1e0

    ax.clear()
    ax.quiver(t[0], t[1], t[2],
              d*vx[0], d*vx[1], d*vx[2], color='r')
    ax.quiver(t[0], t[1], t[2],
              d*vy[0], d*vy[1], d*vy[2], color='g')
    ax.quiver(t[0], t[1], t[2],
              d*vz[0], d*vz[1], d*vz[2])
    ax.scatter3D(slam.map.points3d[:,0], slam.map.points3d[:,1], slam.map.points3d[:,2])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('path', metavar='path', type=str, help='data')
    parser.add_argument('feature_detector', metavar='feature_detector', type=str)
    args = parser.parse_args()

    path = args.path
    feature_detector = args.feature_detector
    image_names = sorted(os.listdir(path))

    camera_matrix = np.array([
            [458.654, 0, 367.215],
            [0, 457.296, 248.375],
            [0, 0, 1]
    ])
    distortion_coefficients = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])

    slam = ORBSLAM(camera_matrix, (480,752), feature_detector=feature_detector)

    img = cv2.imread(path + '/' + image_names[0], cv2.IMREAD_GRAYSCALE)
    img = cv2.undistort(img, camera_matrix, distortion_coefficients)
    slam.update(img)

    img = cv2.imread(path + '/' + image_names[2], cv2.IMREAD_GRAYSCALE)
    img = cv2.undistort(img, camera_matrix, distortion_coefficients)
    slam.update(img)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ani = animation.FuncAnimation(fig, animate, frames=image_names[2:-1:2], interval=1, fargs=(slam, path))
    plt.show()

    # t = tqdm.tqdm(image_names, total=len(image_names))
    # for name in t:
    #     img = cv2.imread(path + '/' + name, cv2.IMREAD_GRAYSCALE)
    #     img = cv2.undistort(img, camera_matrix, distortion_coefficients)
    #     slam.update(img)
