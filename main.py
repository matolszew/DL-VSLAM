import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
import pandas as pd

from slam import SLAM

import tqdm

#grand_truth = pd.read_csv("data/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv")

w = 800
h = 600
calibration_matrix = np.array([
        [0.535719308086809*w, 0, 0.493248545285398*w],
        [0, 0.669566858850269*h, 0.500408664348414*h],
        [0, 0, 1]
])
sigma = 0.897966326944875
# w=1241
# h=376
# calibration_matrix = np.array([
#     [7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02],
#     [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02],
#     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00],
# ])
distortion_coefficients = None
# w = 752
# h = 480
# calibration_matrix = np.array([
#         [458.654, 0, 367.215],
#         [0, 457.296, 248.375],
#         [0, 0, 1]
# ])
# distortion_coefficients = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])
# initial_pose = np.array([
#         [0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556],
#         [ 0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024],
#         [-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038],
#         [0.0, 0.0, 0.0, 1.0]])


position = np.zeros((10000, 3))
position = [[],[],[]]
position_gt = np.zeros_like(position)

def animate(i, slam, path):
    global position
    print(i)
    #name = path + '/' + names[(i+1)*2]
    name = path + '/' + i
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    #img = cv2.undistort(img, calibration_matrix, distortion_coefficients)
    img = cv2.resize(img, (w,h))
    slam.run3(img)

    ax.clear()
    #position[i] = slam.state_estimator.position[:,0]
    # position_gt[i] = grand_truth.iloc[i, 1:4].values
    #print(position_gt[i])
    t = slam.state_estimator.position[:,0]
    position[0].append(slam.state_estimator.position[0,0])
    position[1].append(slam.state_estimator.position[1,0])
    position[2].append(slam.state_estimator.position[2,0])
    cd = slam.state_estimator.camera_direction
    r = slam.state_estimator.rotation_matrix
    vx = r[:,0]
    vy = r[:,1]
    d = 1e1
    ax.quiver(t[0], t[1], t[2],
              d*vx[0], d*vx[1], d*vx[2], color='r')
    ax.quiver(t[0], t[1], t[2],
              d*vy[0], d*vy[1], d*vy[2], color='g')
    ax.quiver(t[0], t[1], t[2],
              d*cd[0], d*cd[1], d*cd[2])
    # ax.set_xlim3d(-5e0,5e0)
    # ax.set_ylim3d(-5e0,5e0)
    # ax.set_zlim3d(-5e0,5e0)
    ax.scatter3D(slam.state_estimator.points[:,0], slam.state_estimator.points[:,1], slam.state_estimator.points[:,2])
    #ax.scatter3D(position[0], position[1], position[2])
    # ax.scatter3D(position_gt[:,0], position_gt[:,1], position_gt[:,2], color='g')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('path', metavar='path', type=str, help='data')
    args = parser.parse_args()

    path = args.path
    image_names = sorted(os.listdir(path))
    names_list = image_names[2:]

    slam = SLAM(dt=0.05, width=w, height=h, calibration_matrix=calibration_matrix, distortion_coefficients=distortion_coefficients)
    # t = tqdm.tqdm(image_names, total=len(image_names))
    # vel = np.zeros((632, 3))
    # ang = np.zeros((632, 3))
    # i = 0
    # for name in t:
    #     img = cv2.imread(path + '/' + name, cv2.IMREAD_GRAYSCALE)
    #     img = cv2.resize(img, (w,h))
    #     vel[i], ang[i] = slam.run2(img)
    #     i+=1
    #
    # df = pd.DataFrame({
    #     'vx': vel[:,0],
    #     'vy': vel[:,1],
    #     'vz': vel[:,2],
    #     'wx': ang[:,0],
    #     'wy': ang[:,1],
    #     'wz': ang[:,2],
    # })
    # df.to_csv('velocities.csv')

    img = cv2.imread(path + '/' + image_names[0], cv2.IMREAD_GRAYSCALE)
    img = cv2.undistort(img, calibration_matrix, distortion_coefficients)
    img = cv2.resize(img, (w,h))
    slam.run3(img)
    img = cv2.imread(path + '/' + image_names[1], cv2.IMREAD_GRAYSCALE)
    img = cv2.undistort(img, calibration_matrix, distortion_coefficients)
    img = cv2.resize(img, (w,h))
    slam.run3(img)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ani = animation.FuncAnimation(fig, animate, frames=names_list, interval=1, fargs=(slam, path))
    plt.show()

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(slam.position[:,0], slam.position[:,1], slam.position[:,2], '-')
    #ax.scatter(slam.state_estimator.points[:,0], slam.state_estimator.points[:,1], slam.state_estimator.points[:,2])
    #ax.set_xlim3d(-20,20)
    #ax.set_ylim3d(-20,20)
    #ax.set_zlim3d(-20,20)
    #plt.show()

    #print(dir(key_point[0]))
    # print('angle: ', key_point[1].angle)
    # print('class_id: ', key_point[1].class_id)
    # print('octave: ', key_point[1].octave)
    # print('pt: ', key_point[1].pt)
    # print('response: ', key_point[1].response)
    # print('size: ', key_point[1].size)
