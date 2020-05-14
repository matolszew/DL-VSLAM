import numpy as np
from matplotlib import pyplot as plt
import cv2
import argparse
import os

import utilites

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('path', metavar='path', type=str, help='data')
    args = parser.parse_args()

    path = args.path

    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    image_names = sorted(os.listdir(path))[:100]

    img = cv2.imread(path + '/' + image_names[0], cv2.IMREAD_GRAYSCALE)
    kp_ref = orb.detect(img, None)
    kp_ref, kp_ref_des = orb.compute(img, kp_ref)

    img = cv2.imread(path + '/' + image_names[1], cv2.IMREAD_GRAYSCALE)
    kp_cur = orb.detect(img, None)
    kp_cur, kp_cur_des = orb.compute(img, kp_ref)

    matches = bf.match(kp_ref_des, kp_cur_des)

    tresh = 30
    matches_ut = [m for m in matches if m.distance < tresh ]
    ref_indicies = [m.trainIdx for m in matches_ut]
    cur_indicies = [m.queryIdx for m in matches_ut]

    kp_ref_pt = [kp_ref[i].pt for i in ref_indicies]
    kp_cur_pt = [kp_cur[i].pt for i in cur_indicies]

    print(utilites.direct_linear_transformation(kp_ref_pt, kp_cur_pt))
    print(utilites.eight_point_algorithm(kp_ref_pt, kp_cur_pt))

    # img2 = cv2.drawKeypoints(img, kp_ref, None, color=(0,255,0), flags=0)
    # for m in matches_ut:
    #     img2 = cv2.drawKeypoints(img2, [kp_ref[m.trainIdx]], None, color=(255,0,0), flags=0)
    # plt.imshow(img2)
    # plt.show()

    #print(dir(key_point[0]))
    # print('angle: ', key_point[1].angle)
    # print('class_id: ', key_point[1].class_id)
    # print('octave: ', key_point[1].octave)
    # print('pt: ', key_point[1].pt)
    # print('response: ', key_point[1].response)
    # print('size: ', key_point[1].size)

    # print(key_point_description[1])
    #
