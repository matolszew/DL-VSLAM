import numpy as np
from matplotlib import pyplot as plt
import cv2
import argparse
import os

class SLAM:
    """First iteration of SLAM class

    Args:
        ...
    """
    def __init__(self, width, height, calibration_matrix, match_treshold=50):
        self.width = width
        self.height = height
        self.calibration_matrix = calibration_matrix
        self.match_treshold = match_treshold

        self.feature_detector = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.W = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        self.projection_ref = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])
        self.projection_cur = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])

        self.ref_img = None
        self.cur_img = None
        self.rotation_hypotheses = np.empty((4,3,3))
        self.translation_hypotheses = np.empty((4,3,1))
        self.map = np.zeros((10000, 3))
        self.points_i = 0

    def run(self, img):
        """Perform one step

        Args:
            img (): ...
        """
        if self.ref_img is None:
            self.ref_img = img
            self.kp_ref = self.feature_detector.detect(img, None)
            self.kp_ref, self.kp_ref_des = self.feature_detector.compute(img, self.kp_ref)
            return
        else:
            self.cur_img = img
            self.kp_cur = self.feature_detector.detect(img, None)
            self.kp_cur, self.kp_cur_des = self.feature_detector.compute(img, self.kp_ref)

        matches = self.bf.match(self.kp_ref_des, self.kp_cur_des)
        matches_ut = [m for m in matches if m.distance < self.match_treshold ]
        ref_indicies = [m.trainIdx for m in matches_ut]
        cur_indicies = [m.queryIdx for m in matches_ut]

        kp_ref_pt = np.array([self.kp_ref[i].pt for i in ref_indicies])
        kp_cur_pt = np.array([self.kp_cur[i].pt for i in cur_indicies])

        kp_ref_pt_homo = cv2.convertPointsToHomogeneous(kp_ref_pt)
        kp_cur_pt_homo = cv2.convertPointsToHomogeneous(kp_cur_pt)

        # Calculation of homography and fundamental could be parallelized
        homography_matrix, _ = cv2.findHomography(kp_ref_pt, kp_cur_pt, 0)
        homography_matrix_inv = np.linalg.inv(homography_matrix)
        fundamental_matrix = cv2.findFundamentalMat(kp_ref_pt, kp_cur_pt, cv2.FM_8POINT)[0]
        # TODO: Add checking correctness of the homography and fundamental

        essential_matrix = np.matmul(np.matmul(calibration_matrix.T, fundamental_matrix), calibration_matrix)
        U, S, Vh = np.linalg.svd(essential_matrix)

        self.rotation_hypotheses[0,:] = self.rotation_hypotheses[1,:] = np.matmul(np.matmul(U, self.W), Vh)
        self.rotation_hypotheses[2,:] = self.rotation_hypotheses[3,:] = np.matmul(np.matmul(U, self.W.T), Vh)

        self.translation_hypotheses[0,:,0] = self.translation_hypotheses[2,:,0] = U[:,2]
        self.translation_hypotheses[1,:,0] = self.translation_hypotheses[3,:,0] = -U[:,2]

        # Choosing proper hypothesis
        i_max = 0
        score_max = 0
        for i in range(4):
            projection_matrix = np.concatenate(
                (np.dot(self.calibration_matrix, self.rotation_hypotheses[i,:]),
                 np.dot(self.calibration_matrix, self.translation_hypotheses[i_max,:])), axis = 1)
            points_homo = cv2.triangulatePoints(self.projection_ref, projection_matrix, kp_ref_pt.T, kp_cur_pt.T)
            points = cv2.convertPointsFromHomogeneous(points_homo.T)
            points = np.reshape(points, (points.shape[0], points.shape[2]))

            score = 0
            # Could be vectorize
            for point in points:
                point2 = np.matmul(self.rotation_hypotheses[i,:], point) + self.translation_hypotheses[i,0,:]
                if point[2] > 0 and point2[2] > 0:
                    score += 1
            if score > score_max:
                score_max = score
                i_max = i

        self.projection_cur = np.concatenate(
            (np.dot(self.calibration_matrix, self.rotation_hypotheses[i_max,:]),
             np.dot(self.calibration_matrix, self.translation_hypotheses[i_max,:])), axis = 1)
        points_homo = cv2.triangulatePoints(self.projection_ref, self.projection_cur, kp_ref_pt.T, kp_cur_pt.T)
        points = cv2.convertPointsFromHomogeneous(points_homo.T)
        points = np.reshape(points, (points.shape[0], points.shape[2]))

        self.map[self.points_i:self.points_i+points.shape[0],:] = points
        self.points_i += points.shape[0]
        print('Number of points: ', self.points_i)

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

    slam = SLAM(width=w, height=h, calibration_matrix=calibration_matrix)

    for name in image_names[:15]:
        img = cv2.imread(path + '/' + name, cv2.IMREAD_GRAYSCALE)
        slam.run(img)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(slam.map[:,0], slam.map[:,1], slam.map[:,2])
    plt.show()

    #print(dir(key_point[0]))
    # print('angle: ', key_point[1].angle)
    # print('class_id: ', key_point[1].class_id)
    # print('octave: ', key_point[1].octave)
    # print('pt: ', key_point[1].pt)
    # print('response: ', key_point[1].response)
    # print('size: ', key_point[1].size)
