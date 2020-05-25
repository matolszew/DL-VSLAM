import numpy as np
import cv2
import scipy.spatial.transform as transform

from kalman_filter import EKF

from time import time

class SLAM:
    """First iteration of SLAM class

    Args:
        ...
    """
    def __init__(self, width, height, calibration_matrix, match_treshold=64):
        self.width = width
        self.height = height
        self.calibration_matrix = calibration_matrix
        self.match_treshold = match_treshold
        self.state_estimator = EKF()

        self.feature_detector = cv2.ORB_create(nfeatures=100)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.W = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        self.base_projection = np.matmul(self.calibration_matrix, np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ]))
        self.projection_ref = np.vstack((self.base_projection, np.zeros((1,4))))

        self.old_R = np.eye(3)
        self.old_T = np.zeros((3,1))

        self.rt_ref = np.eye(4)
        self.ref_img = None

        self.position = np.zeros((2200, 3))
        self.points_i = 0
        self.pos_i = 0

        self.z_versor = np.array([[0, 0, 1]]).T

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

        #t = time()
        kp_ref_pt, kp_cur_pt, cur_description = self._find_matches(img)
        #print(kp_ref_pt.shape, kp_cur_pt.shape, cur_description.shape)
        #print('Finding matches on imgs: ', time() - t)

        #t = time()
        if self.state_estimator.points_number > 0:
            camera_direction = np.matmul(self.state_estimator.rotation_matrix, self.z_versor)
            #print(camera_direction)
            #print(self.state_estimator.points_viewing_versor)
            point_angle = np.array([np.dot(camera_direction[:,0], point_versor) for point_versor in self.state_estimator.points_viewing_versor])
            #print(point_angle)
            correct_angle = point_angle > np.cos(np.deg2rad(60))
            #print(self.state_estimator.points[correct_angle].shape)
            #print(self.state_estimator.position.shape)
            camera_point_direction = (self.state_estimator.points[correct_angle]-self.state_estimator.position.T) \
                / np.expand_dims(np.linalg.norm(self.state_estimator.points[correct_angle]-self.state_estimator.position.T, axis=1), axis=1)
            #print(camera_point_direction)
            camera_point_angle = np.array([np.dot(camera_direction[:,0], cp_dir) for cp_dir in camera_point_direction])
            in_front_of_camera = camera_point_angle > 0
            correct_angle_indicies = np.argwhere(correct_angle)[:,0]
            visible_points_indicies = correct_angle_indicies[in_front_of_camera]

            #print(self.state_estimator.points_descriptor[visible_points_indicies])
            #print(cur_description)
            matches = self.bf.match(self.state_estimator.points_descriptor[visible_points_indicies], cur_description)
            matches_bt = [m for m in matches if m.distance < self.match_treshold]
            old_points_indicies = [visible_points_indicies[m.queryIdx] for m in matches_bt]
            old_points_descriptors = self.state_estimator.points[old_points_indicies]
            #print(old_points_indicies)
            cur_points_old_indicies = [m.trainIdx for m in matches_bt]
            cur_points_new_indicies = [i for i in range(len(kp_cur_pt)) if i not in cur_points_old_indicies]
        else:
            old_points = np.zeros((0,3))
            old_points_descriptors = np.zeros((0,32))
            old_points_indicies = np.array([], dtype=np.int)
            cur_points_new_indicies = range(len(kp_cur_pt))
            cur_points_old_indicies = []
        #print('Finding matches in map: ', time() - t)
        # Calculation of homography and fundamental could be parallelized
        #homography_matrix, _ = self._find_homography(kp_ref_pt, kp_cur_pt)
        points_in_map = []

        #t = time()
        # For now use only triangulation
        if True:
            extrinsic_params = self._find_extrinsic_triangulation(kp_ref_pt, kp_cur_pt)
        else:
            extrinsic_params = self._find_extrinsic_pnp(self.state_estimator.points[cur_points_old_indicies], kp_cur_pt[cur_indicies_of_old_points])
        #print('Finding extrinscic: ', time() - t)
        #print(extrinsic_params)

        new_pos = extrinsic_params[:3,3]
        new_rot = extrinsic_params[:3,:3]
        #print(new_pos)
        self.position[self.pos_i,:] = new_pos
        self.pos_i += 1

        #print(self.state_estimator.extrinsic_matrix_3x4)
        #print(extrinsic_params[:3,:])
        #t = time()
        points_homo = cv2.triangulatePoints(np.matmul(self.calibration_matrix, self.state_estimator.extrinsic_matrix_3x4), np.matmul(self.calibration_matrix, extrinsic_params[:3,:]), kp_ref_pt.T, kp_cur_pt.T)
        points = cv2.convertPointsFromHomogeneous(points_homo.T)
        points = np.reshape(points, (points.shape[0], points.shape[2]))
        #print('Triangulation: ', time() - t)

        new_points = points[cur_points_new_indicies]
        old_points = points[cur_points_old_indicies]
        new_desciptions = cur_description[cur_points_new_indicies]
        #print('new ', new_points.shape)
        #print('old ', old_points.shape)

        #t = time()
        self.state_estimator.update(new_pos, new_rot, new_points, new_desciptions, old_points, old_points_descriptors, old_points_indicies)
        #print('Update Kalman: ', time() - t)

        self.ref_img = img

    def _find_extrinsic_triangulation(self, reference_points, current_points):
        """Find extrinscic matrix using triangulation

        Args:

        """
        fundamental = cv2.findFundamentalMat(reference_points, current_points, cv2.FM_RANSAC)[0]
        extrinscic_change = self._calculate_camera_shift(reference_points, current_points, fundamental)

        return np.matmul(self.state_estimator.extrinsic_matrix_4x4, extrinscic_change)

    def _find_extrinsic_pnp(self, object_points, image_points):
        """Find extrinscic matrix using PnP algorithm

        Args:

        """
        # NOTE Could be provided initial guess of rvec and tvec
        #      so maybe get prediction of it from Kalman filter
        succes, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, self.calibration_matrix, distCoeffs=None)
        #print(cv2.solvePnPRansac(object_points, image_points, self.calibration_matrix, distCoeffs=None))
        rotation_matrix = self.state_estimator.rotation_matrix

        return np.vstack((
            np.hstack((rotation_matrix, tvec)),
            np.array([0, 0, 0, 1])
        ))

    def _calculate_camera_shift(self, kp_ref_pt, kp_cur_pt, fundamental_matrix):
        """
        """
        essential_matrix = np.matmul(np.matmul(self.calibration_matrix.T, fundamental_matrix), self.calibration_matrix)
        U, S, Vh = np.linalg.svd(essential_matrix)

        rotation_hypotheses = np.zeros((4,3,3))
        translation_hypotheses = np.zeros((4,3,1))
        rotation_hypotheses[0,:] = rotation_hypotheses[1,:] = np.matmul(np.matmul(U, self.W), Vh)
        rotation_hypotheses[2,:] = rotation_hypotheses[3,:] = np.matmul(np.matmul(U, self.W.T), Vh)
        translation_hypotheses[0,:,0] = translation_hypotheses[2,:,0] = U[:,2]
        translation_hypotheses[1,:,0] = translation_hypotheses[3,:,0] = U[:,2]

        # Choosing proper hypothesis
        i_max = 0
        score_max = 0
        for i in range(4):
            rt_matrix = np.hstack((rotation_hypotheses[i], translation_hypotheses[i]))
            points_homo = cv2.triangulatePoints(self.base_projection, np.matmul(self.calibration_matrix, rt_matrix), kp_ref_pt.T, kp_cur_pt.T)
            points = cv2.convertPointsFromHomogeneous(points_homo.T)
            points = np.reshape(points, (points.shape[0], points.shape[2]))

            score = 0
            # Could be vectorize
            for point in points:
                point2 = np.matmul(rotation_hypotheses[i,:], point) + translation_hypotheses[i,0,:]
                if point[2] > 0 and point2[2] > 0:
                    score += 1
            if score > score_max:
                score_max = score
                i_max = i

        return np.vstack((np.hstack((rotation_hypotheses[i], translation_hypotheses[i])), [0, 0, 0, 1]))

    def _find_matches(self, cur_img):
        """
        """
        kp_cur = self.feature_detector.detect(cur_img, None)
        kp_cur, kp_cur_des = self.feature_detector.compute(cur_img, kp_cur)

        matches = self.bf.match(self.kp_ref_des, kp_cur_des)
        matches_bt = [m for m in matches if m.distance < self.match_treshold]

        ref_indicies = [m.queryIdx for m in matches_bt]
        cur_indicies = [m.trainIdx for m in matches_bt]

        kp_ref_pt = np.array([self.kp_ref[i].pt for i in ref_indicies])
        kp_cur_pt = np.array([kp_cur[i].pt for i in cur_indicies])
        description = np.array([kp_cur_des[i] for i in cur_indicies])

        self.kp_ref = kp_cur
        self.kp_ref_des = kp_cur_des

        return kp_ref_pt, kp_cur_pt, description

    def _find_homography(self, kp_ref_pt, kp_cur_pt):
        """
        """
        homography_matrix, _ = cv2.findHomography(kp_ref_pt, kp_cur_pt, cv2.RANSAC)
        score = self._matrix_score(homography_matrix, kp_ref_pt, kp_cur_pt, 5.99)

        return homography_matrix, score

    def _matrix_score(self, matrix, ref_pt, cur_pt, treshhold):
        """
        """
        return None
