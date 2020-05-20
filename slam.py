import numpy as np
import cv2
import scipy.spatial.transform as transform

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

        self.feature_detector = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

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
        self.cur_img = None
        self.rotation_hypotheses = np.empty((4,3,3))
        self.translation_hypotheses = np.empty((4,3,1))
        self.map = np.zeros((20000, 3))
        self.map_descriptors = np.zeros((20000, 32), dtype=np.uint8)
        self.position = np.zeros((2200, 3))
        self.points_i = 0
        self.pos_i = 0

    def run(self, img):
        """Perform one step

        Args:
            img (): ...
        """
        if self.ref_img is None:
            self.ref_img = img
            return

        kp_ref_pt, kp_cur_pt, description = self._find_matches(img)

        # Calculation of homography and fundamental could be parallelized
        #homography_matrix, _ = self._find_homography(kp_ref_pt, kp_cur_pt)

        fundamental_matrix = cv2.findFundamentalMat(kp_ref_pt, kp_cur_pt, cv2.FM_RANSAC)[0]
        # TODO: Add checking correctness of the homography and fundamental

        rt_change = self._calculate_camera_shift(kp_ref_pt, kp_cur_pt, fundamental_matrix)
        #print('change: ', rt_change)

        rt_cur = np.matmul(self.rt_ref, rt_change)
        #print(rt_cur)

        # TODO: actualize camera state
        #print(self.old_R)
        #print(R_change)

        self.position[self.pos_i,:] = rt_cur[:3,3]
        self.pos_i += 1

        r = transform.Rotation.from_matrix(rt_cur[:3,:3])
        quaternion = r.as_quat()
        print(quaternion)

        #projection_cur = np.concatenate(
        #    (np.dot(self.calibration_matrix, new_R_matrix),
        #     np.dot(self.calibration_matrix, new_T)), axis = 1)
        #projection_cur = np.zeros((3,4))
        #projection_cur[:3,:3] = np.matmul(self.calibration_matrix, new_R)
        #projection_cur[:,3] = new_T[:,0]

        points_homo = cv2.triangulatePoints(self.projection_ref[:3,:], np.matmul(self.calibration_matrix, rt_cur[:3,:]), kp_ref_pt.T, kp_cur_pt.T)
        points = cv2.convertPointsFromHomogeneous(points_homo.T)
        points = np.reshape(points, (points.shape[0], points.shape[2]))

        # TODO: Choose points from map that could be visible

        if np.sum(self.map_descriptors) > 0:
            # TODO: Look for matches only in visible points
            matches = self.bf.match(description, self.map_descriptors[:self.points_i,:])
            for m in matches:
                if m.distance > self.match_treshold:
                    self.map[self.points_i,:] = points[m.queryIdx]
                    self.map_descriptors[self.points_i,:] = description[m.queryIdx]
                    self.points_i += 1
                else:
                    # TODO: actualize points in map
                    pass
        else:
            self.map[self.points_i:self.points_i+points.shape[0],:] = points
            self.map_descriptors[self.points_i:self.points_i+points.shape[0],:] = description
            self.points_i += points.shape[0]
        #print('Number of points: ', self.points_i)

        # TODO: actualize camera state based on visible points in map

        self.ref_img = img
        self.projection_ref = np.matmul(self.calibration_matrix, rt_cur[:3,:])
        self.rt_ref = rt_cur

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
            # projection_matrix = np.concatenate(
            #     (np.dot(self.calibration_matrix, rotation_hypotheses[i,:]),
            #      np.dot(self.calibration_matrix, translation_hypotheses[i,:])), axis = 1)
            rt_matrix = np.hstack((rotation_hypotheses[i], translation_hypotheses[i]))
            points_homo = cv2.triangulatePoints(self.base_projection, np.matmul(self.calibration_matrix, rt_matrix), kp_ref_pt.T, kp_cur_pt.T)
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

        return np.vstack((np.hstack((rotation_hypotheses[i], translation_hypotheses[i])), [0, 0, 0, 1]))

    def _find_matches(self, cur_img):
        """
        """
        # TODO: calculate ref only once
        kp_ref = self.feature_detector.detect(self.ref_img, None)
        kp_ref, kp_ref_des = self.feature_detector.compute(self.ref_img, kp_ref)

        kp_cur = self.feature_detector.detect(cur_img, None)
        kp_cur, kp_cur_des = self.feature_detector.compute(cur_img, kp_cur)

        matches = self.bf.match(kp_ref_des, kp_cur_des)
        matches_bt = [m for m in matches if m.distance < self.match_treshold]

        ref_indicies = [m.queryIdx for m in matches_bt]
        cur_indicies = [m.trainIdx for m in matches_bt]

        kp_ref_pt = np.array([kp_ref[i].pt for i in ref_indicies])
        kp_cur_pt = np.array([kp_cur[i].pt for i in cur_indicies])
        description = np.array([kp_ref_des[i] for i in ref_indicies])

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
