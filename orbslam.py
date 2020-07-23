import numpy as np
import cv2
from scipy.optimize import least_squares

from map import Map
from key_frame import KeyFrame

class ORBSLAM:
    """ORB-SLAM class

    Based on "ORB-SLAM: A Versatile and Accurate Monocular SLAM System",
    Raúl Mur-Artal, J. M. M. Montiel, Juan D. Tardós

    Args:
        feature_detector (string):
    """
    def __init__(self, camera_matrix, feature_detector='orb', initial_pose=None):
        self.camera_matrix = camera_matrix
        if feature_detector=='orb':
            self.feature_detector = cv2.ORB_create(nfeatures=100)
        else:
            raise ValueError('Not known feature detector')
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.match_treshold = 64

        self.keyframes = []
        self.map = Map()

        self.initialize = True
        self.features = []
        self.features_desc = []

        self.base_projection = np.matmul(self.camera_matrix, np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ]))

    def update(self, img):
        """

        Args:
            img (cv2.):
        """
        keypoints = self.feature_detector.detect(img, None)
        keypoints, descriptors = self.feature_detector.compute(img, keypoints)
        self.keyframes.append(KeyFrame(keypoints, descriptors))

        if len(self.keyframes) == 1:
            # TODO: add possibility to initialize camera rotation and position
            self.keyframes[0].update_rotation_from_quat([0,0,0,1])
            self.keyframes[0].update_position(np.zeros((3,1)))
            return

        if self.initialize:
            self._initialize()

    def _initialize(self):
        """
        """
        matches = self.matcher.match(self.keyframes[0].descriptors, self.keyframes[-1].descriptors)
        matches = [m for m in matches if m.distance < self.match_treshold]

        ref_indicies = np.array([m.queryIdx for m in matches])
        cur_indicies = np.array([m.trainIdx for m in matches])

        points_ref = self.keyframes[0].points2d[ref_indicies]
        points_cur = self.keyframes[-1].points2d[cur_indicies]
        description = self.keyframes[0].descriptors[ref_indicies]

        # TODO: Calculate in parallel threads
        F, F_score, F_inliers = find_fundamental_matrix(points_ref, points_cur)
        H, H_score, H_inliers = find_homography(points_ref, points_cur)

        rate = H_score / (H_score+F_score)

        if rate <= 0.45:
            points_ref = points_ref[F_inliers[:,0]]
            points_cur = points_cur[F_inliers[:,0]]
            description = description[F_inliers[:,0]]
            ref_indicies = ref_indicies[F_inliers[:,0]]
            cur_indicies = cur_indicies[F_inliers[:,0]]
            points3d, rt_matrix, inliers = self._position_from_fundamental(F, points_ref, points_cur)
            m_indices = self.map.add_points(points3d, description[inliers], self.matcher, self.match_treshold)
            self.keyframes[0].update_indicies(m_indices, ref_indicies[inliers])
            self.keyframes[-1].update_indicies(m_indices, cur_indicies[inliers])
            self.keyframes[-1].update_position_and_rotation(rt_matrix)
        else:
            # TODO: add calculation position from homography
            pass

        self.bundle_adjustment()

    def bundle_adjustment(self):
        """
        """
        n = 7*len(self.keyframes) + 3*len(self.map)
        x0 = np.empty((n))
        for i, keyframe in enumerate(self.keyframes):
            j = i*7
            x0[j:j+4] = keyframe.camera_quaternion
            x0[j+4:j+7] = keyframe.camera_position[:,0]
        x0[-3*len(self.map):] = self.map.points3d.flatten()

        res = least_squares(self._ba_projection, x0,
                            #jac=projection_jacobian, tr_solver='lsmr',
                            #jac_sparsity=projection_sparsity,
                            verbose=0)
        for i, keyframe in enumerate(self.keyframes):
            j = i*7
            keyframe.update_rotation_from_quat(res.x[j:j+4])
            keyframe.update_position(res.x[j+4:j+7])
        print(self.map.points3d)
        self.map.update_points(res.x[-3*len(self.map):].reshape((len(self.map),3)))
        print(self.map.points3d)
        input("Press Enter to continue...")

    def _ba_projection(self, x):
        """
        """
        # NOTE: For now will work only when map points are visible in all keyframes
        Nf = len(self.keyframes)
        Nm = len(self.map)
        Np = 2*Nm*Nf
        out = np.zeros((Np))
        for i, frame in enumerate(self.keyframes):
            p_img, map_indicies = frame.points_on_img
            p, j = cv2.projectPoints(self.map.points3d[map_indicies], frame.camera_rotation_vector, frame.camera_position, self.camera_matrix, None)

            for p_i, m_i in enumerate(map_indicies):
                out[i*2+2*m_i : i*2+2*m_i+2] = p_img[p_i,:] - p[p_i,0,:]

        return out

    def _position_from_fundamental(self, F, points_ref, points_cur):
        """

        Args:
            F (np.array): fundamental matrix
        Returns:

        """
        essential_matrix = np.matmul(np.matmul(self.camera_matrix.T, F), self.camera_matrix)
        R1, R2, T = cv2.decomposeEssentialMat(essential_matrix)
        rt_hyptheses = [
            np.hstack((R1, T)),
            np.hstack((R1, -T)),
            np.hstack((R2, T)),
            np.hstack((R2, -T)),
        ]
        score_max = 0
        i_max = 0
        points3d = None
        rt_matrix = None
        in_fronts = np.zeros((points_ref.shape[0]), dtype=np.bool)
        for i, rt in enumerate(rt_hyptheses):
            # TODO: add calculation of projection based on reference camera position
            points_homo = cv2.triangulatePoints(self.base_projection, np.matmul(self.camera_matrix, rt), points_ref.T, points_cur.T)
            points = cv2.convertPointsFromHomogeneous(points_homo.T)
            points = np.reshape(points, (points.shape[0], points.shape[2]))

            score = 0
            # Could be vectorize
            good_points = np.zeros((points.shape[0]), dtype=np.bool)
            for i, point in enumerate(points):
                point2 = np.matmul(rt[:3,:3], point) + rt[:,3]
                if point[2] > 0 and point2[2] > 0:
                    good_points[i] = True
                    score += 1
            if score >= score_max:
                score_max = score
                i_max = i
                points3d = points[good_points]
                rt_matrix = rt
                in_fronts = good_points

        return points3d, rt_matrix, in_fronts

def find_fundamental_matrix(points_ref, points_cur):
    """

    Args:

    Returns:

    """
    fundamental, mask = cv2.findFundamentalMat(points_ref, points_cur, cv2.FM_RANSAC)
    mask = mask.astype('bool')

    # TODO: add calculation of score
    score = 1
    return fundamental, score, mask

def find_homography(points_ref, points_cur):
    """
    """
    # TODO: add calculation of homography
    homography = None
    score = 0
    mask = np.ones((points_cur.shape[0]), dtype=np.bool)
    return homography, score, mask
