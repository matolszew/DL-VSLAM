import numpy as np
import cv2
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import multiprocessing

from map import Map
from key_frame import KeyFrame
from covisibility_graph import CovisibilityGraph

class ORBSLAM:
    """ORB-SLAM class

    Based on "ORB-SLAM: A Versatile and Accurate Monocular SLAM System",
    Raúl Mur-Artal, J. M. M. Montiel, Juan D. Tardós

    Args:
        feature_detector (string):
    """
    def __init__(self, camera_matrix, min_matches=100,
                feature_detector='orb', initial_pose=None):
        self.camera_matrix = camera_matrix
        if feature_detector=='orb':
            self.feature_detector = cv2.ORB_create(nfeatures=500)
        elif feature_detector=='mser':
            self.feature_detector = cv2.MSER_create()
        elif feature_detector=='fast':
            self.feature_detector = cv2.FastFeatureDetector_create()
        elif feature_detector=='surf':
            self.feature_detector = cv2.SURF(400)
        else:
            raise ValueError('Not known feature detector')

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.match_treshold = 64

        self.graph = CovisibilityGraph(self.matcher, self.match_treshold)
        self.map = Map()

        self.initialize = True
        self.features = []
        self.features_desc = []

        self.base_projection = np.matmul(self.camera_matrix, np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ]))
        self.projection = np.copy(self.base_projection)
        self.motion_model = self._calculate_velocity_model(self.projection)

        self._last_key_frame = None
        self._tracking_successful = False
        self._global_relocalization_i = 0
        self._key_frame_insertion_i = 0

    def update(self, img):
        """

        Args:
            img (cv2.):
        """
        keypoints, descriptors = self.feature_detector.detectAndCompute(img, None)
        new_key_frame = KeyFrame(keypoints, descriptors)

        img2 = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0), flags=0)
        cv2.imshow('frame',img2)
        cv2.waitKey(1)

        if len(self.graph) == 0:
            # TODO: add possibility to initialize camera rotation and position
            new_key_frame.update_rotation_from_quat([0,0,0,1])
            new_key_frame.update_position(np.zeros((3,1)))
            self.graph.add_node(new_key_frame)
            self._last_key_frame = new_key_frame
            return

        if self.initialize:
            self._initialize(new_key_frame)
        else:
            self._tracking(new_key_frame)

        if len(self.keyframes) >= 5:
            self.initialize = False


    def _tracking(self, new_key_frame):
        """
        """
        if self.tracking_successful:
            new_key_frame.update_position_and_rotation(np.matmul(self.motion_model, self.projection))

            matches = self.matcher.match(new_key_frame.descriptors, self._last_key_frame.descriptors_in_map)
            matches = [m for m in matches if m.distance < self.match_treshold]
            ref_indicies = np.array([m.queryIdx for m in matches])
            cur_indicies = np.array([m.trainIdx for m in matches])
            map_indicies = self.graph[-1].map_indicies[ref_indicies]
            new_key_frame.map_indicies[cur_indicies] = map_indicies

            points3d = self.map.points3d[map_indicies]
            pred_points2d = cv2.projectPoints(points3d,
                                              new_key_frame.camera_rotation_vector,
                                              new_key_frame.camera_position,
                                              self.camera_matrix, None)
            pred_error = np.sum(np.abs(new_key_frame.points2d[cur_indicies] - pred_points2d))

            # TODO: Determine value when tracking is lost
            if pred_error > 1e6:
                self._tracking_successful = False
            else:
                self._motion_only_bundle_adjustment()
                self._global_relocalization_i += 1

        else:
            # global relocalization
            self._motion_only_bundle_adjustment()
            self._global_relocalization_i = 0

        # get all map points which could be visible in frame
        # optimize pose with all of that points

        # decide to add new keyframe or not
        add_keyframe = True
        if self._global_relocalization_i < 20:
            add_keyframe = False
        elif self._key_frame_insertion_i < 20: # or local mapping is stopped
            add_keyframe = False
        elif # less than 50 map points in the keyframe:
            add_keyframe = False
        elif # more than 90% of the same points in ref keyframe
            add_keyframe = False

        if add_keyframe:
            self.graph.add_node(key_frame)
            self._key_frame_insertion_i = 0
        else:
            self._key_frame_insertion_i += 1


    def _local_mapping(self):
        """
        """
        pass

    def _loop_closing(self):
        """
        """
        pass


    def _initialize(self, new_key_frame):
        """
        """
        matches = self.matcher.match(self.graph[0].descriptors, new_key_frame.descriptors)
        matches = [m for m in matches if m.distance < self.match_treshold]

        if len(matches) < self.min_matches:
            self.graph.reset()
            self.graph.add_node(new_key_frame)
            return

        ref_indicies = np.array([m.queryIdx for m in matches])
        cur_indicies = np.array([m.trainIdx for m in matches])

        points_ref = sself.graph[0].points2d[ref_indicies]
        points_cur = new_key_frame.points2d[cur_indicies]
        description = self.graph[0].descriptors[ref_indicies]

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
            self.graph[0].update_indicies(m_indices, ref_indicies[inliers])
            new_key_frame.update_indicies(m_indices, cur_indicies[inliers])
            new_key_frame.update_position_and_rotation(rt_matrix)
        else:
            # TODO: add calculation position from homography
            pass

        self.graph.add_node(new_key_frame)
        self.bundle_adjustment()
        self._last_key_frame = self.graph[-1]

    def _calculate_velocity_model(self, projection):
        """
        """
        projection = np.eye(4)
        R = projection[:3,:3].T
        T = -np.matmul(R, projection[:3,3])
        projection[:3,:3] = R
        projection[:3,3] = T

        return projection

    def bundle_adjustment(self):
        """
        """
        n = 6*len(self.graph) + 3*len(self.map)
        x0 = np.empty((n))
        for i, keyframe in enumerate(self.graph):
            j = i*6
            x0[j:j+3] = keyframe.camera_rotation_vector
            x0[j+3:j+6] = keyframe.camera_position[:,0]
        x0[-3*len(self.map):] = self.map.points3d.flatten()

        res = least_squares(self._ba_projection, x0,
                            jac=self._calculate_jacobian, tr_solver='lsmr',
                            jac_sparsity=self._jacobian_sparsity(),
                            verbose=0)
        for i, keyframe in enumerate(self.graph):
            j = i*6
            keyframe.update_rotation_from_rotation_vector(res.x[j:j+3])
            keyframe.update_position(res.x[j+3:j+6])
            #print(res.x[j+3:j+6])
        self.map.update_points(res.x[-3*len(self.map):].reshape((len(self.map),3)))
        input("Press Enter to continue...")

    def _motion_only_bundle_adjustment(self):
        """
        """

    def _jacobian_sparsity(self):
        """
        """
        Nf = len(self.keyframes)
        Nm = len(self.map)

        sparsity = np.zeros((2*Nm*Nf, 6*Nf+Nm), dtype=np.bool)
        for i, frame in enumerate(self.keyframes):
            p_img, map_indicies = frame.points_on_img
            for p_i, m_i in enumerate(map_indicies):
                # camera jacobian
                sparsity[m_i*Nf+i:m_i*Nf+i+1, i:i+6] = True
                # point jacobians
                sparsity[m_i*Nf+i, 6+m_i*3:9+m_i*3] = True
                sparsity[m_i*Nf+i+1, 6+m_i*3:9+m_i*3] = True

        return sparsity

    def _calculate_jacobian(self, x):
        """
        """
        Nf = len(self.keyframes)
        Nm = len(self.map)

        points3d = x[-3*Nm:].reshape((Nm,3))

        jacobian = np.zeros((2*Nm*Nf, 6*Nf+3*Nm))
        for i, frame in enumerate(self.keyframes):
            p_img, map_indicies = frame.points_on_img
            j = i*6
            rvec = x[j:j+3]
            pos = x[j+3:j+6]

            p, j = cv2.projectPoints(points3d[map_indicies], rvec, pos, self.camera_matrix, None)

            r = Rotation.from_rotvec(rvec).as_matrix()
            fx = self.camera_matrix[0,0]
            fy = self.camera_matrix[1,1]
            px = self.camera_matrix[0,2]
            py = self.camera_matrix[1,2]
            du_dp = np.array([fx*r[0,0]+px*r[2,0], fx*r[0,1]+px*r[2,1], fx*r[0,2]+px*r[2,2]])
            dv_dp = np.array([fy*r[1,0]+py*r[2,0], fy*r[1,1]+py*r[2,1], fy*r[1,2]+py*r[2,2]])
            dw_dp = r[2,:]

            for p_i, m_i in enumerate(map_indicies):
                # camera jacobian
                jacobian[m_i*Nf+i:m_i*Nf+i+1, i:i+6] = j[p_i:p_i+1,:6]
                # point jacobians
                jacobian[m_i*Nf+i, 6+m_i*3:9+m_i*3] = du_dp - p[p_i,0,0]*dw_dp
                jacobian[m_i*Nf+i+1, 6+m_i*3:9+m_i*3] = dv_dp - p[p_i,0,1]*dw_dp

        return jacobian

    def _ba_projection(self, x):
        """
        """
        Nf = len(self.keyframes)
        Nm = len(self.map)
        Np = 2*Nm*Nf

        points3d = x[-3*Nm:].reshape((Nm,3))

        out = np.zeros((Np))
        for i, frame in enumerate(self.keyframes):
            p_img, map_indicies = frame.points_on_img
            j = i*6
            rvec = x[j:j+3]
            pos = x[j+3:j+6]

            p, jacobian = cv2.projectPoints(points3d[map_indicies], rvec, pos, self.camera_matrix, None)

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
