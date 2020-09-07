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
    def __init__(self, camera_matrix, img_shape, min_matches=50,
                feature_detector='orb', initial_pose=None):
        self.camera_matrix = camera_matrix
        self.img_shape = img_shape
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
        self.min_matches = min_matches

        self.graph = CovisibilityGraph(self.matcher, self.match_treshold, self.min_matches)
        self.map = Map()

        self.initialize = True
        self.features = []
        self.features_desc = []

        self.base_projection = np.matmul(self.camera_matrix, np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]))
        self.projection = np.vstack((np.copy(self.base_projection), [0, 0, 0, 1]))
        self.motion_model = np.eye(4)

        self._last_key_frame = KeyFrame([],[], self.camera_matrix)
        self._last_key_frame.update_position_and_rotation(np.eye(4))
        self._tracking_successful = False
        self._global_relocalization_i = 0
        self._key_frame_insertion_i = 0

    def update(self, img):
        """

        Args:
            img (cv2.):
        """
        keypoints, descriptors = self.feature_detector.detectAndCompute(img, None)
        new_key_frame = KeyFrame(keypoints, descriptors, self.camera_matrix)

        img2 = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0), flags=0)
        cv2.imshow('frame',img2)
        cv2.waitKey(1)

        print(len(self.graph))
        input("KeyFrame calculated. Press Enter to continue...")

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

        print(self._last_key_frame.camera_extrinsic)

        # if len(self.graph.keyframes) >= 3:
        #     self.initialize = False


    def _tracking(self, new_key_frame):
        """
        """
        self.tracking_successful = True
        if self.tracking_successful:
            new_key_frame.update_position_and_rotation(np.matmul(self.motion_model, self._last_key_frame.camera_extrinsic))
            print(new_key_frame.camera_extrinsic)
            # UPDATE without motion model
            # new_key_frame.update_position_and_rotation(self._last_key_frame.camera_extrinsic)
            # zero = np.array([
            #     [1, 0, 0, 0],
            #     [0, 1, 0, 0],
            #     [0, 0, 1, 0]], dtype=np.float)
            # new_key_frame.update_position_and_rotation(zero)
            matches = self.matcher.match(self._last_key_frame.descriptors_in_map, new_key_frame.descriptors)
            matches = [m for m in matches if m.distance < self.match_treshold]
            ref_indices = np.array([m.queryIdx for m in matches])
            cur_indices = np.array([m.trainIdx for m in matches])
            map_indices = self.graph[-1].map_indices[ref_indices]
            new_key_frame.map_indices[cur_indices] = map_indices

            points3d = self.map.points3d[map_indices]
            pred_points2d, _ = cv2.projectPoints(points3d,
                                              new_key_frame.camera_rotation_vector,
                                              new_key_frame.camera_position,
                                              self.camera_matrix, None)
            pred_error = np.sum(np.abs(new_key_frame.points2d[cur_indices] - pred_points2d))
            #print('pred_error', pred_error)

            # TODO: Determine value when tracking is lost
            if pred_error > 1e9:
                self._tracking_successful = False
            else:
                self.graph.add_node_candidat(new_key_frame)
                #print(self.graph.candidat_edges)
                self._motion_only_bundle_adjustment()
                self._last_key_frame = self.graph.candidat
                self._update_motion_model(self.graph.candidat)
                self._global_relocalization_i += 1

        else:
            # global relocalization
            self.graph.add_node_candidat(new_key_frame)
            self._motion_only_bundle_adjustment()
            self._global_relocalization_i = 0

        # get all map points which could be visible in frame
        # optimize pose with all of that points

        # decide to add new keyframe or not
        # NOTE changed i from 20 to 10 for development
        add_keyframe = True
        if self._global_relocalization_i < 5:
            add_keyframe = False
        elif self._key_frame_insertion_i < 5: # or local mapping is stopped
            add_keyframe = False
        # elif # less than 50 map points in the keyframe:
        #     add_keyframe = False
        # elif # more than 90% of the same points in ref keyframe
        #     add_keyframe = False

        if add_keyframe:
            self.graph.add_candidat_to_graph()
            self._key_frame_insertion_i = 0
            # TODO: Local mapping should be in separate thread
            self._local_mapping()
        else:
            # NOTE add new node in every step for now
            #self.graph.add_candidat_to_graph()

            self.graph.remove_candidat()
            self._key_frame_insertion_i += 1


    def _local_mapping(self):
        """
        """
        # TODO: Calculate bag of words to speed up searching graph points
        # TODO: Recent map points culling

        # New map point creation
        added_frame = self.graph[-1]
        added_frame_desc, added_frame_ind = added_frame.unmatched_descriptors
        added_indices = None
        connected_indices = None
        max_matches = 0
        max_i = 0
        for i, edge in enumerate(self.graph.connected_frames(added_frame)):
            connected_frame_desc, connected_frame_ind = edge.node.unmatched_descriptors
            matches = self.matcher.match(added_frame_desc, connected_frame_desc)
            matches = [m for m in matches if m.distance < self.match_treshold]
            if len(matches) > max_matches:
                added_indices = added_frame_ind[[m.queryIdx for m in matches]]
                connected_indices = connected_frame_ind[[m.trainIdx for m in matches]]
                max_matches = len(matches)
                max_i = i

        # trainagulate new points
        points_homo = cv2.triangulatePoints(added_frame.camera_projection,
                                            self.graph.connected_frames(added_frame)[max_i].node.camera_projection,
                                            added_frame.points2d[added_indices].T,
                                            self.graph.connected_frames(added_frame)[max_i].node.points2d[connected_indices].T)
        points3d = cv2.convertPointsFromHomogeneous(points_homo.T)
        points3d = np.reshape(points3d, (points3d.shape[0], points3d.shape[2]))
        map_indices = self.map.add_new_points(points3d, added_frame.descriptors[added_indices])
        self.graph[-1].update_indices(map_indices, added_indices)
        self.graph.connected_frames(added_frame)[max_i].node.update_indices(map_indices, connected_indices)

        # Look for added points in other connected frames
        for i, edge in enumerate(self.graph.connected_frames(added_frame)):
            if i != max_i:
                connected_frame_desc, connected_frame_ind = edge.node.unmatched_descriptors
                matches = self.matcher.match(self.map.points_descriptors[map_indices], connected_frame_desc)
                matches = [m for m in matches if m.distance < self.match_treshold]
                m_indices = map_indices[[m.queryIdx for m in matches]]
                f_indicies = connected_frame_ind[[m.trainIdx for m in matches]]
                self.graph.connected_frames(added_frame)[i].node.update_indices(m_indices, f_indicies)

        # Local BA
        self._local_bundle_adjustment(added_frame)
        # Local Keyframe culling


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
            self._last_key_frame = new_key_frame
            return

        ref_indices = np.array([m.queryIdx for m in matches])
        cur_indices = np.array([m.trainIdx for m in matches])

        points_ref = self.graph[0].points2d[ref_indices]
        points_cur = new_key_frame.points2d[cur_indices]
        description = self.graph[0].descriptors[ref_indices]

        # TODO: Calculate in parallel threads
        F, F_score, F_inliers = find_fundamental_matrix(points_ref, points_cur)
        H, H_score, H_inliers = find_homography(points_ref, points_cur)

        rate = H_score / (H_score+F_score)

        if rate <= 0.45:
            points_ref = points_ref[F_inliers[:,0]]
            points_cur = points_cur[F_inliers[:,0]]
            description = description[F_inliers[:,0]]
            ref_indices = ref_indices[F_inliers[:,0]]
            cur_indices = cur_indices[F_inliers[:,0]]
            points3d, rt_matrix, inliers = self._position_from_fundamental(F, points_ref, points_cur)
            m_indices = self.map.add_points(points3d, description[inliers], self.matcher, self.match_treshold)
            self.graph[0].update_indices(m_indices, ref_indices[inliers])
            new_key_frame.update_indices(m_indices, cur_indices[inliers])
            new_key_frame.update_position_and_rotation(rt_matrix)
        else:
            # TODO: add calculation position from homography
            pass

        self.graph.add_node(new_key_frame)
        self._update_motion_model(new_key_frame)
        self.bundle_adjustment()
        self._last_key_frame = self.graph[-1]
        self.initialize = False

    def _update_motion_model(self, frame):
        """
        """
        lastTwc = self._calculate_world2camera_projection(self._last_key_frame.camera_extrinsic)
        self.motion_model = np.matmul(frame.camera_extrinsic, lastTwc)

    def _calculate_world2camera_projection(self, projection):
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

    def _local_bundle_adjustment(self, keyframe):
        """
        """
        keyframes = self.graph
        map_indices = set()
        for kf in keyframes:
            map_indices.update(kf.indices_in_map)
        map_indices = np.array(list(map_indices))
        # TODO: add also keyframes not connected, but from which its points are
        # visibile. Those keyframes should remain as fixed

        n = 6*len(keyframes) + 3*len(map_indices)
        x0 = np.empty((n))
        for i, keyframe in enumerate(keyframes):
            j = i*6
            x0[j:j+3] = keyframe.camera_rotation_vector
            x0[j+3:j+6] = keyframe.camera_position[:,0]
        x0[-3*len(map_indices):] = self.map.points3d[map_indices].flatten()

        res = least_squares(self._local_ba_projection, x0,
                            jac=self._calculate_local_jacobian, tr_solver='lsmr',
                            jac_sparsity=self._local_jacobian_sparsity(keyframes, map_indices),
                            verbose=0,
                            args=[keyframes, map_indices]
                            )
        #print("---new keyframes matricies---")
        for i, keyframe in enumerate(keyframes):
            j = i*6
            keyframe.update_rotation_from_rotation_vector(res.x[j:j+3])
            keyframe.update_position(res.x[j+3:j+6])
            #print(keyframe.camera_extrinsic)

        self.map.update_local_points(res.x[-3*len(map_indices):].reshape((len(map_indices),3)), map_indices)



    def _motion_only_bundle_adjustment(self):
        """
        """
        keyframes = self.graph.get_local_keyframes_and_neigbours()
        # print('---keyframes matricies before BA---')
        # for keyframe in keyframes:
        #     print(keyframe.camera_extrinsic)
        map_indices = set()
        for kf in keyframes:
            map_indices.update(kf.indices_in_map)
        map_indices = np.array(list(map_indices))

        # check if points coud be visible in current frame
        points_cur, _ = cv2.projectPoints(self.map.points3d[map_indices],
                                       self.graph.candidat.camera_rotation_vector,
                                       self.graph.candidat.camera_position,
                                       self.camera_matrix, None)
        in_image = np.ones((len(map_indices)), dtype=np.bool)
        in_image[points_cur[:,0,0] < 0] = False
        in_image[points_cur[:,0,0] > self.img_shape[0]] = False
        in_image[points_cur[:,0,1] < 0] = False
        in_image[points_cur[:,0,1] > self.img_shape[1]] = False
        map_indices = map_indices[in_image]

        # viewing_versors = self.map.points_viewing_versor[map_indices]
        # camera_direction = self.graph.candidat.camera_direction
        # point_angle = np.array([np.dot(camera_direction, point_versor) for point_versor in viewing_versors])
        # correct_angle = point_angle > np.cos(np.deg2rad(60))
        # map_indices = map_indices[correct_angle]
        # print(map_indices)

        # TODO: Add discarding points due distance to camera center

        matches = self.matcher.match(self.map.points_descriptors[map_indices], self.graph.candidat.descriptors)
        matches = [m for m in matches if m.distance < self.match_treshold]
        ref_indices = np.array([m.queryIdx for m in matches])
        candidat_indices = np.array([m.trainIdx for m in matches])
        self.graph.candidat.update_indices(ref_indices, candidat_indices)

        n = 6*len(keyframes) #+ 3*len(map_indices)
        x0 = np.empty((n))
        for i, keyframe in enumerate(keyframes):
            j = i*6
            x0[j:j+3] = keyframe.camera_rotation_vector
            x0[j+3:j+6] = keyframe.camera_position[:,0]
        #x0[-3*len(map_indices):] = self.map.points3d[map_indices].flatten()
        res = least_squares(self._mo_ba_projection, x0,
                            jac=self._calculate_mo_jacobian, tr_solver='lsmr',
                            jac_sparsity=self._mo_jacobian_sparsity(keyframes, map_indices),
                            verbose=0,
                            args=[keyframes, map_indices]
                            )
        #print("---new keyframes matricies---")
        for i, keyframe in enumerate(keyframes):
            j = i*6
            keyframe.update_rotation_from_rotation_vector(res.x[j:j+3])
            keyframe.update_position(res.x[j+3:j+6])
            #print(keyframe.camera_extrinsic)

    def _jacobian_sparsity(self):
        """
        """
        Nf = len(self.graph.keyframes)
        Nm = len(self.map)

        sparsity = np.zeros((2*Nm*Nf, 6*Nf+3*Nm), dtype=np.bool)
        for i, frame in enumerate(self.graph.keyframes):
            p_img, map_indices = frame.points_in_map
            for p_i, m_i in enumerate(map_indices):
                # camera jacobian
                sparsity[m_i*Nf+i:m_i*Nf+i+1, i:i+6] = True
                # point jacobians
                sparsity[m_i*Nf+i, 6+m_i*3:9+m_i*3] = True
                sparsity[m_i*Nf+i+1, 6+m_i*3:9+m_i*3] = True

        return sparsity

    def _local_jacobian_sparsity(self, keyframes, map_indices):
        """
        """
        Nf = len(keyframes)
        Nm = len(map_indices)

        local_indices = {}
        for i, map_i in enumerate(map_indices):
            local_indices[map_i] = i

        sparsity = np.zeros((2*Nm*Nf, 6*Nf+3*Nm), dtype=np.bool)
        for i, frame in enumerate(keyframes):
            p_img, frame_map_indices = frame.chosen_points_in_map(map_indices)
            for p_i, m_i in enumerate(map_indices):
                j = local_indices[m_i]*Nf+i
                k = 6+local_indices[m_i]*3
                # camera jacobian
                sparsity[j:j+1, i:i+6] = True
                # point jacobians
                sparsity[j, k:k+3] = True
                sparsity[j+1, k:k+3] = True

        return sparsity

    def _mo_jacobian_sparsity(self, keyframes, map_indices):
        """
        """
        Nf = len(keyframes)
        Nm = len(map_indices)

        local_indices = {}
        for i, map_i in enumerate(map_indices):
            local_indices[map_i] = i

        sparsity = np.zeros((2*Nm*Nf, 6*Nf), dtype=np.bool)
        for i, frame in enumerate(keyframes):
            p_img, frame_map_indices = frame.chosen_points_in_map(map_indices)
            for p_i, m_i in enumerate(map_indices):
                j = local_indices[m_i]*Nf+i
                sparsity[j:j+1, i:i+6] = True

        return sparsity

    def _calculate_jacobian(self, x):
        """
        """
        Nf = len(self.graph.keyframes)
        Nm = len(self.map)

        points3d = x[-3*Nm:].reshape((Nm,3))

        jacobian = np.zeros((2*Nm*Nf, 6*Nf+3*Nm))
        for i, frame in enumerate(self.graph.keyframes):
            p_img, map_indices = frame.points_in_map
            j = i*6
            rvec = x[j:j+3]
            pos = x[j+3:j+6]

            p, jac = cv2.projectPoints(points3d[map_indices], rvec, pos, self.camera_matrix, None)

            r = Rotation.from_rotvec(rvec).as_matrix()
            fx = self.camera_matrix[0,0]
            fy = self.camera_matrix[1,1]
            px = self.camera_matrix[0,2]
            py = self.camera_matrix[1,2]
            du_dp = np.array([fx*r[0,0]+px*r[2,0], fx*r[0,1]+px*r[2,1], fx*r[0,2]+px*r[2,2]])
            dv_dp = np.array([fy*r[1,0]+py*r[2,0], fy*r[1,1]+py*r[2,1], fy*r[1,2]+py*r[2,2]])
            dw_dp = r[2,:]

            for p_i, m_i in enumerate(map_indices):
                # camera jacobian
                jacobian[m_i*Nf+i:m_i*Nf+i+1, i:i+6] = jac[p_i:p_i+1,:6]
                # point jacobians
                jacobian[m_i*Nf+i, 6+m_i*3:9+m_i*3] = du_dp - p[p_i,0,0]*dw_dp
                jacobian[m_i*Nf+i+1, 6+m_i*3:9+m_i*3] = dv_dp - p[p_i,0,1]*dw_dp

        return jacobian

    def _calculate_local_jacobian(self, x, keyframes, map_indices):
        """
        """
        Nf = len(keyframes)
        Nm = len(map_indices)

        local_indices = {}
        for i, map_i in enumerate(map_indices):
            local_indices[map_i] = i

        jacobian = np.zeros((2*Nm*Nf, 6*Nf+3*Nm))
        for i, frame in enumerate(keyframes):
            p_img, frame_map_indices = frame.chosen_points_in_map(map_indices)
            if len(frame_map_indices) == 0:
                continue
            j = i*6
            rvec = x[j:j+3]
            pos = x[j+3:j+6]

            p, jac = cv2.projectPoints(self.map.points3d[frame_map_indices], rvec, pos, self.camera_matrix, None)

            r = Rotation.from_rotvec(rvec).as_matrix()
            fx = self.camera_matrix[0,0]
            fy = self.camera_matrix[1,1]
            px = self.camera_matrix[0,2]
            py = self.camera_matrix[1,2]
            du_dp = np.array([fx*r[0,0]+px*r[2,0], fx*r[0,1]+px*r[2,1], fx*r[0,2]+px*r[2,2]])
            dv_dp = np.array([fy*r[1,0]+py*r[2,0], fy*r[1,1]+py*r[2,1], fy*r[1,2]+py*r[2,2]])
            dw_dp = r[2,:]

            for p_i, m_i in enumerate(frame_map_indices):
                k = local_indices[m_i]*Nf+i
                g = 6+local_indices[m_i]*3
                jacobian[k:k+1, i:i+6] = jac[p_i:p_i+1,:6]
                # point jacobians
                jacobian[k:k+1, g:g+3] = du_dp - p[p_i,0,0]*dw_dp
                jacobian[k:k+1, g:g+3] = dv_dp - p[p_i,0,1]*dw_dp

        return jacobian

    def _calculate_mo_jacobian(self, x, keyframes, map_indices):
        """
        """
        Nf = len(keyframes)
        Nm = len(map_indices)

        local_indices = {}
        for i, map_i in enumerate(map_indices):
            local_indices[map_i] = i

        jacobian = np.zeros((2*Nm*Nf, 6*Nf))
        for i, frame in enumerate(keyframes):
            p_img, frame_map_indices = frame.chosen_points_in_map(map_indices)
            if len(frame_map_indices) == 0:
                continue
            j = i*6
            rvec = x[j:j+3]
            pos = x[j+3:j+6]

            p, jac = cv2.projectPoints(self.map.points3d[frame_map_indices], rvec, pos, self.camera_matrix, None)

            for p_i, m_i in enumerate(frame_map_indices):
                k = local_indices[m_i]*Nf+i*6
                jacobian[k:k+1, i:i+6] = jac[p_i:p_i+1,:6]

        return jacobian

    def _ba_projection(self, x):
        """
        """
        Nf = len(self.graph.keyframes)
        Nm = len(self.map)
        Np = 2*Nm*Nf

        points3d = x[-3*Nm:].reshape((Nm,3))

        out = np.zeros((Np))
        for i, frame in enumerate(self.graph.keyframes):
            p_img, map_indices = frame.points_in_map
            j = i*6
            rvec = x[j:j+3]
            pos = x[j+3:j+6]

            p, jacobian = cv2.projectPoints(points3d[map_indices], rvec, pos, self.camera_matrix, None)

            for p_i, m_i in enumerate(map_indices):
                out[i*2*Nf+2*m_i : i*2*Nf+2*m_i+2] = p_img[p_i,:] - p[p_i,0,:]

        return out

    def _local_ba_projection(self, x, keyframes, map_indices):
        """Local bundle adjustment projection function
        """
        Nf = len(keyframes)
        Nm = len(map_indices)
        Np = 2*Nm*Nf

        local_indices = {}
        for i, map_i in enumerate(map_indices):
            local_indices[map_i] = i

        out = np.zeros((Np))
        for i, frame in enumerate(keyframes):
            p_img, frame_map_indices = frame.chosen_points_in_map(map_indices)
            if len(frame_map_indices) == 0:
                continue
            j = i*6
            rvec = x[j:j+3]
            pos = x[j+3:j+6]

            p, jacobian = cv2.projectPoints(self.map.points3d[frame_map_indices], rvec, pos, self.camera_matrix, None)

            for p_i, m_i in enumerate(frame_map_indices):
                k = local_indices[m_i]*Nf*2 + i*2
                out[k:k+2] = p_img[p_i,:] - p[p_i,0,:]

        return out

    def _mo_ba_projection(self, x, keyframes, map_indices):
        """Motion-only bundle adjustment projection function
        """
        #print('x: ', x)
        Nf = len(keyframes)
        Nm = len(map_indices)
        Np = 2*Nm*Nf

        local_indices = {}
        for i, map_i in enumerate(map_indices):
            local_indices[map_i] = i

        #print(keyframes)
        #print(local_indices)

        out = np.zeros((Np))
        for i, frame in enumerate(keyframes):
            #print('map', map_indices)
            p_img, frame_map_indices = frame.chosen_points_in_map(map_indices)
            #print(frame_map_indices)
            if len(frame_map_indices) == 0:
                continue
            j = i*6
            rvec = x[j:j+3]
            pos = x[j+3:j+6]

            #print('frame', frame_map_indices)
            p, jacobian = cv2.projectPoints(self.map.points3d[frame_map_indices], rvec, pos, self.camera_matrix, None)

            for p_i, m_i in enumerate(frame_map_indices):
                #print('i', i)
                k = local_indices[m_i]*Nf*2 + i*2
                #print(k)
                out[k:k+2] = p_img[p_i,:] - p[p_i,0,:]

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
