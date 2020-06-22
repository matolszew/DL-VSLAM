import numpy as np
import cv2
import scipy.spatial.transform as transform
import matplotlib.pyplot as plt

from kalman_filter import EKF
#from keyframe import Keyframe

from time import time

class SLAM:
    """First iteration of SLAM class

    Args:
        ...
    """
    def __init__(self, dt, width, height, calibration_matrix, distortion_coefficients=None,
                 initial_pose=None, match_treshold=64):
        self.dt = dt
        self.width = width
        self.height = height
        self.calibration_matrix = calibration_matrix
        self.match_treshold = match_treshold
        self.state_estimator = EKF(initial_pose, calibration_matrix, distortion_coefficients, dt)

        self.feature_detector = cv2.ORB_create(nfeatures=500)
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
        self.projection_ref = np.vstack((self.base_projection, [0, 0, 0, 1]))

        if initial_pose is None:
            self.proj = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0]
            ])
        else:
            self.proj = initial_pose

        self.old_R = np.eye(3)
        self.old_T = np.zeros((3,1))

        self.rt_ref = np.eye(4)
        self.ref_img = None

        self.position = np.zeros((2200, 3))
        self.points_i = 0
        self.pos_i = 0

        self.z_versor = np.array([[0, 0, 1]]).T

        self.keyframe_collection = []

        self.ini = True
        self.counter = 0
        self.counter_reset = 10

    def run3(self, img):
        """Send only image points to EKF
        """
        print(self.counter)
        self.counter += 1
        if self.ref_img is None:
            self.ref_img = img
            self.kp_ref = self.feature_detector.detect(img, None)
            self.kp_ref, self.kp_ref_des = self.feature_detector.compute(img, self.kp_ref)
            return [0, 0, 0], [0, 0, 0]

        if self.counter == self.counter_reset:
            self.ini = True

        kp_cur = self.feature_detector.detect(img, None)
        kp_cur, kp_cur_des = self.feature_detector.compute(img, kp_cur)

        print('Detected points: ', len(kp_cur_des))

        if self.ini:
            matches = self.bf.match(self.kp_ref_des, kp_cur_des)
            print('Matches in ref and cur: ', len(kp_cur_des))
            matches_bt = [m for m in matches if m.distance < self.match_treshold]

            ref_indicies = [m.queryIdx for m in matches_bt]
            cur_indicies = [m.trainIdx for m in matches_bt]

            kp_ref_pt = np.array([self.kp_ref[i].pt for i in ref_indicies])
            kp_cur_pt = np.array([kp_cur[i].pt for i in cur_indicies])
            description = np.array([kp_cur_des[i] for i in cur_indicies])

            print('Matches below treshold: ', np.shape(description)[0])
        else:
            kp_cur_pt = np.array([kp.pt for kp in kp_cur])
            description = kp_cur_des

        if self.ini:
            fundamental, mask = cv2.findFundamentalMat(kp_ref_pt, kp_cur_pt, cv2.FM_RANSAC)
            mask = mask.astype('bool')
            ref_inliers = kp_ref_pt[mask[:,0],:]
            cur_inliers = kp_cur_pt[mask[:,0],:]
            description = description[mask[:,0],:]

            print('Inliers for fundamental: ', np.shape(description)[0])

            self.kp_ref = kp_cur
            self.kp_ref_des = kp_cur_des

            essential_matrix = np.matmul(np.matmul(self.calibration_matrix.T, fundamental), self.calibration_matrix)
            R1, R2, T = cv2.decomposeEssentialMat(essential_matrix)
            rt_hyptheses = [
                np.hstack((R1, T)),
                np.hstack((R1, -T)),
                np.hstack((R2, T)),
                np.hstack((R2, -T)),
            ]
            score_max = 0
            i_max = 0
            points_max = None
            in_fronts = np.zeros((description.shape[0]), dtype=np.bool)
            # print('-------------------------')
            for i, rt in enumerate(rt_hyptheses):
                # print('h ', i)
                # print(rt)
                # print(np.matmul(self.calibration_matrix, rt))
                points_homo = cv2.triangulatePoints(self.base_projection, np.matmul(self.calibration_matrix, rt), ref_inliers.T, cur_inliers.T)
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
                    points_max = points[good_points]
                    in_fronts = good_points
                # print('score: ', score)

            ref_inliers = ref_inliers[in_fronts]
            cur_inliers = cur_inliers[in_fronts]
            description = description[in_fronts]

            print('In front: ', np.shape(description)[0])

            # in_front = points_max[:,2]>=0
            # points_max = points_max[in_front]
            # ref_inliers = ref_inliers[in_front]
            # cur_inliers = cur_inliers[in_front]
            # description = description[in_front]
            #
            not_far = points_max[:,2]<=1e1
            points_max = points_max[not_far]
            ref_inliers = ref_inliers[not_far]
            cur_inliers = cur_inliers[not_far]
            description = description[not_far]

            print('Not far: ', np.shape(description)[0])
            self.ini = False
        else:
            points_max = np.zeros((0,3))
            cur_inliers = kp_cur_pt

        if self.state_estimator.points_number > 0 and len(description) > 0:
            point_angle = np.array([np.dot(self.state_estimator.camera_direction, point_versor) for point_versor in self.state_estimator.points_viewing_versor])
            correct_angle = point_angle > np.cos(np.deg2rad(60))
            print('Correct angle', np.sum(correct_angle))
            camera_point_direction = (self.state_estimator.points[correct_angle]-self.state_estimator.position.T) \
                / np.expand_dims(np.linalg.norm(self.state_estimator.points[correct_angle]-self.state_estimator.position.T, axis=1), axis=1)
            camera_point_angle = np.array([np.dot(self.state_estimator.camera_direction, cp_dir) for cp_dir in camera_point_direction])
            in_front_of_camera = camera_point_angle > 0
            print('In front points in map', np.sum(in_front_of_camera))
            correct_angle_indicies = np.argwhere(correct_angle)[:,0]
            visible_points_indicies = correct_angle_indicies[in_front_of_camera]
            #print('visible: ', visible_points_indicies)

            matches = self.bf.match(self.state_estimator.points_descriptor[visible_points_indicies], description)
            print('Matches in map', len(matches))
            matches_bt = [m for m in matches if m.distance < self.match_treshold]
            print('Matches below treshold', len(matches_bt))
            old_points_indicies = [visible_points_indicies[m.queryIdx] for m in matches_bt]
            #print('matches: ', old_points_indicies)
            cur_points_old_indicies = [m.trainIdx for m in matches_bt]
            #print('old: ', cur_points_old_indicies)
            if self.counter == self.counter_reset:
                cur_points_new_indicies = [i for i in range(len(cur_inliers)) if i not in cur_points_old_indicies]
            else:
                cur_points_new_indicies = []

            #print('new: ', cur_points_new_indicies)
        else:
            old_points_indicies = []
            cur_points_new_indicies = list(range(len(description)))
            cur_points_old_indicies = []

        if len(description) > 0:
            points_world_coords = np.zeros_like(points_max[cur_points_new_indicies])
            # print(self.state_estimator.rotation_matrix)
            for i, p in enumerate(points_max[cur_points_new_indicies]):
                #print('before: ', p)
                points_world_coords[i] = np.matmul(self.state_estimator.rotation_matrix, p) + self.state_estimator.position[:,0]
                #print('after: ', points_world_coords[i])
            # print(np.mean(points_world_coords, axis=0))
            new_point_indicies = self.state_estimator.initialize_points(points_world_coords, description[cur_points_new_indicies])
            indices = old_points_indicies + new_point_indicies
            if len(cur_points_new_indicies) > 0:
                points2D = np.empty_like(cur_inliers)
                points2D[:len(old_points_indicies)] = cur_inliers[cur_points_old_indicies]
                points2D[-len(cur_points_new_indicies):] = cur_inliers[cur_points_new_indicies]
            else:
                points2D = cur_inliers[cur_points_old_indicies]
            print('points: ', len(indices))
            self.state_estimator.update2(points2D, indices)

        if self.counter == self.counter_reset:
            self.ref_img = img
            self.kp_ref = self.feature_detector.detect(img, None)
            self.kp_ref, self.kp_ref_des = self.feature_detector.compute(img, self.kp_ref)
            self.counter = 0

        img2 = cv2.putText(img, 'pos: ' + str(self.state_estimator.position), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
        img2 = cv2.putText(img2, str(self.state_estimator.rotation), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
        img2 = cv2.putText(img2, 'vel: ' + str(self.state_estimator.velocity), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
        img2 = cv2.putText(img, 'ang vel: ' + str(self.state_estimator.angular_velocity), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
        img2 = cv2.drawKeypoints(img, kp_cur, None, color=(0,255,0), flags=0)
        cv2.imshow('frame',img2)
        cv2.waitKey(1)

        input("Press Enter to continue...")



    def run2(self, img):
        if self.ref_img is None:
            self.ref_img = img
            self.kp_ref = self.feature_detector.detect(img, None)
            self.kp_ref, self.kp_ref_des = self.feature_detector.compute(img, self.kp_ref)
            return [0, 0, 0], [0, 0, 0]

        kp_cur = self.feature_detector.detect(img, None)
        kp_cur, kp_cur_des = self.feature_detector.compute(img, kp_cur)

        matches = self.bf.match(self.kp_ref_des, kp_cur_des)
        matches_bt = [m for m in matches if m.distance < self.match_treshold]

        ref_indicies = [m.queryIdx for m in matches_bt]
        cur_indicies = [m.trainIdx for m in matches_bt]

        kp_ref_pt = np.array([self.kp_ref[i].pt for i in ref_indicies])
        kp_cur_pt = np.array([kp_cur[i].pt for i in cur_indicies])
        description = np.array([kp_cur_des[i] for i in cur_indicies])

        self.kp_ref = kp_cur
        self.kp_ref_des = kp_cur_des

        fundamental, mask = cv2.findFundamentalMat(kp_ref_pt, kp_cur_pt, cv2.FM_RANSAC)
        mask = mask.astype('bool')
        ref_inliers = kp_ref_pt[mask[:,0],:]
        cur_inliers = kp_cur_pt[mask[:,0],:]
        description = description[mask[:,0],:]

        essential_matrix = np.matmul(np.matmul(self.calibration_matrix.T, fundamental), self.calibration_matrix)
        R1, R2, T = cv2.decomposeEssentialMat(essential_matrix)
        rt_hyptheses = [
            np.hstack((R1, T)),
            np.hstack((R1, -T)),
            np.hstack((R2, T)),
            np.hstack((R2, -T)),
        ]
        score_max = 0
        i_max = 0
        points_max = None
        for i, rt in enumerate(rt_hyptheses):
            points_homo = cv2.triangulatePoints(self.base_projection, np.   matmul(self.calibration_matrix, rt), ref_inliers.T, cur_inliers.T)
            points = cv2.convertPointsFromHomogeneous(points_homo.T)
            points = np.reshape(points, (points.shape[0], points.shape[2]))

            score1 = 0
            score2 = 0
            # Could be vectorize
            for point in points:
                point2 = np.matmul(rt[:3,:3], point) + rt[:,3]
                # if point[2] > 0 and point2[2] > 0:
                #     score += 1
                if point[2] > 0:
                    score1 += 1
                if point2[2] > 0:
                    score2 += 1
            # print('i: ', i)
            # print('score 1: ', score1)
            # print('score 2: ', score2)
            # print('------------------')
            if score1 + score2 >= score_max:
                score_max = score1 + score2
                i_max = i
                points_max = points
        # print('##########################')

        in_front = points_max[:,2]>=0
        points_max = points_max[in_front]
        ref_inliers = ref_inliers[in_front]
        cur_inliers = cur_inliers[in_front]
        description = description[in_front]

        not_far = points_max[:,2]<=100
        points_max = points_max[not_far]
        ref_inliers = ref_inliers[not_far]
        cur_inliers = cur_inliers[not_far]
        description = description[not_far]

        # TODO: Vectorize
        points_world_coords = np.zeros_like(points_max)
        for i, p in enumerate(points_max):
            points_world_coords[i] = np.matmul(self.state_estimator.rotation_matrix, p) + self.state_estimator.position[:,0]

        rt = rt_hyptheses[i_max]
        velocity = -rt[:3,3] / self.dt
        velocity = np.matmul(self.state_estimator.rotation_matrix, velocity)
        #print(self.state_estimator.rotation_matrix)
        W = (rt[:3,:3] - np.eye(3)) / self.dt
        #print(W)
        angular_velocity = np.array([
            (W[2,1]-W[1,2])/2,
            (W[0,2]-W[2,0])/2,
            (W[1,0]-W[0,1])/2
        ]) #/ 0.036
        #print('v: ', velocity)
        #print('w: ', angular_velocity)
        self.proj = np.matmul(self.proj, np.vstack((rt, [0, 0, 0, 1])))

        if self.state_estimator.points_number > 0 and len(description) > 0:
            #print(camera_direction)
            #print(self.state_estimator.points_viewing_versor)
            point_angle = np.array([np.dot(self.state_estimator.camera_direction, point_versor) for point_versor in self.state_estimator.points_viewing_versor])
            #print(point_angle)
            correct_angle = point_angle > np.cos(np.deg2rad(60))
            #print(self.state_estimator.points[correct_angle])
            #print(self.state_estimator.position.shape)
            camera_point_direction = (self.state_estimator.points[correct_angle]-self.state_estimator.position.T) \
                / np.expand_dims(np.linalg.norm(self.state_estimator.points[correct_angle]-self.state_estimator.position.T, axis=1), axis=1)
            #print(camera_point_direction)
            camera_point_angle = np.array([np.dot(self.state_estimator.camera_direction, cp_dir) for cp_dir in camera_point_direction])
            in_front_of_camera = camera_point_angle > 0
            #print(in_front_of_camera)
            correct_angle_indicies = np.argwhere(correct_angle)[:,0]
            visible_points_indicies = correct_angle_indicies[in_front_of_camera]
            #print(visible_points_indicies)

            # print(self.state_estimator.points_descriptor[visible_points_indicies].shape)
            # print(len(description))
            #print(cur_description)
            matches = self.bf.match(self.state_estimator.points_descriptor[visible_points_indicies], description)
            #print(len(matches))
            matches_bt = [m for m in matches if m.distance < self.match_treshold]
            #print(len(matches_bt))
            old_points_indicies = [visible_points_indicies[m.queryIdx] for m in matches_bt]
            #print(old_points_indicies)
            #old_points_descriptors = self.state_estimator.points[old_points_indicies]
            #print(old_points_indicies)
            cur_points_old_indicies = [m.trainIdx for m in matches_bt]
            #print('old: ', len(cur_points_old_indicies))
            cur_points_new_indicies = [i for i in range(len(cur_inliers)) if i not in cur_points_old_indicies]
            #print('new: ', len(cur_points_new_indicies))
            # p3d = np.expand_dims(self.state_estimator.points[old_points_indicies], axis=2)
            # p2d = np.expand_dims(kp_cur_pt[cur_points_old_indicies], axis=2)
            # print(p3d.shape)
            # print(p2d.shape)
            # _, rvec, tvec = cv2.solvePnP(p3d, p2d, self.calibration_matrix, None)
            # print(tvec)
        else:
            cur_points_old_indicies = []
            old_points_indicies = []
            cur_points_new_indicies = range(points_world_coords.shape[0])


        self.state_estimator.update(velocity, angular_velocity,
                                  points_world_coords[cur_points_new_indicies], description[cur_points_new_indicies],
                                  points_world_coords[cur_points_old_indicies], old_points_indicies)

        img2 = cv2.putText(img, 'pos: ' + str(self.state_estimator.position), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
        img2 = cv2.putText(img2, str(self.state_estimator.rotation), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
        img2 = cv2.putText(img2, 'vel: ' + str(self.state_estimator.velocity), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
        img2 = cv2.putText(img, 'ang vel: ' + str(self.state_estimator.angular_velocity), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
        img2 = cv2.drawKeypoints(img, kp_cur, None, color=(0,255,0), flags=0)
        cv2.imshow('frame',img2)
        cv2.waitKey(1)

        # input("Press Enter to continue...")
        # print('Points after update: ')
        # print(self.state_estimator.points)
        # e = np.empty((0,))
        # self.state_estimator.update(velocity, angular_velocity,
        #                            e, e,
        #                            e, [])


        # fig = plt.figure()
        # ax = plt.axes(projection="3d")
        # t = self.state_estimator.position
        # r = self.state_estimator.rotation_matrix
        # ax.quiver(t[0], t[1], t[2], r[0,2], r[1,2], r[2,2], color='r')
        #
        # ax.scatter3D(points_max[:,0], points_max[:,1], points_max[:,2])
        # plt.show()
        return velocity, angular_velocity

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

        #### Actualize state estimator based on cur key points



        #### add new points base on triangulation

        #t = time()
        kp_ref_pt, kp_cur_pt, cur_description = self._find_matches(img)
        #print(kp_ref_pt.shape, kp_cur_pt.shape, cur_description.shape)
        #print('Finding matches on imgs: ', time() - t)


        # #t = time()
        # if self.state_estimator.points_number > 0:
        #     camera_direction = np.matmul(self.state_estimator.rotation_matrix, self.z_versor)
        #     #print(camera_direction)
        #     #print(self.state_estimator.points_viewing_versor)
        #     point_angle = np.array([np.dot(camera_direction[:,0], point_versor) for point_versor in self.state_estimator.points_viewing_versor])
        #     #print(point_angle)
        #     correct_angle = point_angle > np.cos(np.deg2rad(60))
        #     #print(self.state_estimator.points[correct_angle].shape)
        #     #print(self.state_estimator.position.shape)
        #     camera_point_direction = (self.state_estimator.points[correct_angle]-self.state_estimator.position.T) \
        #         / np.expand_dims(np.linalg.norm(self.state_estimator.points[correct_angle]-self.state_estimator.position.T, axis=1), axis=1)
        #     #print(camera_point_direction)
        #     camera_point_angle = np.array([np.dot(camera_direction[:,0], cp_dir) for cp_dir in camera_point_direction])
        #     in_front_of_camera = camera_point_angle > 0
        #     correct_angle_indicies = np.argwhere(correct_angle)[:,0]
        #     visible_points_indicies = correct_angle_indicies[in_front_of_camera]
        #
        #     #print(self.state_estimator.points_descriptor[visible_points_indicies])
        #     #print(cur_description)
        #     matches = self.bf.match(self.state_estimator.points_descriptor[visible_points_indicies], cur_description)
        #     matches_bt = [m for m in matches if m.distance < self.match_treshold]
        #     old_points_indicies = [visible_points_indicies[m.queryIdx] for m in matches_bt]
        #     old_points_descriptors = self.state_estimator.points[old_points_indicies]
        #     #print(old_points_indicies)
        #     cur_points_old_indicies = [m.trainIdx for m in matches_bt]
        #     cur_points_new_indicies = [i for i in range(len(kp_cur_pt)) if i not in cur_points_old_indicies]
        # else:
        #     old_points = np.zeros((0,3))
        #     old_points_descriptors = np.zeros((0,32))
        #     old_points_indicies = np.array([], dtype=np.int)
        #     cur_points_new_indicies = range(len(kp_cur_pt))
        #     cur_points_old_indicies = []
        # #print('Finding matches in map: ', time() - t)
        # # Calculation of homography and fundamental could be parallelized
        # #homography_matrix, _ = self._find_homography(kp_ref_pt, kp_cur_pt)
        # points_in_map = []

        #t = time()
        # For now use only triangulation
        if True:
            extrinsic_params, ext_change = self._find_extrinsic_triangulation(kp_ref_pt, kp_cur_pt)
        else:
            extrinsic_params = self._find_extrinsic_pnp(self.state_estimator.points[cur_points_old_indicies], kp_cur_pt[cur_indicies_of_old_points])
        #print('Finding extrinscic: ', time() - t)
        #print(extrinsic_params)

        self.proj = np.matmul(ext_change, self.proj)
        self.z_versor = np.matmul(ext_change[:3,:3], self.z_versor)

        new_pos = extrinsic_params[:3,3]
        new_rot = extrinsic_params[:3,:3]
        #print(new_rot)
        #print(pos_rotate)
        #print(new_pos)
        self.position[self.pos_i,:] = new_pos
        self.pos_i += 1
        #print(np.log(ext_change[:3,:3]))
        R = ext_change[:3,:3]
        #print( np.linalg.inv(R)-R.T )
        #vel = ext_change[:3,3]
        # ang_vel = np.array([
        #     [(ext_change[1,2]+ext_change[2,1])/2],
        #     [(ext_change[0,2]+ext_change[2,0])/2],
        #     [(ext_change[0,1]+ext_change[1,0])/2],
        # ])
        # print(vel)
        # print(ang_vel.T)

        #print(self.state_estimator.extrinsic_matrix_3x4)
        #print(extrinsic_params[:3,:])
        #t = time()
        #print(self.state_estimator.extrinsic_matrix_3x4)
        points_homo = cv2.triangulatePoints(np.matmul(self.calibration_matrix, self.state_estimator.extrinsic_matrix_3x4), np.matmul(self.calibration_matrix, extrinsic_params[:3,:]), kp_ref_pt.T, kp_cur_pt.T)
        points = cv2.convertPointsFromHomogeneous(points_homo.T)
        #print(points_homo)
        points = np.reshape(points, (points.shape[0], points.shape[2]))
        #print('Triangulation: ', time() - t)

        # ax.clear()
        # ax.quiver(0, 0, 0, 0, 0, 1)
        # r = np.matmul(new_rot, [0,0,1])
        # ax.quiver(new_pos[0], new_pos[1], new_pos[2], r[0], r[1], r[2], color='r')

        # ax.scatter3D(points[:,0], points[:,1], points[:,2])
        # plt.show()
        #
        # new_points = points[cur_points_new_indicies]
        # old_points = points[cur_points_old_indicies]
        # new_desciptions = cur_description[cur_points_new_indicies]
        # #print('new ', new_points.shape)
        #print('old ', old_points.shape)

        #t = time()
        #self.state_estimator.update(new_pos, new_rot, new_points, new_desciptions, old_points, old_points_descriptors, old_points_indicies)
        #print('Update Kalman: ', time() - t)

        self.ref_img = img

    def _find_extrinsic_triangulation(self, reference_points, current_points):
        """Find extrinscic matrix using triangulation

        Args:

        """
        fundamental = cv2.findFundamentalMat(reference_points, current_points, cv2.FM_RANSAC)[0]
        extrinscic_change = self._calculate_camera_shift(reference_points, current_points, fundamental)
        #r = extrinscic_change[:3,:3]
        #print(r)
        #new_extrinsic = np.matmul(extrinscic_change[:3,:3], self.calibration_matrix)
        #print(new_extrinsic)

        #print(1e3 * (r - np.eye(3)))
        #print(r)

        #print(np.matmul(self.state_estimator.extrinsic_matrix_3x4, extrinscic_change))


        return np.matmul(self.state_estimator.extrinsic_matrix_4x4, extrinscic_change), extrinscic_change

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
        translation_hypotheses[1,:,0] = translation_hypotheses[3,:,0] = -U[:,2]

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
                point2 = np.matmul(rt_matrix[:3,:3], point) + rt_matrix[:,3]
                if point[2] > 0 and point2[2] > 0:
                    score += 1
            if score > score_max:
                score_max = score
                i_max = i
            #print(score)
        return np.vstack((np.hstack((rotation_hypotheses[i_max], translation_hypotheses[i_max])), [0, 0, 0, 1]))
        #return np.hstack((rotation_hypotheses[i_max], translation_hypotheses[i_max]))

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

        img = cv2.putText(cur_img, 'pos: ' + str(self.state_estimator.position), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
        img = cv2.putText(cur_img, str(self.state_estimator.rotation), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
        img = cv2.putText(cur_img, 'vel: ' + str(self.state_estimator.velocity), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
        img = cv2.putText(cur_img, 'ang vel: ' + str(self.state_estimator.angular_velocity), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
        img = cv2.drawKeypoints(img, kp_cur, None, color=(0,255,0), flags=0)
        cv2.imshow('frame',img)
        cv2.waitKey(1)

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
