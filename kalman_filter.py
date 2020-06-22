import numpy as np
import quaternion
import cv2

from time import time

def yupsilon(q):
    """
    """
    q = quaternion.as_float_array(q)
    return np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1],  q[0], -q[3],  q[2]],
        [q[2],  q[3],  q[0],  q[1]],
        [q[3], -q[2],  q[1],  q[0]]
    ])

def yupsilon_complementar(q):
    """
    """
    q = quaternion.as_float_array(q)
    return np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1],  q[0],  q[3], -q[2]],
        [q[2], -q[3],  q[0],  q[1]],
        [q[3],  q[2], -q[1],  q[0]]
    ])

class EKF:
    """Extended Kalman Filter for Visual SLAM

    Args:
        dt (float):
        acceleration_variance (float):
        angular_acceleration_variance (float):
    """
    #dt=0.012
    def __init__(self, initial_pose, camera_matrix, distortion_coefficients=None,
                 dt=0.012, acceleration_variance=2e-5, angular_acceleration_variance=2e-5, initial_variance=1e-1):
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.position = np.zeros((3,1))
        self.rotation = np.quaternion(1,0,0,0)
        # self.position = np.expand_dims(initial_pose[:3,3], axis=1)
        # self.rotation = quaternion.from_rotation_matrix(initial_pose[:3,:3])
        self.velocity = np.zeros((3,1))
        self.angular_velocity = 0*np.ones((3,1))
        self.dt = dt
        self.initial_variance = initial_variance
        self.covariance_matrix_camera = np.zeros((13,13))

        self.points_placehorder = np.zeros((1000000, 3))
        self.points_descriptors_placehorder = np.zeros((1000000, 32), dtype=np.uint8)
        self.points_variance_placehorder = np.zeros((1000000,3))
        self.points_viewing_versor_placehorder = np.zeros((1000000, 3))

        self.points_number = 0

        self.acc_var = acceleration_variance*dt*np.eye(3)/self.dt
        self.ang_acc_var = angular_acceleration_variance*dt*np.eye(3)/self.dt

        self.process_noise = np.vstack((
            np.hstack((self.acc_var, np.zeros((3,3)))),
            np.hstack((np.zeros((3,3)), self.ang_acc_var))
        ))
        self.covariance_matrix_camera[7:,7:] = self.process_noise

        self.observation_noise = np.diag([1e1, 1e1, 1e1, 0.6, 0.6, 0.6])

    @property
    def rotation_matrix(self):
        return quaternion.as_rotation_matrix(self.rotation)

    @property
    def rotation_vector(self):
        return quaternion.as_rotation_vector(self.rotation)

    @property
    def points(self):
        return self.points_placehorder[:self.points_number,:]

    @property
    def points_descriptor(self):
        return self.points_descriptors_placehorder[:self.points_number,:]

    @property
    def points_variance(self):
        return self.points_variance_placehorder[:self.points_number,:]

    @property
    def points_viewing_versor(self):
        return self.points_viewing_versor_placehorder[:self.points_number,:]

    @property
    def extrinsic_matrix_3x4(self):
        return np.hstack((self.rotation_matrix, self.position))

    @property
    def extrinsic_matrix_4x4(self):
        return np.vstack((self.extrinsic_matrix_3x4, np.array([0, 0, 0, 1])))

    @property
    def camera_direction(self):
        return self.rotation_matrix[:,2]

    def initialize_points(self, points, descriptors):
        """Add new points
        """
        N = points.shape[0]
        r = range(self.points_number, self.points_number+N)
        self.points_placehorder[r] = points
        self.points_descriptors_placehorder[r] = descriptors
        self.points_viewing_versor_placehorder[r] = self.camera_direction
        self.points_number += N
        self.points_variance[r] = 1e-3*np.ones_like(points)

        return list(r)

    def update2(self, points_observation, points_indicies):
        """Update state using only points
        """
        #print(points_indicies)
        points_n = len(points_indicies)
        state_n = points_n*3 + 13
        state = np.empty((state_n, 1))
        state[:3] = self.position
        state[3:7] = quaternion.as_float_array([self.rotation]).T
        state[7:10] = self.velocity
        state[10:13] = self.angular_velocity
        state[13:] = np.reshape(self.points[points_indicies], (points_n*3, 1))

        observation = np.reshape(points_observation, (points_n*2, 1))

        covariance_matrix = np.zeros((state_n, state_n))
        covariance_matrix[:13,:13] = self.covariance_matrix_camera
        covariance_matrix[13:,13:] = np.diag(self.points_variance[points_indicies].reshape((points_n*3)))

        state_prediction = np.zeros_like(state)
        state_prediction[:3] = self.position + self.velocity * self.dt
        w_norm = np.linalg.norm(self.angular_velocity*self.dt)
        q2 = np.quaternion(1,0,0,0)
        if w_norm != 0:
            q2 = np.quaternion(
                np.cos(w_norm/2),
                self.dt*self.angular_velocity[0]*np.sin(w_norm/2)/w_norm,
                self.dt*self.angular_velocity[1]*np.sin(w_norm/2)/w_norm,
                self.dt*self.angular_velocity[2]*np.sin(w_norm/2)/w_norm,
            )
        rotation_prediction = self.rotation * q2

        state_prediction[3:7] = quaternion.as_float_array([rotation_prediction]).T
        state_prediction[7:13] = state[7:13]

        jacobian_q_q = yupsilon_complementar(q2)
        #jacobian_q_omega = np.matmul(yupsilon(self.rotation), self._jacobian_omega_omega(self.angular_velocity))
        jacobian_q_omega = np.matmul(yupsilon(rotation_prediction), self._jacobian_omega_omega(self.angular_velocity))

        jacobian_F_x = np.eye(state_n)
        jacobian_F_x[:13,:13] = np.vstack((
            np.hstack((np.eye(3), np.zeros((3,4)), self.dt*np.eye(3), np.zeros((3,3)))),
            np.hstack((np.zeros((4,3)), jacobian_q_q, np.zeros((4,3)), jacobian_q_omega)),
            np.hstack((np.zeros((6,7)), np.eye(6)))
        ))

        jacobian_F_n = np.vstack((
            np.hstack((np.eye(3)*self.dt, np.zeros((3,3)))),
            np.hstack((np.zeros((4,3)), jacobian_q_omega)),
            np.eye(6)
        ))
        Qk = np.zeros((state_n, state_n))
        Qk[:13,:13] = np.matmul(np.matmul(jacobian_F_n, self.process_noise), jacobian_F_n.T)
        #print(np.matmul(np.matmul(jacobian_F_x, covariance_matrix), jacobian_F_x.T))
        covariance_matrix_prediction = np.matmul(np.matmul(jacobian_F_x, covariance_matrix),
                                                 jacobian_F_x.T) + Qk
        #print(covariance_matrix_prediction[:13, :13])

        #print(state[:13])
        #print(points_observation[:10])
        #print(self.rotation_vector)
        #print(cv2.Rodrigues(self.rotation_matrix))
        observation_prediction, projection_jacobian = cv2.projectPoints(self.points[points_indicies],
                                                   cv2.Rodrigues(self.rotation_matrix)[0], self.position,
                                                   self.camera_matrix, distCoeffs=None)
        #print(observation_prediction[:10])

        observation_prediction = np.reshape(observation_prediction, (points_n*2,1))
        innovation = observation - observation_prediction

        jacobian_proj_rvec = projection_jacobian[:,:3]
        jacobian_proj_position = projection_jacobian[:,3:6]
        jacobian_proj_focal = projection_jacobian[:,6:8]
        jacobian_proj_principal = projection_jacobian[:,8:10]

        jacobian_proj_cam = np.zeros((points_n*2, 2))
        for i in range(0, points_n*2, 2):
            #jacobian_proj_cam[i:i+2,:] = np.matmul(jacobian_proj_focal[i:i+2], jacobian_proj_principal[i:i+2])
            jacobian_proj_cam[i:i+2,:] = np.eye(2)
            #jacobian_proj_cam[i:i+2,:] = jacobian_proj_focal[i:i+2]
        jacobian_q2rvec = self._jacobian_quat2rvec()

        quat_conj = np.conjugate(rotation_prediction)
        inv_rotation = quaternion.as_rotation_matrix(quat_conj)

        observation_jacobian = np.zeros((points_n*2, state_n))
        for i in range(0, points_n*2, 2):
            observation_jacobian[i:i+2,:3] = np.matmul(jacobian_proj_cam[i:i+2], jacobian_proj_position[i:i+2])
            observation_jacobian[i:i+2,3:7] = np.matmul(np.matmul(jacobian_proj_cam[i:i+2], jacobian_proj_rvec[i:i+2]),
                                                        jacobian_q2rvec)
            # NOTE: Checi if below is correct
            # observation_jacobian[i:i+2,i+13:i+16] = np.matmul(jacobian_proj_cam[i:i+2], np.ones((2,3)))
            #observation_jacobian[i:i+2,i+13:i+16] = inv_rotation[:2,:]
            observation_jacobian[i:i+2,i+13:i+16] = np.matmul(jacobian_proj_cam[i:i+2], inv_rotation[:2,:])

        # print(jacobian_proj_cam)
        #print(observation_jacobian[:20,:20])

        innovation_matrix = np.matmul(np.matmul(observation_jacobian, covariance_matrix_prediction),
                                      observation_jacobian.T) + np.eye(points_n*2)

        gain = np.matmul(np.matmul(covariance_matrix_prediction, observation_jacobian.T),
                         np.linalg.inv(innovation_matrix))

        new_state = state + np.matmul(gain, innovation)
        new_covariance_matrix = covariance_matrix - \
            np.matmul(np.matmul(gain, innovation_matrix), gain.T)

        # normalize quaternion
        quat_before_norm = new_state[3:7]
        quat_norm = np.linalg.norm(quat_before_norm)
        new_state[3:7] = quat_before_norm / quat_norm
        new_covariance_matrix[3:7,3:7] = (quat_norm**2 - np.matmul(new_state[3:7], new_state[3:7].T)) / (quat_norm**3)

        self.position = new_state[:3]
        self.rotation = np.quaternion(new_state[3,0], new_state[4,0], new_state[5,0], new_state[6,0])
        self.velocity = new_state[7:10]
        self.angular_velocity = new_state[10:13]
        new_points = np.reshape(new_state[13:], (points_n,3))
        self.points[points_indicies] = new_points
        self.points_variance[points_indicies] = np.reshape(np.diag(new_covariance_matrix[13:,13:]), (points_n,3))
        #print(new_covariance_matrix[:16,:16])
        #print(np.mean(self.points_variance, axis=0))

        self.points_viewing_versor[points_indicies] += self.camera_direction
        # Good axis???
        #print(np.linalg.norm(self.points_viewing_versor[points_indicies], axis=0).shape)
        #print(np.linalg.norm(self.points_viewing_versor[points_indicies], axis=1).shape)
        for i in points_indicies:
            self.points_viewing_versor[i] /= np.linalg.norm(self.points_viewing_versor[i])
        #self.points_viewing_versor[points_indicies] /= np.linalg.norm(self.points_viewing_versor[points_indicies], axis=1)

        self.covariance_matrix_camera = new_covariance_matrix[:13,:13]

    def update(self, velocity_observation, angular_velocity_observation,
               new_points, new_points_descriptors,
               old_points, old_points_indicies):
        """
        """
        #rotation_observation_quat = quaternion.from_rotation_matrix(rotation_observation)
        # NOTE In estimation step points are unnecessary couse dont change their position
        new_n = 3*new_points.shape[0]
        old_n = 3*old_points.shape[0]
        state_len = 13 + new_n + old_n
        observation_len = 6 + new_n + old_n
        #print(state_len)

        #t = time()
        state = np.zeros((state_len, 1))
        state[:3] = self.position
        state[3:7] = quaternion.as_float_array([self.rotation]).T
        state[7:10] = self.velocity
        state[10:13] = self.angular_velocity
        if old_n > 0:
            state[13:13+old_n] = self.points_placehorder[old_points_indicies].reshape((old_n,1))
        if new_n > 0:
            state[-new_n:,0] = new_points.reshape((new_n))
        #print('pos state: ', state[:3].T)
        #print('rot state: ', state[3:7].T)
        #print('Creating state: ', time() - t)

        #t = time()
        covariance_matrix = np.eye(state_len)
        #covariance_matrix = np.zeros((state_len, state_len))
        covariance_matrix[:13,:13] = self.covariance_matrix_camera
        if old_n > 0:
           covariance_matrix[13:13+old_n] = np.diag(self.points_variance_placehorder[old_points_indicies].reshape((old_n,1)))
        #print('Creating cov matrix: ', time() - t)

        #t = time()
        observation = np.empty((observation_len,1))
        observation[:3,0] = velocity_observation
        observation[3:6,0] = angular_velocity_observation
        #observation[3:7] =  quaternion.as_float_array([rotation_observation_quat]).T
        if old_n > 0:
            observation[6:6+old_n,0] = old_points.reshape((old_n))
        if new_n > 0:
            observation[-new_n:,0] = new_points.reshape((new_n))
        #print('Creating observation: ', time() - t)

        #t = time()
        prediction = np.zeros((state_len,1))
        prediction[:3] = self.position + self.velocity * self.dt
        #print(position_prediction)
        w_norm = np.linalg.norm(self.angular_velocity*self.dt)
        if w_norm == 0:
            q2 = np.quaternion(1,0,0,0)
        else:
            q2 = np.quaternion(
                np.cos(w_norm/2),
                self.dt*self.angular_velocity[0]*np.sin(w_norm/2)/w_norm,
                self.dt*self.angular_velocity[1]*np.sin(w_norm/2)/w_norm,
                self.dt*self.angular_velocity[2]*np.sin(w_norm/2)/w_norm,
            )
        rotation_prediction = self.rotation * q2

        prediction[3:7] = quaternion.as_float_array([rotation_prediction]).T
        #print(rotation_prediction)
        prediction[7:] = state[7:]
        # print('prediction ')
        # print(prediction.T)
        #print('pos pred:', state[:3].T)
        #print('rot pred:', prediction[3:7].T)
        #print('State prediction: ', time() - t)

        t = time()
        jacobian_q_q = yupsilon_complementar(q2)
        #print(jacobian_q_q.shape)
        jacobian_q_omega = np.matmul(yupsilon(self.rotation), self._jacobian_omega_omega(self.angular_velocity))
        #print(jacobian_q_omega.shape)

        jacobian_F_x = np.vstack((
            np.hstack((np.eye(3), np.zeros((3,4)), self.dt*np.eye(3), np.zeros((3,3)))),
            np.hstack((np.zeros((4,3)), jacobian_q_q, np.zeros((4,3)), jacobian_q_omega)),
            np.hstack((np.zeros((6,7)), np.eye(6)))
        ))
        #print('dF/dx: ', jacobian_F_x[:7,:7])

        jacobian_F_n = np.vstack((
            np.hstack((np.eye(3)*self.dt, np.zeros((3,3)))),
            np.hstack((np.zeros((4,3)), jacobian_q_omega)),
            np.eye(6)
        ))
        covariance_prediction = np.copy(covariance_matrix)
        covariance_prediction[:13,:13] = np.matmul(np.matmul(jacobian_F_x, self.covariance_matrix_camera), jacobian_F_x.T) \
            + np.matmul(np.matmul(jacobian_F_n, self.process_noise), jacobian_F_n.T)
        #print('Covariance prediction: ', time() - t)
        #print(covariance_prediction)
        ### Uodate step ####
        #print(observation.shape)
        #print(prediction.shape)
        t = time()
        observation_matrix = np.zeros((observation_len,state_len))
        observation_matrix[:,7:] = np.eye(observation_len)
        #observation_matrix[:7,:7] = np.eye(7)
        #observation_matrix[7:, 13:] = np.eye(state_len-13)
        # observation_matrix = np.vstack((
        #     np.hstack((np.eye(7), np.zeros((7,state_len-7)))),
        #     np.hstack((np.zeros((state_len-7,13)), np.eye(state_len-7)))
        # ))
        #print('obs: ', observation.T)
        innovation = observation - np.matmul(observation_matrix, prediction)
        #print('pred: ', prediction[7:].T)
        #print(innovation.shape)
        #print('Innovation: ', time() - t)

        # jacobian = np.vstack((
        #     np.hstack((np.eye(7), np.zeros((7, state_len-7)))),
        #     np.hstack((np.zeros((state_len-13,13)), np.eye(state_len-13)))
        # ))
        # TODO: add variable to change camera position observation noise on initialization
        t = time()
        observation_noise = np.eye(observation_len)
        observation_noise[:6,:6] = self.observation_noise
        #observation_noise[:7,:7] = 1
        innovation_cov = np.matmul(np.matmul(observation_matrix, covariance_prediction), observation_matrix.T) + observation_noise
        #print(innovation_cov.shape)
        #print('Innovation cov: ', time() - t)
        t = time()
        gain = np.matmul(np.matmul(covariance_prediction, observation_matrix.T), np.linalg.inv(innovation_cov))
        # print('gain')
        # print(gain)
        #print(gain.shape)
        #print(innovation.shape)
        #print('Gain: ', time() - t)

        t = time()
        #print(state.shape)
        new_state = state + np.matmul(gain, innovation)
        #new_state[:7] = prediction[:7]
        #print(np.matmul(gain, innovation).T)
        #print(np.matmul(gain, innovation).shape)
        #print(new_state.shape)
        new_covariance = covariance_prediction - np.matmul(np.matmul(gain, innovation_cov), gain.T)
        #print('New state and cov: ', time() - t)

        t = time()
        # Normalize quaternion
        #print('before: ', new_state[3:7])
        # new_state[3:7] /= np.linalg.norm(new_state[3:7])
        # self.rotation = np.quaternion(new_state[3], new_state[4], new_state[5], new_state[6])
        # TODO: correct covariance matrix adfter normalization
        #new_covariance[]
        #print('Normalize quat: ', time() - t)

        # print('new state')
        # print(new_state.T)

        t = time()
        #print('pos obs: ', position_observation)
        #self.position = new_state[:3]
        #print('pos new: ', self.position.T)
        # NOTE coś jest źle z rotacją bo jest za duży obrót w jednym kroku!!!!!!
        #print(self.rotation_matrix)
        #print('rot obs:', rotation_observation_quat)
        #self.rotation = np.quaternion(new_state[3], new_state[4], new_state[5], new_state[6])
        #print('rot new: ', self.rotation)
        #print(self.rotation_matrix)
        self.velocity = new_state[7:10]
        self.angular_velocity = new_state[10:13]
        #print(new_state[:13])
        self.position = self.position + self.velocity * self.dt

        w_norm = np.linalg.norm(self.angular_velocity * self.dt)
        if w_norm == 0:
            q2 = np.quaternion(1,0,0,0)
        else:
            q2 = np.quaternion(
                np.cos(w_norm/2),
                self.dt*self.angular_velocity[0]*np.sin(w_norm/2)/w_norm,
                self.dt*self.angular_velocity[1]*np.sin(w_norm/2)/w_norm,
                self.dt*self.angular_velocity[2]*np.sin(w_norm/2)/w_norm,
            )
        self.rotation = self.rotation * q2

        #print(self.extrinsic_matrix_3x4)

        self.covariance_matrix_camera = new_covariance[:13,:13]
        #print(self.covariance_matrix_camera)
        #print('Actualzie states and cov: ', time() - t)

        #t = time()
        self.camera_direction = self.rotation_matrix[:,2]
        if old_n > 0:
            self.points_placehorder[old_points_indicies] = new_state[13:13+old_n].reshape((int(old_n/3),3))
            self.points_variance_placehorder[old_points_indicies] = np.diag(new_covariance[13:13+old_n]).reshape((int(old_n/3),3))
            self.points_viewing_versor_placehorder[old_points_indicies] += self.camera_direction.T
            self.points_viewing_versor_placehorder[old_points_indicies] /= np.expand_dims(np.linalg.norm(self.points_viewing_versor_placehorder[old_points_indicies], axis=1), axis=1)
        if new_n > 0:
            self.points_placehorder[self.points_number:self.points_number+int(new_n/3)] = new_state[-new_n:].reshape((int(new_n/3),3))
            self.points_variance_placehorder[self.points_number:self.points_number+int(new_n/3)] = np.diag(new_covariance[-new_n:]).reshape((int(new_n/3),3))
            self.points_viewing_versor_placehorder[self.points_number:self.points_number+int(new_n/3)] = self.camera_direction.T
            self.points_descriptors_placehorder[self.points_number:self.points_number+int(new_n/3)] = new_points_descriptors
            self.points_number += int(new_n/3)
        #print('Actuaize points: ', time() - t)

    # def _jacobian_quat2rvec(self):
    #     """
    #     """
    #     q = quaternion.as_float_array(self.rotation)
    #     r = q[0]
    #     x = q[1]
    #     y = q[2]
    #     z = q[3]
    #
    #     a_phi = r**2 - x**2 - y**2 + z**2
    #     b_phi = r*x + y*z
    #     M_phi = 4*(b_phi**2) / (a_phi**2) + 1
    #
    #     M_theta = np.sqrt(1 - r*((x*z-r*y)**2))
    #
    #     a_psi = r**2 + x**2 - y**2 - z**2
    #     b_psi = r*z + x*y
    #     M_psi = 4*(b_psi**2) / (a_psi**2) + 1
    #
    #     dphi_dr = (2*x/a_phi - 4*r*b_phi/(a_phi**2)) / M_phi
    #     dphi_dx = (2*r/a_phi + 4*x*b_phi/(a_phi**2)) / M_phi
    #     dphi_dy = (2*z/a_phi + 4*y*b_phi/(a_phi**2)) / M_phi
    #     dphi_dz = (2*y/a_phi - 4*z*b_phi/(a_phi**2)) / M_phi
    #
    #     dtheta_dr = 2*y/M_theta
    #     dtheta_dx = -2*z/M_theta
    #     dtheta_dy = 2*r/M_theta
    #     dtheta_dz = -2*x/M_theta
    #
    #     dpsi_dr = (2*z/a_psi - 4*r*b_psi/(a_psi**2)) / M_psi
    #     dpsi_dx = (2*y/a_psi - 4*x*b_psi/(a_psi**2)) / M_psi
    #     dpsi_dy = (2*x/a_psi + 4*y*b_psi/(a_psi**2)) / M_psi
    #     dpsi_dz = (2*r/a_psi + 4*z*b_psi/(a_psi**2)) / M_psi
    #
    #     return np.array([
    #         [dphi_dr, dphi_dx, dphi_dy, dphi_dz],
    #         [dtheta_dr, dtheta_dx, dtheta_dy, dtheta_dz],
    #         [dpsi_dr, dpsi_dx, dpsi_dy, dpsi_dz],
    #     ])
    def _jacobian_quat2rvec(self):
        """
        """
        q = quaternion.as_float_array(self.rotation)
        r = q[0]
        x = q[1]
        y = q[2]
        z = q[3]

        A = 1 - 2*(x**2+y**2)
        B = r*x + y*z
        C = ((4*B**2)/(A**2)) + 1

        D = np.sqrt(1 - 4*((r*y-x*z)**2))

        E = 1 - 2*(y**2+z**2)
        F = r*z + x*y
        G = ((4*F**2)/(E**2)) + 1

        dphi_dr = 2*x/(A*C)
        dphi_dx = ((8*x*B/A**2) + (2*r/A)) / C
        dphi_dy = ((8*y*B/A**2) + (2*z/A)) / C
        dphi_dz = 2*y/(A*C)

        dtheta_dr = 2*y/D
        dtheta_dx = -2*z/D
        dtheta_dy = 2*r/D
        dtheta_dz = -2*x/D

        dpsi_dr = 2*z/(E*G)
        dpsi_dx = 2*y/(E*G)
        dpsi_dy = ((8*y*F/E**2) + (2*x/E)) / G
        dpsi_dz = ((8*z*F/E**2) + (2*r/E)) / G

        return np.array([
            [dphi_dr, dphi_dx, dphi_dy, dphi_dz],
            [dtheta_dr, dtheta_dx, dtheta_dy, dtheta_dz],
            [dpsi_dr, dpsi_dx, dpsi_dy, dpsi_dz],
        ])

    def _jacobian_omega_omega(self, omega):
        """

        Args:
            omega (np.array): angular velocity (3,1)
        """
        norm = np.linalg.norm(omega)
        if norm > 0:
            n_omega = omega / norm
            x = norm*self.dt/2
            sinc = np.sinc(x)
            return (self.dt/2) * np.vstack((
                -np.sin(x)*n_omega.T,
                np.eye(3)*sinc + (np.cos(x)-sinc)*np.matmul(n_omega, n_omega.T)
                ))
        else:
            return (self.dt/2) * np.vstack((np.zeros((1,3)), np.eye(3)))
