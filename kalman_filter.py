import numpy as np
import quaternion

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
    def __init__(self, dt=1, acceleration_variance=1e-2, angular_acceleration_variance=1e-2, initial_variance=1e-2):
        self.position = np.zeros((3,1))
        self.rotation = np.quaternion(1,0,0,0)
        self.velocity = np.zeros((3,1))
        self.angular_velocity = np.zeros((3,1))
        self.dt = dt
        self.initial_variance = initial_variance
        self.covariance_matrix_camera = initial_variance*np.eye(13)
        self.covariance_matrix_camera[:7,:7] = 0
        self.points_placehorder = np.zeros((100000, 3))
        self.points_descriptors_placehorder = np.zeros((100000, 32), dtype=np.uint8)
        self.points_variance_placehorder = np.zeros((100000,3))
        self.points_viewing_versor_placehorder = np.zeros((100000, 3))

        self.points_number = 0

        self.acc_var = acceleration_variance*dt*np.eye(3)
        self.ang_acc_var = angular_acceleration_variance*dt*np.eye(3)

        self.process_noise = np.vstack((
            np.hstack((self.acc_var, np.zeros((3,3)))),
            np.hstack((np.zeros((3,3)), self.ang_acc_var))
        ))

    @property
    def rotation_matrix(self):
        return quaternion.as_rotation_matrix(self.rotation)

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

    def update(self, position_observation, rotation_observation,
               new_points, new_points_descriptors,
               old_points, old_points_descriptors, old_points_indicies):
        """
        """
        rotation_observation_quat = quaternion.from_rotation_matrix(rotation_observation)
        # NOTE In estimation step points are unnecessary couse dont change their position
        new_n = 3*new_points.shape[0]
        old_n = 3*old_points.shape[0]
        state_len = 13 + new_n + old_n
        observation_len = 7 + new_n + old_n
        #print(state_len)

        #t = time()
        state = np.zeros((state_len, 1))
        state[:3] = self.position
        state[3:7] = quaternion.as_float_array([self.rotation]).T
        state[7:10] = self.velocity
        state[10:13] = self.angular_velocity
        if old_n > 0:
            state[13:13+old_n] = self.points_placehorder[old_points_indicies].reshape((old_n,1))
        #print('pos state: ', state[:3].T)
        #print('rot state: ', state[3:7].T)
        #print('Creating state: ', time() - t)

        #t = time()
        covariance_matrix = np.eye(state_len)
        covariance_matrix[:13,:13] = self.covariance_matrix_camera
        if old_n > 0:
            covariance_matrix[13:13+old_n] = np.diag(self.points_variance_placehorder[old_points_indicies].reshape((old_n,1)))
        #print('Creating cov matrix: ', time() - t)

        #t = time()
        observation = np.empty((observation_len,1))
        observation[:3,0] = position_observation
        observation[3:7] =  quaternion.as_float_array([rotation_observation_quat]).T
        if old_n > 0:
            observation[7:7+old_n,0] = old_points.reshape((old_n))
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
        #print(q2)
        rotation_prediction = self.rotation * q2
        prediction[3:7] = quaternion.as_float_array([rotation_prediction]).T
        #print(rotation_prediction)
        prediction[7:] = state[7:]
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
        observation_matrix[:7,:7] = np.eye(7)
        observation_matrix[7:, 13:] = np.eye(state_len-13)
        # observation_matrix = np.vstack((
        #     np.hstack((np.eye(7), np.zeros((7,state_len-7)))),
        #     np.hstack((np.zeros((state_len-7,13)), np.eye(state_len-7)))
        # ))
        innovation = observation - np.matmul(observation_matrix, prediction)
        #print(innovation.shape)
        #print('Innovation: ', time() - t)

        # jacobian = np.vstack((
        #     np.hstack((np.eye(7), np.zeros((7, state_len-7)))),
        #     np.hstack((np.zeros((state_len-13,13)), np.eye(state_len-13)))
        # ))
        # TODO: add variable to change camera position observation noise on initialization
        t = time()
        observation_noise = np.eye(observation_len)*1e-2
        observation_noise[:7,:7] = 1
        innovation_cov = np.matmul(np.matmul(observation_matrix, covariance_prediction), observation_matrix.T) + observation_noise
        #print(innovation_cov.shape)
        #print('Innovation cov: ', time() - t)
        t = time()
        gain = np.matmul(np.matmul(covariance_prediction, observation_matrix.T), np.linalg.inv(innovation_cov))
        #print(gain.shape)
        #print(innovation.shape)
        #print('Gain: ', time() - t)

        t = time()
        #print(state.shape)
        new_state = state + np.matmul(gain, innovation)
        #print(np.matmul(gain, innovation).shape)
        #print(new_state.shape)
        new_covariance = covariance_prediction - np.matmul(np.matmul(gain, innovation_cov), gain.T)
        #print('New state and cov: ', time() - t)

        t = time()
        # Normalize quaternion
        new_state[3:7] /= np.linalg.norm(new_state[3:7])
        self.rotation = np.quaternion(new_state[3], new_state[4], new_state[5], new_state[6])
        # TODO: correct covariance matrix adfter normalization
        #new_covariance[]
        #print('Normalize quat: ', time() - t)

        t = time()
        #print('pos obs: ', position_observation)
        self.position = new_state[:3]
        #print('pos new: ', self.position.T)
        # NOTE coś jest źle z rotacją bo jest za duży obrót w jednym kroku!!!!!!
        #print(self.rotation_matrix)
        #print('rot obs:', rotation_observation_quat)
        self.rotation = np.quaternion(new_state[3], new_state[4], new_state[5], new_state[6])
        #print('rot new: ', self.rotation)
        #print(self.rotation_matrix)
        self.velocity = new_state[7:10]
        self.angular_velocity = new_state[10:13]
        #print(new_state[:13])

        #print(self.extrinsic_matrix_3x4)

        self.covariance_matrix_camera = new_covariance[:13,:13]
        #print('Actualzie states and cov: ', time() - t)

        #t = time()
        camera_direction = np.matmul(self.rotation_matrix, np.array([[0,0,1]]).T)
        if old_n > 0:
            self.points_placehorder[old_points_indicies] = new_state[13:13+old_n].reshape((int(old_n/3),3))
            self.points_variance_placehorder[old_points_indicies] = np.diag(new_covariance[13:13+old_n]).reshape((int(old_n/3),3))
            self.points_viewing_versor_placehorder[old_points_indicies] += camera_direction.T
            self.points_viewing_versor_placehorder[old_points_indicies] /= np.expand_dims(np.linalg.norm(self.points_viewing_versor_placehorder[old_points_indicies], axis=1), axis=1)

        if new_n > 0:
            self.points_placehorder[self.points_number:self.points_number+int(new_n/3)] = new_state[-new_n:].reshape((int(new_n/3),3))
            self.points_variance_placehorder[self.points_number:self.points_number+int(new_n/3)] = np.diag(new_covariance[-new_n:]).reshape((int(new_n/3),3))
            self.points_viewing_versor_placehorder[self.points_number:self.points_number+int(new_n/3)] = camera_direction.T
            self.points_descriptors_placehorder[self.points_number:self.points_number+int(new_n/3)] = new_points_descriptors
            self.points_number += int(new_n/3)
        #print('Actuaize points: ', time() - t)

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
