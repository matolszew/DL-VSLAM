import numpy as np
import quaternion

class EKF:
    """Extended Kalman Filter for Visual SLAM

    Args:
        dt (float):
        acceleration_variance (float):
        angular_acceleration_variance (float):
    """
    def __init__(self, dt=1, acceleration_variance=1, angular_acceleration_variance=1, initial_variance=1):
        self.position = np.zeros((3,1))
        self.rotation = np.quaternion(1,0,0,0)
        self.velocity = np.zeros((3,1))
        self.angular_velocity = np.zeros(3,1)
        self.dt = dt
        self.initial_variance = initial_variance
        self.covariance_matrix = initial_variance*np.ones((13,13))
        self.points = []
        self.points_descriptors = []
        self.points_variance = []

        self.acc_var = acceleration_variance*dt*np.eye(3)
        self.ang_acc_var = angular_acceleration_variance*dt*np.eye(3)

        self.process_noise = np.vstack((
            np.hstack(acc, np.zeros((3,3))),
            np.hstack(np.zeros((3,3)), ang_acc)
        ))

    def update(self, position_observation, rotation_observation, points, points_descriptors):
        """
        """
        # NOTE In estimation step points are unnecessary couse dont change their position
        state_len = 13 + 6*points.shape[0]
        cov_matrix = self.initial_variance*np.eye((state_len,state_len))
        cov_matrix[:13,:13] = self.covariance_matrix[:13,:13]
        # TODO: for points seen before change value in cov_matrix

        position_prediction = self.points + self.velocity*dt
        w_norm = np.linalg.norm(self.angular_velocity)
        q2 = np.quaternion(
            np.cos(w_norm/2),
            np.sin(w_norm/2)*self.angular_velocity[0]/w_norm,
            np.sin(w_norm/2)*self.angular_velocity[1]/w_norm,
            np.sin(w_norm/2)*self.angular_velocity[2]/w_norm,
        )
        rotation_prediction = self.rotation * q2

        jacobian_q_q = self._yupsilon_complementar(q2)
        jacobian_q_omega = np.matmul(self._yupsilon(rotation_prediction), self._jacobian_omega_omega(self.angular_velocity))

        jacobian_F_x = np.vstack((
            np.hstack((np.eye(3), np.zeros((3,3)), self.dt*np.eye(3), np.zeros((3,3)))),
            np.hstack((np.zeros((4,3)), jacobian_q_q, np.zeros((4,3)), jacobian_q_omega)),
            np.hstack((np.zeros((6,6)), np.eye(6)))
        ))

        jacobian_F_n = np.vstack((
            np.hstack((np.eye(3)*self.dt, np.zeros(3,3))),
            np.hstack((np.zeros((4,3)), jacobian_q_omega)),
            np.eye(6)
        ))
        covariance_prediction = np.matmul(np.matmul(jacobian1, cov_matrix), jacobian1.T) +
            np.matmul(np.matmul(jacobian_F_n, self.process_noise), jacobian_F_n.T)

        ### Uodate step ####


    def _yupsilon(q):
        """
        """
        return numpy.array([
            [q[0], -q[1], -q[2], -q[3]],
            [q[1],  q[0], -q[3],  q[2]],
            [q[2],  q[3],  q[0],  q[1]],
            [q[3], -q[2],  q[1],  q[0]]
        ])

    def _yupsilon_complementar(q):
        """
        """
        return numpy.array([
            [q[0], -q[1], -q[2], -q[3]],
            [q[1],  q[0],  q[3], -q[2]],
            [q[2], -q[3],  q[0],  q[1]],
            [q[3],  q[2], -q[1],  q[0]]
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
            return (self.dt/2) * np.vstack((np.zeros(1,3), np.eye(3)))
