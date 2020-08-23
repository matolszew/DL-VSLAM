import numpy as np
from scipy.spatial.transform import Rotation

class KeyFrame:
    """
    """
    def __init__(self, keypoints, descriptors, camera_position=None, camera_rotation=None):
        self.keypoints = keypoints
        self.points2d = np.array([kp.pt for kp in self.keypoints])
        self.descriptors = descriptors
        self.camera_position = None
        self.camera_rotation = None
        self.map_indicies = -1 * np.ones((len(self.keypoints)), dtype=np.int)

    @property
    def camera_quaternion(self):
        # NOTE: scalar part of quaternion is on the last position!!!!!!!!!!
        return self.camera_rotation.as_quat()

    @property
    def camera_rotaion_matrix(self):
        return self.camera_rotation.as_matrix()

    @property
    def camera_rotation_vector(self):
        return self.camera_rotation.as_rotvec()

    @property
    def camera_extrinsic(self):
        proj = np.eye(4)
        proj[:3,:3] = self.camera_rotaion_matrix
        proj[:3,3] = self.camera_position
        return proj

    @property
    def points_on_img(self):
        in_map = self.map_indicies >= 0
        indicies = self.map_indicies[in_map]
        return self.points2d[indicies], indicies

    @property
    def descriptors_in_map(self):
        in_map = self.map_indicies >= 0
        return self.descriptors[in_map]

    def update_position_and_rotation(self, rt):
        """

        Args:
            rt (np.array): Rotation-translation matrix
        """
        self.camera_rotation = Rotation.from_matrix(rt[:3,:3])
        self.camera_position = rt[:,3].reshape((3,1))

    def update_rotation_from_quat(self, q):
        """
        """
        self.camera_rotation = Rotation.from_quat(q)

    def update_rotation_from_rotation_vector(self, rvec):
        """
        """
        self.camera_rotation = Rotation.from_rotvec(rvec)

    def update_position(self, t):
        """
        """
        self.camera_position = t.reshape((3,1))

    def update_indicies(self, map_indicies, frame_indicies):
        """
        """
        self.map_indicies[frame_indicies] = map_indicies
