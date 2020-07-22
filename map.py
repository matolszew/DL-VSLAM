import numpy as np

class Map:
    """
    """
    def __init__(self, max_points=10000):
        self.N_points = 0
        self.max_points = max_points
        self.points_placehorder = np.empty((max_points, 3))
        self.points_descriptors_placehorder = np.empty((max_points, 32), dtype=np.uint8)
        self.points_viewing_versor_placehorder = np.zeros((max_points, 3))

    def __len__(self):
        return self.N_points

    @property
    def points3d(self):
        return self.points_placehorder[:self.N_points]

    def add_points(self, points, descriptors):
        """
        """
        n = points.shape[0]
        self.points_placehorder[self.N_points:self.N_points+n,:] = points
        self.points_descriptors_placehorder[self.N_points:self.N_points+n,:] = descriptors
        # add saving viewing versor
        self.N_points += n
