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

    @property
    def points_descriptors(self):
        return self.points_descriptors_placehorder[:self.N_points]

    def add_points(self, points, descriptors, matcher, match_treshold):
        """
        """
        # Look for points already in map
        # TODO: add chosing points by viewing versor
        matches = matcher.match(self.points_descriptors, descriptors)
        matches = [m for m in matches if m.distance < match_treshold]
        old_indicies = [m.queryIdx for m in matches]
        points_in_map = [m.trainIdx for m in matches]
        points_not_in_map = [i for i in range(points.shape[0]) if i not in points_in_map]
        new_points = points[points_not_in_map]

        # add new points to map
        n = new_points.shape[0]
        new_indicies = np.array(range(self.N_points, self.N_points+n))
        self.points_placehorder[new_indicies,:] = new_points
        self.points_descriptors_placehorder[new_indicies,:] = descriptors[points_not_in_map]
        # add saving viewing versor
        self.N_points += n

        indicies = np.empty((points.shape[0]), dtype=np.int)
        indicies[points_in_map] = old_indicies
        indicies[points_not_in_map] = new_indicies

        return indicies

    def update_points(self, p):
        """
        """
        # NOTE: for now work only when updating all points
        self.points_placehorder[:self.N_points,:] = p
