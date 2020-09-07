import collections

Edge = collections.namedtuple('Edge', ['node', 'weight'])

class CovisibilityGraph:
    """
    """
    def __init__(self, matcher, match_treshold, min_weight=100):
        self.graph = collections.OrderedDict()
        self.matcher = matcher
        self.match_treshold = match_treshold
        self.min_weight = min_weight
        self.candidat = None
        self.candidat_edges = None

    @property
    def keyframes(self):
        return list(self.graph.keys())

    def connected_frames(self, keyframe):
        return self.graph[keyframe]

    def add_node(self, new_key_frame):
        """
        """
        new_edges = []
        for keyframe in self.graph.keys():
            matches = self.matcher.match(new_key_frame.descriptors, keyframe.descriptors)
            weight = len([m for m in matches if m.distance < self.match_treshold])
            if weight >= self.min_weight:
                new_edges.append(Edge(keyframe, weight))
                self.graph[keyframe].append(Edge(new_key_frame, weight))

        self.graph[new_key_frame] = new_edges

    def add_node_candidat(self, new_key_frame):
        """
        """
        self.candidat_edges = []
        for keyframe in self.graph.keys():
            matches = self.matcher.match(new_key_frame.descriptors, keyframe.descriptors)
            weight = len([m for m in matches if m.distance < self.match_treshold])
            if weight >= self.min_weight:
                self.candidat_edges.append(Edge(keyframe, weight))
        self.candidat = new_key_frame

    def add_candidat_to_graph(self):
        """
        """
        self.graph[self.candidat] = self.candidat_edges
        for edge in self.candidat_edges:
            self.graph[edge.node].append(Edge(self.candidat, edge.weight))
        self.candidat = None
        self.candidat_edges = None

    def remove_candidat(self):
        """
        """
        self.candidat = None
        self.candidat_edges = None

    def get_local_keyframes_and_neigbours(self):
        """Returns keyframes connected to candidat and their neigbours
        """
        kfs = set()
        kfs.add(self.candidat)
        for edge in self.candidat_edges:
            kfs.add(edge.node)
            for edge_neigbour in self.graph[edge.node]:
                kfs.add(edge_neigbour.node)
        return kfs

    def get_local_keyframes(self, keyframe):
        kfs = [keyframe]
        for edge in self.connected_frames[keyframe]:
            kfs.append(edge.node)
        return kfs

    def reset(self):
        self.graph = {}

    def __len__(self):
        return len(self.graph.keys())

    def __getitem__(self, position):
        return self.keyframes[position]
