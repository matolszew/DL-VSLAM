import collections

Edge = collections.namedtuple('Edge', ['node', 'weight'])

class CovisibilityGraph:
    """
    """
    def __init__(self, matcher, match_treshold, min_weight=100):
        self.graph = collections.OrderedDict()
        self.matcher = matcher
        self.min_weight = min_weight

    def add_node(self, new_key_frame):
        """
        """
        new_edges = []
        for key_frame in self.graph.keys():
            matches = self.matcher.match(new_key_frame.descriptors, key_frame.descriptors)
            weight = len([m for m in matches if m.distance < self.match_treshold])
            if weight >= min_weight:
                new_edges.append(Edge(keyframe, weight))
                self.graph[keyframe].append(Edge(new_key_frame, weight))

        self.graph[new_key_frame] = new_edges

    def reset(self):
        self.graph = {}

    def __len__(self):
        return len(self.graph.keys())

    def __getitem__(self):
        return self.graph.keys()
