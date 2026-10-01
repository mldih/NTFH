import numpy as np
from ripser import ripser


class TopologyAnalyzer:
    """Topology analysis module for neural manifold analysis."""

    def __init__(self, config=None):
        self.config = config or {}
        self.persistence_results = {}

    def compute_persistence_diagram(self, point_cloud, max_dim=2):
        """Compute persistence diagram of a point cloud using ripser."""
        result = ripser(point_cloud, maxdim=max_dim)
        dgms = result['dgms']
        persistence_diagram = []
        for dim in range(len(dgms)):
            if len(dgms[dim]) > 0:
                persistence_diagram.append(
                    np.column_stack([dgms[dim][:, 0], dgms[dim][:, 1],
                                     np.full(len(dgms[dim]), dim)])
                )
        if len(persistence_diagram) == 0:
            return np.zeros((0, 3))
        return np.vstack(persistence_diagram)

    def calculate_topological_complexity(self, persistence_diagram, q=2):
        """Calculate topological complexity from persistence diagram."""
        total_complexity = 0
        dimension_weights = {0: 1.0, 1: 0.8, 2: 0.6}
        unique_dims = np.unique(persistence_diagram[:, 2])
        for dim in unique_dims:
            dim_mask = persistence_diagram[:, 2] == dim
            dim_points = persistence_diagram[dim_mask]
            if len(dim_points) > 0:
                persistence = dim_points[:, 1] - dim_points[:, 0]
                finite_mask = ~np.isinf(persistence)
                if finite_mask.any():
                    persistence = np.where(np.isinf(persistence),
                                           np.max(persistence[finite_mask]),
                                           persistence)
                dim_complexity = np.sum(persistence ** q)
                weight = dimension_weights.get(int(dim), 0.5)
                total_complexity += weight * dim_complexity
        return total_complexity

    def analyze_point_cloud(self, point_cloud, q=2):
        """Full topology analysis of a point cloud."""
        diagram = self.compute_persistence_diagram(point_cloud)
        t_value = self.calculate_topological_complexity(diagram, q=q)
        return {
            'persistence_diagram': diagram,
            'topological_complexity': t_value
        }
