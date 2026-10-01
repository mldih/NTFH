import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from step1_test_topology import calculate_topological_complexity


class TestTopologyAnalysis:
    def test_topological_complexity_basic(self):
        test_persistence = np.array([
            [0.0, 1.0, 0],
            [0.0, 2.0, 1],
        ])
        complexity = calculate_topological_complexity(test_persistence)
        assert complexity > 0

    def test_topological_complexity_circle(self):
        from ripser import ripser as ripser_tda
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_data = np.column_stack([np.cos(theta), np.sin(theta)])
        result = ripser_tda(circle_data, maxdim=1)
        dgms = result['dgms']
        diagrams = np.vstack([
            np.column_stack([dgms[0][:, 0], dgms[0][:, 1], np.zeros(len(dgms[0]))]),
            np.column_stack([dgms[1][:, 0], dgms[1][:, 1], np.ones(len(dgms[1]))])
        ])
        complexity = calculate_topological_complexity(diagrams)
        assert complexity > 0

    def test_topological_complexity_zero(self):
        test_persistence = np.array([[0.0, 0.0, 0]])
        complexity = calculate_topological_complexity(test_persistence)
        assert complexity == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
