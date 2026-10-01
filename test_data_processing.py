import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from step15_simulated_data import generate_simulated_dataset


class TestDataProcessing:
    def test_data_generator_initialization(self):
        dataset = generate_simulated_dataset(n_subjects=100, apoe4_prevalence=0.3)
        assert 'clinical_data' in dataset
        assert len(dataset['clinical_data']) == 100

    def test_neural_activity_generation(self):
        dataset = generate_simulated_dataset(n_subjects=50)
        df = dataset['clinical_data']
        assert 'gamma_power' in df.columns
        assert 'topological_complexity' in df.columns
        assert np.std(df['gamma_power'].values) > 0

    def test_dataset_generation(self):
        dataset = generate_simulated_dataset(n_subjects=50)
        df = dataset['clinical_data']
        assert 'APOE4_status' in df.columns
        assert 'gamma_power' in df.columns
        assert 'topological_complexity' in df.columns
        apoe4_ratio = df['APOE4_status'].mean()
        assert 0.1 <= apoe4_ratio <= 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
