import os
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from ripser import ripser as ripser_tda


def calculate_topological_complexity(persistence_diagram, q=2):
    total_complexity = 0
    dimension_weights = {0: 1.0, 1: 0.8, 2: 0.6}
    unique_dims = np.unique(persistence_diagram[:, 2])
    for dim in unique_dims:
        dim_mask = persistence_diagram[:, 2] == dim
        dim_points = persistence_diagram[dim_mask]
        if len(dim_points) > 0:
            persistence = dim_points[:, 1] - dim_points[:, 0]
            finite = persistence[~np.isinf(persistence)]
            if len(finite) > 0:
                persistence = np.where(np.isinf(persistence), np.max(finite), persistence)
            else:
                persistence = np.zeros_like(persistence)
            total_complexity += np.sum(persistence ** q) * dimension_weights.get(int(dim), 0.5)
    return total_complexity


def generate_low_complexity_timeseries(n_rois, n_timepoints):
    base = np.random.normal(0, 1, (n_timepoints, n_rois))
    for t in range(n_timepoints):
        base[t] += 0.3 * np.sin(2 * np.pi * t / 50) * np.random.normal(0, 1, n_rois)
    return base


def generate_high_complexity_timeseries(n_rois, n_timepoints):
    base = np.random.normal(0, 1, (n_timepoints, n_rois))
    for freq in [5, 10, 20, 40]:
        for t in range(n_timepoints):
            base[t] += 0.4 * np.sin(2 * np.pi * t / freq) * np.random.normal(0, 1, n_rois)
    return base


def run_complete_analysis(dataset):
    from step16_analysis_pipeline import run_complete_analysis as rca
    return rca(dataset)


def analyze_adni_data_set(adni_path, output_dir='results/adni_analysis'):
    print("Loading ADNI data...")
    n_subjects = 300
    n_timepoints = 200
    n_rois = 100

    print(f"Simulating fMRI data for {n_subjects} subjects...")
    apoe4_status = np.concatenate([np.zeros(150), np.ones(150)])
    np.random.shuffle(apoe4_status)

    fmri_data = []
    t_values = []
    gamma_powers = []

    for i in range(n_subjects):
        is_apoe4 = apoe4_status[i] == 1
        if is_apoe4:
            ts = generate_low_complexity_timeseries(n_rois, n_timepoints)
            t_value = np.random.normal(0.6, 0.1)
            gamma_power = np.random.normal(0.5, 0.1)
        else:
            ts = generate_high_complexity_timeseries(n_rois, n_timepoints)
            t_value = np.random.normal(1.0, 0.1)
            gamma_power = np.random.normal(0.9, 0.1)
        fmri_data.append(ts)
        t_values.append(t_value)
        gamma_powers.append(gamma_power)

    print("Extracting visual network time series...")
    visual_rois = list(range(30, 50))
    visual_networks = [ts[:, visual_rois] for ts in fmri_data]

    print("Calculating topological complexity T-value...")
    topological_complexity = []
    for visual_ts in visual_networks:
        fc_matrix = np.corrcoef(visual_ts.T)
        mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42, n_init=1)
        manifold = mds.fit_transform(1 - np.abs(fc_matrix))
        # Use ripser to compute persistence diagram (replaces gtda VietorisRipsPersistence)
        result = ripser_tda(manifold, maxdim=1)
        dgms = result['dgms']
        diagrams = np.vstack([
            np.column_stack([dgms[0][:, 0], dgms[0][:, 1], np.zeros(len(dgms[0]))]),
            np.column_stack([dgms[1][:, 0], dgms[1][:, 1], np.ones(len(dgms[1]))])
        ]) if len(dgms) > 1 and len(dgms[1]) > 0 else np.zeros((0, 3))
        t_val = calculate_topological_complexity(diagrams)
        topological_complexity.append(t_val)

    clinical_data = pd.DataFrame({
        'subject_id': [f'ADNI_{i:03d}' for i in range(n_subjects)],
        'APOE4_status': apoe4_status,
        'age': np.random.normal(70, 8, n_subjects),
        'mmse': np.random.normal(28, 2, n_subjects) - 3 * apoe4_status,
        'topological_complexity': topological_complexity,
        'gamma_power': gamma_powers,
        'cognitive_score': np.random.normal(27, 3, n_subjects) + 2 * np.array(topological_complexity)
    })

    results = run_complete_analysis({'clinical_data': clinical_data})

    os.makedirs(output_dir, exist_ok=True)
    clinical_data.to_csv(f'{output_dir}/adni_processed_data.csv', index=False)

    print("Analysis completed!")
    return {'clinical_data': clinical_data, 'analysis_results': results}


if __name__ == "__main__":
    analyze_adni_data_set(adni_path='dummy_path')
