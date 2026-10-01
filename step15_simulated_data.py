import numpy as np
import pandas as pd


def generate_simulated_dataset(n_subjects=200, apoe4_prevalence=0.3,
                                n_neurons=100, n_timepoints=1000,
                                healthy_gamma_mean=0.8, healthy_gamma_std=0.2,
                                pathological_gamma_reduction=0.4, random_seed=42):
    np.random.seed(random_seed)
    subject_ids = []
    groups = []
    apoe4_status = []
    gamma_powers = []
    t_values = []
    cognitive_scores = []
    ages = []

    for i in range(n_subjects):
        is_apoe4 = np.random.rand() < apoe4_prevalence
        apoe4_status.append(int(is_apoe4))
        groups.append('APOE4' if is_apoe4 else 'Control')

        if is_apoe4:
            gamma_base = np.random.normal(healthy_gamma_mean * (1 - pathological_gamma_reduction),
                                          healthy_gamma_std * 0.7)
            t_base = np.random.normal(0.6, 0.15)
            age = np.random.normal(70, 8)
        else:
            gamma_base = np.random.normal(healthy_gamma_mean, healthy_gamma_std)
            t_base = np.random.normal(1.0, 0.1)
            age = np.random.normal(65, 8)

        gamma_power = gamma_base + np.random.normal(0, 0.05)
        t_value = t_base + np.random.normal(0, 0.05)
        cognitive_score = 25 + 3 * t_value + np.random.normal(0, 2)

        subject_ids.append(f"SUB_{i:03d}")
        gamma_powers.append(gamma_power)
        t_values.append(t_value)
        cognitive_scores.append(cognitive_score)
        ages.append(age)

    data = pd.DataFrame({
        'subject_id': subject_ids,
        'group': groups,
        'APOE4_status': apoe4_status,
        'gamma_power': gamma_powers,
        'topological_complexity': t_values,
        'cognitive_score': cognitive_scores,
        'age': ages
    })
    return {'clinical_data': data}


if __name__ == "__main__":
    dataset = generate_simulated_dataset()
    df = dataset['clinical_data']
    print(f"Generated {len(df)} subjects")
    print(f"APOE4 carriers: {df['APOE4_status'].sum()} ({df['APOE4_status'].mean():.1%})")
