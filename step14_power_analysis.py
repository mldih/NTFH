import numpy as np
from statsmodels.stats.power import TTestIndPower
from scipy import stats


class StatisticalPowerValidation:
    def __init__(self):
        self.power_results = {}

    def calculate_statistical_power(self, effect_size, alpha=0.05, power=0.8):
        analysis = TTestIndPower()
        required_n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1.0)
        return int(np.ceil(required_n))

    def validate_current_sample_sizes(self, dataset_sizes, expected_effect_sizes):
        print("== Statistical Power Validation ==")
        validation_results = {}
        for dataset, info in dataset_sizes.items():
            current_n = info['sample_size']
            effect_size = expected_effect_sizes[dataset]
            required_n = self.calculate_statistical_power(effect_size)
            power_achieved = self.calculate_achieved_power(current_n, effect_size)
            validation_results[dataset] = {
                'current_sample_size': current_n,
                'required_sample_size': required_n,
                'effect_size': effect_size,
                'power_achieved': power_achieved,
                'adequacy': 'Adequate' if current_n >= required_n else 'Inadequate',
                'recommendation': f"{'Maintain' if current_n >= required_n else 'Increase'} sample size"
            }
            print(f"{dataset}: Current n={current_n}, Required n={required_n}, "
                  f"Power={power_achieved:.3f}, {validation_results[dataset]['adequacy']}")
        self.power_results = validation_results
        return validation_results

    def calculate_achieved_power(self, sample_size, effect_size, alpha=0.05):
        analysis = TTestIndPower()
        power = analysis.power(effect_size=effect_size, nobs1=sample_size, alpha=alpha, ratio=1.0)
        return float(power)

    def bootstrap_power_analysis(self, data, n_bootstrap=1000):
        print("\n=== Bootstrap Power Analysis ===")
        groups = data['group'].unique()
        power_estimates = {}
        for i, group1 in enumerate(groups):
            for group2 in groups[i + 1:]:
                group1_data = data[data['group'] == group1]['gamma_power'].values
                group2_data = data[data['group'] == group2]['gamma_power'].values
                significant_count = 0
                effect_sizes = []
                for _ in range(n_bootstrap):
                    boot1 = np.random.choice(group1_data, size=len(group1_data), replace=True)
                    boot2 = np.random.choice(group2_data, size=len(group2_data), replace=True)
                    t_stat, p_value = stats.ttest_ind(boot1, boot2)
                    if p_value < 0.05:
                        significant_count += 1
                    cohens_d = (np.mean(boot1) - np.mean(boot2)) / np.sqrt((np.std(boot1)**2 + np.std(boot2)**2) / 2)
                    effect_sizes.append(cohens_d)
                empirical_power = significant_count / n_bootstrap
                mean_effect_size = float(np.mean(effect_sizes))
                power_estimates[f"{group1}_vs_{group2}"] = {
                    'empirical_power': float(empirical_power),
                    'mean_effect_size': mean_effect_size,
                    'n_bootstrap': n_bootstrap
                }
                print(f"{group1} vs {group2}: Power={empirical_power:.3f}, Effect size={mean_effect_size:.3f}")
        return power_estimates


if __name__ == "__main__":
    from step15_simulated_data import generate_simulated_dataset
    dataset = generate_simulated_dataset()
    df = dataset['clinical_data']

    validator = StatisticalPowerValidation()
    dataset_sizes = {
        'ADNI': {'sample_size': 1200},
        'PPMI': {'sample_size': 420},
        'PUMCH': {'sample_size': 1650}
    }
    expected_effect_sizes = {'ADNI': 0.5, 'PPMI': 0.4, 'PUMCH': 0.3}
    validator.validate_current_sample_sizes(dataset_sizes, expected_effect_sizes)
    validator.bootstrap_power_analysis(df, n_bootstrap=500)
