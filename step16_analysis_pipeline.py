import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def run_complete_analysis(dataset):
    data = dataset['clinical_data']
    results = {}

    print("=" * 50)
    print("Group Comparisons")
    print("=" * 50)
    apoe4_data = data[data['APOE4_status'] == 1]
    control_data = data[data['APOE4_status'] == 0]

    for var in ['gamma_power', 'topological_complexity', 'cognitive_score']:
        t_stat, p_value = stats.ttest_ind(control_data[var], apoe4_data[var])
        mean_diff = control_data[var].mean() - apoe4_data[var].mean()
        pooled_std = np.sqrt((control_data[var].std()**2 + apoe4_data[var].std()**2) / 2)
        cohens_d = mean_diff / pooled_std
        results[f'{var}_t-test'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'control_mean': float(control_data[var].mean()),
            'apoe4_mean': float(apoe4_data[var].mean()),
            'mean_diff': float(mean_diff)
        }
        print(f"{var}:")
        print(f"  Control: {control_data[var].mean():.3f} +/- {control_data[var].std():.3f}")
        print(f"  APOE4:   {apoe4_data[var].mean():.3f} +/- {apoe4_data[var].std():.3f}")
        print(f"  Difference: {mean_diff:.3f} (Cohen's d = {cohens_d:.3f})")
        print(f"  t-test: t = {t_stat:.3f}, p = {p_value:.4f}")
        print()

    print("=" * 50)
    print("Mediation Effect Analysis")
    print("=" * 50)
    X = data['APOE4_status'].values
    M = data['topological_complexity'].values
    Y = data['cognitive_score'].values

    X_with_const = sm.add_constant(X)
    model_c = sm.OLS(Y, X_with_const).fit()
    c_effect = model_c.params[1]

    model_a = sm.OLS(M, X_with_const).fit()
    a_effect = model_a.params[1]

    MX_with_const = sm.add_constant(np.column_stack([M, X]))
    model_b = sm.OLS(Y, MX_with_const).fit()
    b_effect = model_b.params[1]
    c_prime_effect = model_b.params[2]

    indirect_effect = a_effect * b_effect
    total_effect = c_effect
    mediation_proportion = indirect_effect / total_effect if total_effect != 0 else 0

    print(f"Path a (APOE4 -> T-value): {a_effect:.3f}")
    print(f"Path b (T-value -> cognition): {b_effect:.3f}")
    print(f"Path c (total effect): {c_effect:.3f}")
    print(f"Path c' (direct effect): {c_prime_effect:.3f}")
    print(f"Indirect effect: {indirect_effect:.3f}")
    print(f"Mediation proportion: {mediation_proportion:.1%}")

    results['mediation'] = {
        'a': float(a_effect),
        'b': float(b_effect),
        'c': float(c_effect),
        'c_prime': float(c_prime_effect),
        'indirect': float(indirect_effect),
        'mediation_proportion': float(mediation_proportion)
    }
    return results


if __name__ == "__main__":
    from step15_simulated_data import generate_simulated_dataset
    dataset = generate_simulated_dataset()
    results = run_complete_analysis(dataset)
