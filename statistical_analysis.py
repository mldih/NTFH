import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    def __init__(self, config):
        self.config = config
        logger.info("Initialized statistical analyzer")

    def mediation_analysis(self, data, independent_var, mediator, dependent_var,
                            n_bootstraps=None, confidence_level=None):
        if n_bootstraps is None:
            n_bootstraps = self.config['mediation']['n_bootstraps']
        if confidence_level is None:
            confidence_level = self.config['mediation']['confidence_level']

        logger.info(f"Mediation analysis: {independent_var} -> {mediator} -> {dependent_var}")
        df = data['clinical_data'].copy()
        X = df[independent_var].values
        M = df[mediator].values
        Y = df[dependent_var].values

        X_with_const = sm.add_constant(X)
        model_c = sm.OLS(Y, X_with_const).fit()
        c_effect = model_c.params[1]
        c_pvalue = model_c.pvalues[1]

        model_a = sm.OLS(M, X_with_const).fit()
        a_effect = model_a.params[1]
        a_pvalue = model_a.pvalues[1]

        MX_with_const = sm.add_constant(np.column_stack([M, X]))
        model_b = sm.OLS(Y, MX_with_const).fit()
        b_effect = model_b.params[1]
        b_pvalue = model_b.pvalues[1]
        c_prime_effect = model_b.params[2]

        indirect_effect = a_effect * b_effect
        total_effect = c_effect
        mediation_proportion = indirect_effect / total_effect if total_effect != 0 else 0

        bootstrap_indirect_effects = []
        bootstrap_direct_effects = []
        bootstrap_total_effects = []
        n_samples = len(X)

        for _ in tqdm(range(n_bootstraps), desc="Bootstrap"):
            indices = resample(range(n_samples))
            X_boot = X[indices]
            M_boot = M[indices]
            Y_boot = Y[indices]
            try:
                X_boot_const = sm.add_constant(X_boot)
                model_a_boot = sm.OLS(M_boot, X_boot_const).fit()
                a_boot = model_a_boot.params[1]
                MX_boot_const = sm.add_constant(np.column_stack([M_boot, X_boot]))
                model_b_boot = sm.OLS(Y_boot, MX_boot_const).fit()
                b_boot = model_b_boot.params[1]
                c_prime_boot = model_b_boot.params[2]
                model_c_boot = sm.OLS(Y_boot, X_boot_const).fit()
                c_boot = model_c_boot.params[1]
                bootstrap_indirect_effects.append(a_boot * b_boot)
                bootstrap_direct_effects.append(c_prime_boot)
                bootstrap_total_effects.append(c_boot)
            except Exception:
                continue

        alpha = 1 - confidence_level
        indirect_ci_lower = np.percentile(bootstrap_indirect_effects, 100 * alpha / 2)
        indirect_ci_upper = np.percentile(bootstrap_indirect_effects, 100 * (1 - alpha / 2))
        direct_ci_lower = np.percentile(bootstrap_direct_effects, 100 * alpha / 2)
        direct_ci_upper = np.percentile(bootstrap_direct_effects, 100 * (1 - alpha / 2))
        total_ci_lower = np.percentile(bootstrap_total_effects, 100 * alpha / 2)
        total_ci_upper = np.percentile(bootstrap_total_effects, 100 * (1 - alpha / 2))

        indirect_pvalue = np.mean(np.array(bootstrap_indirect_effects) <= 0)
        indirect_pvalue = 2 * min(indirect_pvalue, 1 - indirect_pvalue)

        results = {
            'paths': {
                'a': {'effect': a_effect, 'pvalue': a_pvalue},
                'b': {'effect': b_effect, 'pvalue': b_pvalue},
                'c': {'effect': c_effect, 'pvalue': c_pvalue},
                "c'": {'effect': c_prime_effect, 'pvalue': model_b.pvalues[2]}
            },
            'effects': {
                'total_effect': total_effect,
                'direct_effect': c_prime_effect,
                'indirect_effect': indirect_effect,
                'mediation_proportion': mediation_proportion
            },
            'confidence_intervals': {
                'indirect_effect': (indirect_ci_lower, indirect_ci_upper),
                'direct_effect': (direct_ci_lower, direct_ci_upper),
                'total_effect': (total_ci_lower, total_ci_upper)
            },
            'significance': {
                'indirect_pvalue': indirect_pvalue,
                'mediation_significant': indirect_pvalue < alpha
            },
            'models': {
                'model_a': model_a,
                'model_b': model_b,
                'model_c': model_c
            },
            'bootstrap_results': {
                'indirect_effects': bootstrap_indirect_effects,
                'direct_effects': bootstrap_direct_effects,
                'total_effects': bootstrap_total_effects
            }
        }
        logger.info(f"Mediation analysis completed: Indirect effect={indirect_effect:.3f} "
                    f"({indirect_ci_lower:.3f}, {indirect_ci_upper:.3f}), "
                    f"p={indirect_pvalue:.4f}, Mediation proportion={mediation_proportion:.1%}")
        return results

    def group_comparisons(self, data, group_var='group', alpha=None):
        if alpha is None:
            alpha = self.config['group_comparisons']['alpha']
        logger.info(f"Group comparisons: {group_var}")
        df = data['clinical_data'].copy()
        groups = df[group_var].unique()
        variables = ['gamma_power', 'topological_complexity', 'v1_atrophy_rate', 'cognitive_score']
        results = {}

        for var in variables:
            if var not in df.columns:
                continue
            logger.info(f"Analyzing variable: {var}")
            group_data = [df[df[group_var] == group][var].values for group in groups]

            levene_stat, levene_p = stats.levene(*group_data)

            if levene_p > alpha:
                anova_stat, anova_p = stats.f_oneway(*group_data)
                test_used = 'ANOVA'
                if anova_p < alpha:
                    tukey_results = pairwise_tukeyhsd(
                        df[var].values, df[group_var].values, alpha=alpha
                    )
                    posthoc = {
                        'method': 'Tukey HSD',
                        'results': tukey_results,
                        'significant_pairs': [
                            (groups[i], groups[j])
                            for i, j in zip(range(len(tukey_results.reject)), range(len(tukey_results.reject)))
                            if tukey_results.reject[i]
                        ]
                    }
                else:
                    posthoc = None
            else:
                kruskal_stat, kruskal_p = stats.kruskal(*group_data)
                anova_stat, anova_p = kruskal_stat, kruskal_p
                test_used = 'Kruskal-Wallis'
                if kruskal_p < alpha:
                    posthoc = {
                        'method': 'Mann-Whitney (Bonferroni)',
                        'results': [],
                        'significant_pairs': []
                    }
                    n_comparisons = len(groups) * (len(groups) - 1) // 2
                    bonferroni_alpha = alpha / n_comparisons
                    for i in range(len(groups)):
                        for j in range(i + 1, len(groups)):
                            mw_stat, mw_p = stats.mannwhitneyu(
                                group_data[i], group_data[j], alternative='two-sided'
                            )
                            posthoc['results'].append({
                                'groups': (groups[i], groups[j]),
                                'statistic': mw_stat,
                                'pvalue': mw_p,
                                'significant': mw_p < bonferroni_alpha
                            })
                            if mw_p < bonferroni_alpha:
                                posthoc['significant_pairs'].append((groups[i], groups[j]))
                else:
                    posthoc = None

            effect_sizes = {}
            for i, group1 in enumerate(groups):
                for j, group2 in enumerate(groups):
                    if i < j:
                        if test_used == 'ANOVA':
                            mean1, mean2 = np.mean(group_data[i]), np.mean(group_data[j])
                            std1, std2 = np.std(group_data[i], ddof=1), np.std(group_data[j], ddof=1)
                            n1, n2 = len(group_data[i]), len(group_data[j])
                            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
                            cohens_d = (mean1 - mean2) / pooled_std
                            effect_sizes[f"{group1}_vs_{group2}"] = float(cohens_d)
                        else:
                            mw_stat, _ = stats.mannwhitneyu(group_data[i], group_data[j])
                            n1, n2 = len(group_data[i]), len(group_data[j])
                            rank_biserial = 1 - (2 * mw_stat) / (n1 * n2)
                            effect_sizes[f"{group1}_vs_{group2}"] = float(rank_biserial)

            descriptive_stats = {}
            for group in groups:
                group_values = df[df[group_var] == group][var]
                descriptive_stats[group] = {
                    'n': int(len(group_values)),
                    'mean': float(np.mean(group_values)),
                    'std': float(np.std(group_values)),
                    'sem': float(stats.sem(group_values)),
                    'ci_95': [float(x) for x in stats.t.interval(
                        0.95, len(group_values) - 1,
                        loc=np.mean(group_values),
                        scale=stats.sem(group_values)
                    )]
                }

            results[var] = {
                'test_used': test_used,
                'test_statistic': float(anova_stat),
                'pvalue': float(anova_p),
                'significant': bool(anova_p < alpha),
                'variance_homogeneity': {
                    'levene_statistic': float(levene_stat),
                    'levene_pvalue': float(levene_p),
                    'homogeneous': bool(levene_p > alpha)
                },
                'posthoc_analysis': posthoc,
                'effect_sizes': effect_sizes,
                'descriptive_stats': descriptive_stats
            }
            logger.info(f"{var}: {test_used} F/H={anova_stat:.3f}, p={anova_p:.4f}, Significant={anova_p < alpha}")

        return results

    def correlation_analysis(self, data, variables=None, method='pearson'):
        df = data['clinical_data'].copy()
        if variables is None:
            variables = ['gamma_power', 'topological_complexity', 'v1_atrophy_rate',
                         'cognitive_score', 'age', 'APOE4_status']
            variables = [v for v in variables if v in df.columns]

        corr_matrix = df[variables].corr(method=method)
        pvalue_matrix = pd.DataFrame(
            np.zeros((len(variables), len(variables))),
            index=variables, columns=variables
        )

        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i <= j:
                    if method == 'pearson':
                        corr, pvalue = stats.pearsonr(df[var1], df[var2])
                    elif method == 'spearman':
                        corr, pvalue = stats.spearmanr(df[var1], df[var2])
                    elif method == 'kendall':
                        corr, pvalue = stats.kendalltau(df[var1], df[var2])
                    else:
                        corr, pvalue = stats.pearsonr(df[var1], df[var2])
                    pvalue_matrix.iloc[i, j] = pvalue
                    pvalue_matrix.iloc[j, i] = pvalue

        n_comparisons = len(variables) * (len(variables) - 1) // 2
        bonferroni_alpha = 0.05 / n_comparisons
        significance_matrix = pvalue_matrix < bonferroni_alpha

        results = {
            'correlation_matrix': corr_matrix,
            'pvalue_matrix': pvalue_matrix,
            'significance_matrix': significance_matrix,
            'method': method,
            'bonferroni_alpha': bonferroni_alpha,
            'n_comparisons': n_comparisons
        }
        logger.info(f"Correlation analysis completed: {method} correlation")
        return results

    def predictive_modeling(self, data, target_var='APOE4_status', test_size=0.3):
        logger.info(f"Predictive modeling: Predicting {target_var}")
        df = data['clinical_data'].copy()
        feature_vars = ['gamma_power', 'topological_complexity', 'v1_atrophy_rate', 'age']
        feature_vars = [v for v in feature_vars if v in df.columns]
        X = df[feature_vars].values
        y = df[target_var].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42)
        }

        results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
            model.fit(X_scaled, y)

            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                importances = None

            results[name] = {
                'model': model,
                'cv_auc_scores': cv_scores,
                'cv_auc_mean': float(np.mean(cv_scores)),
                'cv_auc_std': float(np.std(cv_scores)),
                'feature_importances': importances.tolist() if importances is not None else None,
                'feature_names': feature_vars
            }
            logger.info(f"{name}: AUC = {np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f}")

        best_model_name = max(results.keys(), key=lambda x: results[x]['cv_auc_mean'])
        best_model = results[best_model_name]
        logger.info(f"Best model: {best_model_name} (AUC = {best_model['cv_auc_mean']:.3f})")

        return {
            'all_models': results,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'features': feature_vars,
            'target': target_var
        }
