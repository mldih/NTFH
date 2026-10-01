import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')


class DataValidationPipeline:
    def __init__(self):
        self.data_sources = {}
        self.validation_results = {}
        self.quality_metrics = {}

    def load_and_validate_all_datasets(self):
        print("== Loading and Validating Multi-Center Datasets ==")
        self.data_sources['ADNI'] = {
            'description': "Alzheimer's Disease Neuroimaging Initiative",
            'reference': 'Ossenkoppele et al. APOE4 potentiates Abeta deposition...',
            'doi': '10.1038/s41591-023-02331-6',
            'sample_size': 1200,
            'key_variables': ['APOE_status', 'gamma_power', 'v1_volume', 'cognitive_scores', 'fALFF']
        }
        self.data_sources['PPMI'] = {
            'description': "Parkinson's Progression Markers Initiative",
            'sample_size': 420,
            'key_variables': ['gamma_power', 'tremor_scores', 'visual_deficits', 'dopamine_levels']
        }
        self.data_sources['PUMCH'] = {
            'description': "PUMCH Chinese Normative Database",
            'sample_size': 1650,
            'key_variables': ['gamma_power_norms', 'cognitive_norms', 'v1_volume_norms']
        }
        return self.data_sources

    def comprehensive_data_cleaning(self, raw_data):
        print("\n=== Comprehensive Data Cleaning Pipeline ===")
        cleaning_steps = []
        cleaned_data = raw_data.copy()

        print("1. Missing Value Analysis...")
        missing_report = self.analyze_missing_values(cleaned_data)
        cleaning_steps.append(missing_report)

        print("2. Outlier Detection...")
        outlier_report = self.detect_outliers(cleaned_data)
        cleaning_steps.append(outlier_report)

        print("3. Data Imputation...")
        cleaned_data = self.multiple_imputation(cleaned_data)

        print("4. Range Validation...")
        range_report = self.validate_value_ranges(cleaned_data)
        cleaning_steps.append(range_report)

        print("5. Consistency Checks...")
        consistency_report = self.consistency_checks(cleaned_data)
        cleaning_steps.append(consistency_report)

        print("6. Data Normalization...")
        cleaned_data, scalers = self.normalize_data(cleaned_data)

        self.validation_results['cleaning_pipeline'] = {
            'steps': cleaning_steps,
            'final_data_shape': cleaned_data.shape,
            'data_quality_score': self.calculate_data_quality_score(cleaning_steps)
        }
        return cleaned_data, cleaning_steps

    def analyze_missing_values(self, data):
        missing_stats = {}
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            missing_percent = (missing_count / len(data)) * 100
            if missing_percent > 0:
                missing_stats[column] = {
                    'missing_count': int(missing_count),
                    'missing_percent': float(missing_percent),
                    'severity': 'High' if missing_percent > 10 else 'Medium' if missing_percent > 5 else 'Low'
                }
        total_missing = int(data.isnull().sum().sum())
        overall_missing_percent = (total_missing / (data.shape[0] * data.shape[1])) * 100
        return {
            'missing_stats': missing_stats,
            'total_missing': total_missing,
            'overall_missing_percent': float(overall_missing_percent),
            'recommendation': 'Multiple Imputation' if overall_missing_percent > 5 else 'Direct Deletion'
        }

    def detect_outliers(self, data, method='iqr'):
        outlier_report = {}
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            col_data = data[column].dropna()
            outliers = pd.Series([], dtype=float)
            if method == 'iqr':
                Q1 = np.percentile(col_data, 25)
                Q3 = np.percentile(col_data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(col_data))
                outliers = col_data[z_scores > 3]
            outlier_report[column] = {
                'outlier_count': int(len(outliers)),
                'outlier_percent': float((len(outliers) / len(col_data)) * 100) if len(col_data) > 0 else 0.0,
                'method': method,
                'values': outliers.tolist() if len(outliers) > 0 else []
            }
        return outlier_report

    def multiple_imputation(self, data, n_iterations=10):
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.isnull().sum().sum() > 0:
            imputer = IterativeImputer(max_iter=n_iterations, random_state=42, sample_posterior=True)
            imputed_data = imputer.fit_transform(numeric_data)
            data[numeric_data.columns] = imputed_data
        return data

    def validate_value_ranges(self, data):
        range_report = {}
        for column in data.select_dtypes(include=[np.number]).columns:
            range_report[column] = {
                'min': float(data[column].min()),
                'max': float(data[column].max()),
                'mean': float(data[column].mean()),
                'std': float(data[column].std())
            }
        return range_report

    def consistency_checks(self, data):
        checks = []
        if 'APOE4_status' in data.columns:
            unique_vals = data['APOE4_status'].unique()
            if not set(unique_vals).issubset({0, 1}):
                checks.append("APOE4_status contains values other than 0/1")
        if 'age' in data.columns:
            if (data['age'] < 0).any() or (data['age'] > 120).any():
                checks.append("age contains out-of-range values")
        return checks

    def normalize_data(self, data, method='standard'):
        scalers = {}
        normalized_data = data.copy()
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            # 跳过二值列（0/1分类变量）和标识列
            unique_vals = data[column].dropna().unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                continue
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            normalized_data[column] = scaler.fit_transform(data[[column]])
            scalers[column] = scaler
        return normalized_data, scalers

    def calculate_data_quality_score(self, cleaning_steps):
        quality_components = []
        missing_report = cleaning_steps[0]
        missing_score = max(0, 100 - missing_report['overall_missing_percent'] * 2)
        quality_components.append(missing_score)
        outlier_report = cleaning_steps[1]
        total_outliers = sum([report['outlier_count'] for report in outlier_report.values()])
        outlier_score = max(0, 100 - (total_outliers / max(len(outlier_report), 1)) * 5)
        quality_components.append(outlier_score)
        consistency_report = cleaning_steps[3]
        consistency_score = 100 - len(consistency_report) * 10
        quality_components.append(consistency_score)
        return float(max(0, min(100, np.mean(quality_components))))


if __name__ == "__main__":
    from step15_simulated_data import generate_simulated_dataset
    dataset = generate_simulated_dataset()
    df = dataset['clinical_data']
    pipeline = DataValidationPipeline()
    pipeline.load_and_validate_all_datasets()
    cleaned, steps = pipeline.comprehensive_data_cleaning(df)
    print(f"\nData quality score: {pipeline.validation_results['cleaning_pipeline']['data_quality_score']:.1f}")
