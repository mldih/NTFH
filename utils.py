import yaml
import logging
import pickle
import gzip
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch


def setup_logging(output_dir, verbose=False):
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"analysis_{timestamp}.log"
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.info(f"Logging system initialized, log file: {log_file}")
    return logger


def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_results(results, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(file_path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger = logging.getLogger(__name__)
    logger.info(f"Analysis results saved to: {file_path}")
    summary_path = file_path.with_suffix('.json')
    summary = create_results_summary(results)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return file_path


def load_results(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Results file does not exist: {file_path}")
    with gzip.open(file_path, 'rb') as f:
        results = pickle.load(f)
    logger = logging.getLogger(__name__)
    logger.info(f"Analysis results loaded from: {file_path}")
    return results


def create_results_summary(results):
    summary = {
        'timestamp': datetime.now().isoformat(),
        'modules_completed': list(results.keys())
    }
    if 'topology' in results:
        topology = results['topology']
        if 'health_vs_pathology' in topology:
            comp = topology['health_vs_pathology']['comparison']
            summary['topology'] = {
                'complexity_reduction_percent': comp['complexity_reduction_percent'],
                'p_value': comp['p_value'],
                'effect_size': comp['effect_size']
            }
    if 'snn' in results:
        snn = results['snn']
        if 'training_results' in snn:
            training = snn['training_results']
            summary['snn'] = {
                'best_accuracy': max(training['test_accuracies']) if 'test_accuracies' in training else 'N/A'
            }
    if 'statistics' in results:
        stats = results['statistics']
        if 'mediation_analysis' in stats:
            mediation = stats['mediation_analysis']['effects']
            summary['statistics'] = {
                'mediation_proportion': mediation['mediation_proportion'],
                'indirect_effect': mediation['indirect_effect']
            }
    return summary


def set_random_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        import snntorch as snn
        # snntorch 使用 torch.manual_seed，不单独提供 set_random_seed
        # 已经在上方设置了 torch.manual_seed(seed)
    except ImportError:
        pass
    logger = logging.getLogger(__name__)
    logger.info(f"Random seeds set to: {seed}")


def format_pvalue(pvalue):
    if pvalue < 0.001:
        return "p < 0.001"
    elif pvalue < 0.01:
        return "p < 0.01"
    elif pvalue < 0.05:
        return "p < 0.05"
    else:
        return f"p = {pvalue:.3f}"


def calculate_effect_size(group1, group2):
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std
    if abs(cohens_d) < 0.2:
        interpretation = "Very small"
    elif abs(cohens_d) < 0.5:
        interpretation = "Small"
    elif abs(cohens_d) < 0.8:
        interpretation = "Medium"
    else:
        interpretation = "Large"
    return {
        'cohens_d': cohens_d,
        'interpretation': interpretation,
        'means': (mean1, mean2),
        'std_devs': (std1, std2)
    }


class Timer:
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        logger = logging.getLogger(__name__)
        logger.info(f"Starting {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = end_time - self.start_time
        logger = logging.getLogger(__name__)
        logger.info(f"{self.name} completed, duration: {duration}")
