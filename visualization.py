import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import logging

logger = logging.getLogger(__name__)


class ResultsVisualizer:
    def __init__(self, config):
        self.config = config
        self.set_plot_style()
        logger.info("Initialized results visualizer")

    def set_plot_style(self):
        style = self.config.get('style', 'seaborn')
        palette = self.config.get('color_palette', 'viridis')
        try:
            if style == 'seaborn':
                plt.style.use('seaborn-v0_8-whitegrid')
            elif style == 'ggplot':
                plt.style.use('ggplot')
            else:
                plt.style.use('default')
        except Exception:
            plt.style.use('default')
        try:
            sns.set_palette(palette)
        except Exception:
            pass
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass
        plt.rcParams['figure.dpi'] = self.config.get('dpi', 300)
        plt.rcParams['savefig.dpi'] = self.config.get('dpi', 300)

    def create_all_figures(self, results):
        figures = {}
        if 'topology' in results:
            figures.update(self.create_topology_figures(results['topology']))
        if 'snn' in results:
            figures.update(self.create_snn_figures(results['snn']))
        if 'statistics' in results:
            figures.update(self.create_statistics_figures(results['statistics']))
        figures.update(self.create_summary_figures(results))
        logger.info(f"Created {len(figures)} figures")
        return figures

    def create_topology_figures(self, topology_results):
        figures = {}
        if 'health_vs_pathology' in topology_results:
            figures['manifold_comparison'] = self.plot_manifold_comparison(
                topology_results['health_vs_pathology'])
            figures['persistence_diagram_comparison'] = self.plot_persistence_diagram_comparison(
                topology_results['health_vs_pathology'])
        if 'gamma_correlation' in topology_results:
            figures['gamma_topology_correlation'] = self.plot_gamma_topology_correlation(
                topology_results['gamma_correlation'])
        return figures

    def create_snn_figures(self, snn_results):
        figures = {}
        if 'training_results' in snn_results:
            figures['training_curves'] = self.plot_training_curves(snn_results['training_results'])
        if 'pathology_results' in snn_results:
            figures['pathology_effects'] = self.plot_pathology_effects(snn_results['pathology_results'])
        if 'model' in snn_results:
            figures['snn_dynamics'] = self.plot_snn_dynamics(snn_results['model'])
        return figures

    def create_statistics_figures(self, statistics_results):
        figures = {}
        if 'mediation_analysis' in statistics_results:
            figures['mediation_analysis'] = self.plot_mediation_analysis(
                statistics_results['mediation_analysis'])
        if 'group_comparisons' in statistics_results:
            figures['group_comparisons'] = self.plot_group_comparisons(
                statistics_results['group_comparisons'])
        return figures

    def create_summary_figures(self, results):
        figures = {}
        figures['theoretical_framework'] = self.plot_theoretical_framework()
        figures['pathological_cascade'] = self.plot_pathological_cascade()
        return figures

    def plot_manifold_comparison(self, topology_results):
        healthy = topology_results['healthy']
        pathological = topology_results['pathological']
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        healthy_manifold = healthy['manifold']
        if healthy_manifold.shape[1] >= 2:
            axes[0, 0].scatter(healthy_manifold[:, 0], healthy_manifold[:, 1],
                               alpha=0.6, s=20, c='blue', label='Healthy')
        axes[0, 0].set_title('Healthy State Neural Manifold')
        axes[0, 0].set_xlabel('Component 1')
        axes[0, 0].set_ylabel('Component 2')
        axes[0, 0].legend()

        pathological_manifold = pathological['manifold']
        if pathological_manifold.shape[1] >= 2:
            axes[0, 1].scatter(pathological_manifold[:, 0], pathological_manifold[:, 1],
                               alpha=0.6, s=20, c='red', label='Pathological')
        axes[0, 1].set_title('Pathological State Neural Manifold')
        axes[0, 1].set_xlabel('Component 1')
        axes[0, 1].set_ylabel('Component 2')
        axes[0, 1].legend()

        complexity_data = {
            'Healthy': healthy['complexity']['total_complexity'],
            'Pathological': pathological['complexity']['total_complexity']
        }
        axes[1, 0].bar(complexity_data.keys(), complexity_data.values(),
                       color=['blue', 'red'], alpha=0.7)
        axes[1, 0].set_title('Topological Complexity Comparison')
        axes[1, 0].set_ylabel('Topological Complexity Index')

        p_value = topology_results['comparison']['p_value']
        y_max = max(complexity_data.values())
        axes[1, 0].text(0.5, y_max * 1.1, f'p = {p_value:.4f}',
                        ha='center', va='bottom', fontsize=12)
        if p_value < 0.05:
            axes[1, 0].plot([0, 1], [y_max * 1.05, y_max * 1.05], 'k-', lw=1)
            axes[1, 0].text(0.5, y_max * 1.15, '*', ha='center', va='bottom', fontsize=16)

        healthy_betti = [healthy['complexity'].get(f'dim_{dim}', {}).get('betti_number', 0) for dim in [0, 1, 2]]
        pathological_betti = [pathological['complexity'].get(f'dim_{dim}', {}).get('betti_number', 0) for dim in [0, 1, 2]]
        x = np.arange(3)
        width = 0.35
        axes[1, 1].bar(x - width / 2, healthy_betti, width, label='Healthy', color='blue', alpha=0.7)
        axes[1, 1].bar(x + width / 2, pathological_betti, width, label='Pathological', color='red', alpha=0.7)
        axes[1, 1].set_title('Betti Number Comparison')
        axes[1, 1].set_xlabel('Homology Dimension')
        axes[1, 1].set_ylabel('Betti Number')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(['H0', 'H1', 'H2'])
        axes[1, 1].legend()

        plt.tight_layout()
        return fig

    def plot_gamma_topology_correlation(self, correlation_results):
        fig, ax = plt.subplots(figsize=(8, 6))
        gamma_power = correlation_results['gamma_power']
        topological_complexity = correlation_results['topological_complexity']
        scatter = ax.scatter(gamma_power, topological_complexity, alpha=0.6,
                             c=gamma_power, cmap='viridis', s=50)
        z = np.polyfit(gamma_power, topological_complexity, 1)
        p = np.poly1d(z)
        ax.plot(gamma_power, p(gamma_power), "r--", alpha=0.8, linewidth=2)
        corr_coef = correlation_results['correlation_coefficient']
        p_value = correlation_results['p_value']
        ax.text(0.05, 0.95, f'r = {corr_coef:.3f}\np = {p_value:.4f}',
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_xlabel('Gamma Oscillation Power')
        ax.set_ylabel('Topological Complexity')
        ax.set_title('Correlation between Gamma Power and Topological Complexity')
        plt.colorbar(scatter, label='Gamma Power')
        plt.tight_layout()
        return fig

    def plot_persistence_diagram_comparison(self, topology_results):
        from gtda.plotting import plot_diagram
        healthy = topology_results['healthy']
        pathological = topology_results['pathological']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        try:
            plot_diagram(healthy['persistence'], ax=ax1)
            ax1.set_title('Healthy State Persistence Diagram')
            plot_diagram(pathological['persistence'], ax=ax2)
            ax2.set_title('Pathological State Persistence Diagram')
        except Exception as e:
            logger.warning(f"plot_diagram failed: {e}")
        plt.tight_layout()
        return fig

    def plot_training_curves(self, training_results):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        epochs = range(1, len(training_results['train_losses']) + 1)

        ax1.plot(epochs, training_results['train_losses'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, training_results['test_losses'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, training_results['train_accuracies'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, training_results['test_accuracies'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3.plot(epochs, training_results['classification_losses'], 'g-', label='Classification Loss', linewidth=2)
        ax3.plot(epochs, training_results['topology_losses'], 'm-', label='Topology Loss', linewidth=2)
        ax3.plot(epochs, training_results['sparsity_losses'], 'c-', label='Sparsity Loss', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Loss Components')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        best_epoch = np.argmax(training_results['test_accuracies']) + 1
        best_accuracy = np.max(training_results['test_accuracies'])
        ax4.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best epoch: {best_epoch}')
        ax4.plot(epochs, training_results['test_accuracies'], 'r-', linewidth=2, label='Validation Accuracy')
        ax4.plot(best_epoch, best_accuracy, 'ro', markersize=8, label=f'Best Accuracy: {best_accuracy:.2f}%')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Best Model Selection')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_pathology_effects(self, pathology_results):
        models = list(pathology_results.keys())
        accuracies = [pathology_results[m]['accuracy'] for m in models]
        reductions = [pathology_results[m]['relative_reduction'] for m in models]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        bars = ax1.bar(models, accuracies, color=['green', 'red', 'orange', 'blue', 'purple'][:len(models)])
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy in Different Model States')
        ax1.tick_params(axis='x', rotation=45)
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                     f'{acc:.1f}%', ha='center', va='bottom')

        colors = ['green' if r == 0 else 'red' if 'APOE4' in m and 'intervention' not in m else 'blue'
                  for m, r in zip(models, reductions)]
        bars2 = ax2.bar(models, reductions, color=colors)
        ax2.set_ylabel('Accuracy Reduction (%)')
        ax2.set_title('Accuracy Reduction Relative to Healthy State')
        ax2.tick_params(axis='x', rotation=45)
        for bar, red in zip(bars2, reductions):
            if red > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                         f'-{red:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        return fig

    def plot_mediation_analysis(self, mediation_results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        paths = mediation_results['paths']
        effects = mediation_results['effects']

        ax1.axis('off')
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.text(2, 7, 'APOE4', ha='center', va='center', fontsize=14,
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightblue"))
        ax1.text(5, 5, 'Gamma Power', ha='center', va='center', fontsize=14,
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen"))
        ax1.text(8, 3, 'Topological\nComplexity', ha='center', va='center', fontsize=14,
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightcoral"))

        ax1.arrow(3, 7, 1.5, -1.5, head_width=0.2, head_length=0.2, fc='k', ec='k')
        ax1.text(4, 6.5, f"a={paths['a']['effect']:.3f}\n(p={paths['a']['pvalue']:.3f})",
                 ha='center', va='center', fontsize=10)
        ax1.arrow(6, 4.5, 1.5, -1, head_width=0.2, head_length=0.2, fc='k', ec='k')
        ax1.text(7.2, 4, f"b={paths['b']['effect']:.3f}\n(p={paths['b']['pvalue']:.3f})",
                 ha='center', va='center', fontsize=10)
        ax1.arrow(3, 7, 4.5, -3.5, head_width=0.2, head_length=0.2, fc='r', ec='r',
                  linestyle='--', alpha=0.7)
        ax1.text(5, 5.5, "c'={:.3f}\n(p={:.3f})".format(paths["c'"]['effect'], paths["c'"]['pvalue']),
                 ha='center', va='center', fontsize=10, color='red')
        ax1.set_title('Mediation Effect Path Diagram')

        effect_types = ['Total Effect', 'Direct Effect', 'Indirect Effect']
        effect_values = [effects['total_effect'], effects['direct_effect'], effects['indirect_effect']]
        ci = mediation_results['confidence_intervals']
        effect_errors = [
            (effects['total_effect'] - ci['total_effect'][0], ci['total_effect'][1] - effects['total_effect']),
            (effects['direct_effect'] - ci['direct_effect'][0], ci['direct_effect'][1] - effects['direct_effect']),
            (effects['indirect_effect'] - ci['indirect_effect'][0], ci['indirect_effect'][1] - effects['indirect_effect'])
        ]
        y_pos = np.arange(len(effect_types))
        bars = ax2.barh(y_pos, effect_values, xerr=np.array(effect_errors).T,
                        color=['lightblue', 'lightcoral', 'lightgreen'], alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(effect_types)
        ax2.set_xlabel('Effect Size')
        ax2.set_title('Mediation Effect Decomposition')
        for bar, val in zip(bars, effect_values):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{val:.3f}', ha='left', va='center')
        mediation_prop = effects['mediation_proportion']
        ax2.text(0.5, 0.95, f'Mediation Proportion: {mediation_prop:.1%}',
                 transform=ax2.transAxes, ha='center', va='top',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        plt.tight_layout()
        return fig

    def plot_group_comparisons(self, group_results):
        variables = list(group_results.keys())
        n_vars = len(variables)
        fig, axes = plt.subplots(2, (n_vars + 1) // 2, figsize=(15, 8))
        axes = np.array(axes).flatten()

        for i, var in enumerate(variables):
            if i >= len(axes):
                break
            ax = axes[i]
            stats_data = group_results[var]
            descriptive_stats = stats_data['descriptive_stats']
            groups = list(descriptive_stats.keys())
            means = [descriptive_stats[g]['mean'] for g in groups]
            sems = [descriptive_stats[g]['sem'] for g in groups]

            ax.bar(groups, means, yerr=sems, capsize=5, alpha=0.7,
                   color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'][:len(groups)])
            ax.set_ylabel(var)
            ax.set_title(f'{var} - Group Comparison')

            if stats_data.get('posthoc_analysis') and stats_data['posthoc_analysis'].get('significant_pairs'):
                y_max = max(means) + max(sems) + 0.1 * max(means)
                for j, (group1, group2) in enumerate(stats_data['posthoc_analysis']['significant_pairs']):
                    idx1 = groups.index(group1)
                    idx2 = groups.index(group2)
                    ax.plot([idx1, idx1, idx2, idx2], [y_max, y_max + 0.5, y_max + 0.5, y_max], 'k-', lw=1)
                    ax.text((idx1 + idx2) / 2, y_max + 0.7, '*', ha='center', va='bottom', fontsize=16)
                    y_max += 1.0

        for i in range(n_vars, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_theoretical_framework(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        ax.set_ylim(0, 10)
        ax.set_xlim(0, 10)
        ax.text(5, 9.5, 'Neural Topological Field Theory Theoretical Framework',
                ha='center', va='center', fontsize=14, weight='bold')

        concepts = [
            (2, 8, 'Gamma Oscillation\nMicrocolumns', 'lightblue'),
            (5, 8, 'Neural Manifold\nGeometry', 'lightgreen'),
            (8, 8, 'Topological\nComplexity', 'lightcoral'),
            (2, 6, 'Dual-scale\nEncoding', 'lightyellow'),
            (8, 6, 'Persistent\nHomology', 'lightpink'),
            (5, 4, 'Topological\nPhase Transition', 'orange'),
            (2, 2, 'APOE4\nPathology', 'red'),
            (8, 2, 'Geometric\nBiomarkers', 'purple')
        ]
        for x, y, text, color in concepts:
            ax.text(x, y, text, ha='center', va='center', fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.5", fc=color, ec="black", lw=1))

        connections = [
            ((2, 8), (5, 8), 'Implements'),
            ((5, 8), (8, 8), 'Quantifies'),
            ((2, 8), (2, 6), 'Mechanism'),
            ((8, 8), (8, 6), 'Tool'),
            ((5, 8), (5, 4), 'Dynamics'),
            ((5, 4), (2, 2), 'Drives'),
            ((5, 4), (8, 2), 'Provides')
        ]
        for (x1, y1), (x2, y2), label in connections:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, label, ha='center', va='center',
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        plt.tight_layout()
        return fig

    def plot_pathological_cascade(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.text(6, 7.5, 'APOE4-Driven Pathological Cascade',
                ha='center', va='center', fontsize=14, weight='bold')

        steps = [
            (1, 6, 'APOE4\nGenotype', 'red'),
            (3, 6, 'GABAergic\nInhibition\nEnhancement', 'orange'),
            (5, 6, 'E/I Balance\nDisruption', 'yellow'),
            (7, 6, 'Gamma Oscillation\nAttenuation', 'lightgreen'),
            (9, 6, 'Topological\nPhase Transition', 'lightblue'),
            (11, 6, 'Cognitive\nFunction\nImpairment', 'purple')
        ]
        for x, y, text, color in steps:
            ax.text(x, y, text, ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", fc=color, ec="black", lw=1.5))

        for i in range(len(steps) - 1):
            x1, y1, _, _ = steps[i]
            x2, y2, _, _ = steps[i + 1]
            ax.annotate("", xy=(x2 - 0.5, y2), xytext=(x1 + 0.5, y1),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2))

        interventions = [
            (3, 4, 'GABA\nModulators', 'blue'),
            (7, 4, 'gamma-tACS', 'green'),
            (9, 4, 'Topology-Targeted\nInterventions', 'cyan')
        ]
        for x, y, text, color in interventions:
            ax.text(x, y, text, ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="blue", lw=1))
            ax.annotate("", xy=(x, y + 0.3), xytext=(x, 6 - 0.3),
                        arrowprops=dict(arrowstyle="->", color="blue", lw=1, linestyle='--'))

        ax.plot([0.5, 11.5], [2, 2], 'k-', lw=2)
        for i, label in enumerate(['Early', 'Middle', 'Late']):
            x = 2 + i * 4
            ax.plot([x, x], [1.8, 2.2], 'k-', lw=1)
            ax.text(x, 1.5, label, ha='center', va='center', fontsize=10)

        plt.tight_layout()
        return fig

    def plot_snn_dynamics(self, snn_model, sample_input=None):
        import torch
        if sample_input is None:
            sample_input = torch.randn(1, snn_model.input_encoder[0].in_features)
        snn_model.eval()
        with torch.no_grad():
            outputs = snn_model(sample_input, num_steps=50)
        spikes = outputs['spikes'][0]
        membrane_potentials = outputs['membrane_potentials'][0]
        gamma_phases = outputs['gamma_phases']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        neuron_indices = np.where(spikes.cpu().numpy() > 0)
        ax1.scatter(neuron_indices[0], neuron_indices[1], s=1, c='black', alpha=0.6)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Neuron Index')
        ax1.set_title('Spike Raster Pattern')

        im = ax2.imshow(membrane_potentials.cpu().numpy().T, aspect='auto', cmap='viridis')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Neuron Index')
        ax2.set_title('Membrane Potential Dynamics')
        plt.colorbar(im, ax=ax2)

        for col in range(min(5, gamma_phases.shape[1])):
            ax3.plot(gamma_phases[:, col].cpu().numpy(), label=f'Microcolumn {col+1}')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Phase (radians)')
        ax3.set_title('Gamma Oscillation Phase Dynamics')
        ax3.legend()

        firing_rates = spikes.mean(dim=0).cpu().numpy()
        ax4.hist(firing_rates, bins=50, alpha=0.7, color='skyblue')
        ax4.set_xlabel('Average Firing Rate')
        ax4.set_ylabel('Number of Neurons')
        ax4.set_title('Firing Rate Distribution')

        plt.tight_layout()
        return fig

    def save_figures(self, figures, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            file_path = output_path / f"{name}.{self.config.get('figure_format', 'png')}"
            fig.savefig(file_path, dpi=self.config.get('dpi', 300),
                        bbox_inches='tight', facecolor='white')
            plt.close(fig)
        logger.info(f"Figures saved to: {output_path}")
