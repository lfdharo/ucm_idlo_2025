"""
Visualization Module for Speaker Identification Results

This module provides easy-to-use visualization functions for displaying
speaker identification results, confusion matrices, ROC curves, and more.
Designed for linguistic students with minimal matplotlib experience.

Author: Luis F. D'Haro
Date: Apr 7, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
import logging

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class VisualizationTools:
    """Simple visualization tools for speaker identification results."""
    
    def __init__(self, output_dir: str = './results/'):
        """Initialize visualization tools.
        
        Args:
            output_dir (str): Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    # ======================== METRICS VISUALIZATION ========================
    
    def plot_performance_metrics(self, 
                                  metrics: Dict[str, float],
                                  title: str = "Speaker Identification Performance",
                                  save_to: Optional[str] = None) -> None:
        """Plot overall performance metrics as a bar chart.
        
        Args:
            metrics (dict): Dictionary with 'accuracy', 'precision', 'recall', 'f1_score'
            title (str): Plot title
            save_to (str, optional): Optional path to save the figure
            
        Example:
            >>> metrics = {'accuracy': 0.92, 'precision': 0.89, 'recall': 0.95, 'f1_score': 0.92}
            >>> plot_performance_metrics(metrics)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metric_values = [
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1_score', 0)
        ]
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2%}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_to:
            plt.savefig(save_to, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_to}")
        
        plt.show()
    
    def plot_speaker_performance(self,
                                 speaker_metrics: Dict[str, Dict],
                                 title: str = "Performance by Speaker",
                                 save_to: Optional[str] = None) -> None:
        """Plot performance metrics for each speaker.
        
        Args:
            speaker_metrics (dict): Dict with speaker IDs as keys and metric dicts as values
            title (str): Plot title
            save_to (str, optional): Path to save figure
            
        Example:
            >>> speaker_metrics = {
            ...     'SPK1': {'accuracy': 0.95, 'precision': 0.92, 'recall': 0.98},
            ...     'SPK2': {'accuracy': 0.87, 'precision': 0.85, 'recall': 0.90}
            ... }
            >>> plot_speaker_performance(speaker_metrics)
        """
        speakers = list(speaker_metrics.keys())
        accuracies = [speaker_metrics[s].get('accuracy', 0) for s in speakers]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(speakers)))
        bars = ax.bar(speakers, accuracies, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.1%}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Speaker ID', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_to:
            plt.savefig(save_to, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_to}")
        
        plt.show()
    
    # ======================== CONFUSION MATRIX ========================
    
    def plot_confusion_matrix(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              labels: Optional[List[str]] = None,
                              title: str = "Confusion Matrix",
                              save_to: Optional[str] = None) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            labels (list, optional): Label names
            title (str): Plot title
            save_to (str, optional): Path to save figure
            
        Example:
            >>> plot_confusion_matrix(true_labels, predicted_labels, 
            ...                       labels=['SPK1', 'SPK2', 'SPK3'])
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'}, ax=ax)
        
        ax.set_ylabel('True Speaker', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Speaker', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_to:
            plt.savefig(save_to, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix saved to {save_to}")
        
        plt.show()
    
    # ======================== SCORE DISTRIBUTIONS ========================
    
    def plot_score_distribution(self,
                               genuine_scores: List[float],
                               impostor_scores: List[float],
                               title: str = "Score Distribution",
                               threshold: Optional[float] = None,
                               save_to: Optional[str] = None) -> None:
        """Plot distribution of genuine and impostor scores.
        
        Args:
            genuine_scores (list): Scores for genuine speaker pairs
            impostor_scores (list): Scores for impostor pairs
            title (str): Plot title
            threshold (float, optional): Decision threshold to mark
            save_to (str, optional): Path to save figure
            
        Example:
            >>> plot_score_distribution(genuine_scores, impostor_scores, threshold=0.5)
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(genuine_scores, bins=30, alpha=0.7, label='Genuine Pairs', color='#2ecc71', edgecolor='black')
        ax.hist(impostor_scores, bins=30, alpha=0.7, label='Impostor Pairs', color='#e74c3c', edgecolor='black')
        
        if threshold is not None:
            ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
        
        ax.set_xlabel('Similarity Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_to:
            plt.savefig(save_to, dpi=300, bbox_inches='tight')
            self.logger.info(f"Distribution plot saved to {save_to}")
        
        plt.show()
    
    # ======================== COMPARISON PLOTS ========================
    
    def compare_models(self,
                       model_results: Dict[str, Dict],
                       metric: str = 'accuracy',
                       title: str = "Model Comparison",
                       save_to: Optional[str] = None) -> None:
        """Compare performance across different models.
        
        Args:
            model_results (dict): Dict with model names as keys and metric dicts as values
            metric (str): Metric to compare ('accuracy', 'precision', 'recall', 'f1_score')
            title (str): Plot title
            save_to (str, optional): Path to save figure
            
        Example:
            >>> results = {
            ...     'wavLM': {'accuracy': 0.92, 'precision': 0.89},
            ...     'SpeechBrain': {'accuracy': 0.88, 'precision': 0.86}
            ... }
            >>> compare_models(results, metric='accuracy')
        """
        models = list(model_results.keys())
        values = [model_results[m].get(metric, 0) for m in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2%}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(metric.capitalize(), fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_to:
            plt.savefig(save_to, dpi=300, bbox_inches='tight')
            self.logger.info(f"Comparison plot saved to {save_to}")
        
        plt.show()
    
    # ======================== UTILITY FUNCTIONS ========================
    
    def create_summary_report(self,
                             metrics: Dict[str, float],
                             speaker_metrics: Dict[str, Dict],
                             output_file: str = 'report.txt') -> None:
        """Create a text summary report.
        
        Args:
            metrics (dict): Overall performance metrics
            speaker_metrics (dict): Per-speaker metrics
            output_file (str): Output text file path
        """
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SPEAKER IDENTIFICATION - EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("OVERALL PERFORMANCE:\n")
            f.write("-"*70 + "\n")
            f.write(f"Accuracy:  {metrics.get('accuracy', 0):.2%}\n")
            f.write(f"Precision: {metrics.get('precision', 0):.2%}\n")
            f.write(f"Recall:    {metrics.get('recall', 0):.2%}\n")
            f.write(f"F1 Score:  {metrics.get('f1_score', 0):.2%}\n\n")
            
            if speaker_metrics:
                f.write("PER-SPEAKER PERFORMANCE:\n")
                f.write("-"*70 + "\n")
                for speaker, metrics_dict in speaker_metrics.items():
                    acc = metrics_dict.get('accuracy', 0)
                    f.write(f"{speaker:10s}: {acc:.2%}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        self.logger.info(f"Report saved to {output_file}")


# ======================== CONVENIENCE FUNCTIONS ========================

def plot_metrics(metrics: Dict[str, float], **kwargs) -> None:
    """Convenience function to quickly plot metrics.
    
    Example:
        >>> plot_metrics({'accuracy': 0.92, 'precision': 0.89, 'recall': 0.95, 'f1_score': 0.92})
    """
    viz = VisualizationTools()
    viz.plot_performance_metrics(metrics, **kwargs)


def plot_by_speaker(speaker_metrics: Dict[str, Dict], **kwargs) -> None:
    """Convenience function to quickly plot per-speaker performance.
    
    Example:
        >>> plot_by_speaker({'SPK1': {'accuracy': 0.95}, 'SPK2': {'accuracy': 0.87}})
    """
    viz = VisualizationTools()
    viz.plot_speaker_performance(speaker_metrics, **kwargs)


def compare_model_performance(model_results: Dict[str, Dict], **kwargs) -> None:
    """Convenience function to compare multiple models.
    
    Example:
        >>> results = {'wavLM': {'accuracy': 0.92}, 'SpeechBrain': {'accuracy': 0.88}}
        >>> compare_model_performance(results)
    """
    viz = VisualizationTools()
    viz.compare_models(model_results, **kwargs)
