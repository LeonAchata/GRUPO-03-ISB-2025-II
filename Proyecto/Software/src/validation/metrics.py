"""
Metrics Module for EEG-BCI Project.

Implements evaluation metrics for classification performance.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    cohen_kappa_score
)
from typing import Dict, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns


class ClassificationMetrics:
    """
    Compute and store classification metrics.
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray,
                 y_proba: Optional[np.ndarray] = None,
                 class_names: Optional[list] = None):
        """
        Initialize metrics calculator.
        
        Parameters
        ----------
        y_true : ndarray
            True labels
        y_pred : ndarray
            Predicted labels
        y_proba : ndarray, optional
            Predicted probabilities (for AUC)
        class_names : list, optional
            Names of classes
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.class_names = class_names
        
    def compute_all(self) -> Dict[str, Any]:
        """
        Compute all metrics.
        
        Returns
        -------
        metrics : dict
            Dictionary of all metrics
        """
        metrics = {
            'accuracy': self.accuracy(),
            'precision': self.precision(),
            'recall': self.recall(),
            'f1_score': self.f1(),
            'confusion_matrix': self.conf_matrix(),
            'kappa': self.kappa()
        }
        
        # Add AUC if probabilities available
        if self.y_proba is not None:
            try:
                metrics['auc'] = self.auc()
            except:
                metrics['auc'] = None
        
        return metrics
    
    def accuracy(self) -> float:
        """Compute accuracy."""
        return accuracy_score(self.y_true, self.y_pred)
    
    def precision(self, average: str = 'macro') -> float:
        """
        Compute precision.
        
        Parameters
        ----------
        average : str
            Averaging method ('macro', 'micro', 'weighted')
        """
        return precision_score(self.y_true, self.y_pred, 
                             average=average, zero_division=0)
    
    def recall(self, average: str = 'macro') -> float:
        """
        Compute recall (sensitivity).
        
        Parameters
        ----------
        average : str
            Averaging method ('macro', 'micro', 'weighted')
        """
        return recall_score(self.y_true, self.y_pred, 
                          average=average, zero_division=0)
    
    def f1(self, average: str = 'macro') -> float:
        """
        Compute F1-score.
        
        Parameters
        ----------
        average : str
            Averaging method ('macro', 'micro', 'weighted')
        """
        return f1_score(self.y_true, self.y_pred, 
                       average=average, zero_division=0)
    
    def conf_matrix(self) -> np.ndarray:
        """Compute confusion matrix."""
        return confusion_matrix(self.y_true, self.y_pred)
    
    def kappa(self) -> float:
        """Compute Cohen's Kappa score."""
        return cohen_kappa_score(self.y_true, self.y_pred)
    
    def auc(self, average: str = 'macro') -> float:
        """
        Compute Area Under ROC Curve (for multi-class: one-vs-rest).
        
        Parameters
        ----------
        average : str
            Averaging method ('macro', 'micro', 'weighted')
        """
        if self.y_proba is None:
            raise ValueError("Probabilities not provided")
        
        return roc_auc_score(self.y_true, self.y_proba, 
                           average=average, multi_class='ovr')
    
    def classification_report_dict(self) -> Dict:
        """Get classification report as dictionary."""
        return classification_report(self.y_true, self.y_pred, 
                                    target_names=self.class_names,
                                    output_dict=True, zero_division=0)
    
    def print_report(self):
        """Print classification report."""
        print("\n" + "="*60)
        print("Classification Report")
        print("="*60)
        
        metrics = self.compute_all()
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  Kappa:     {metrics['kappa']:.4f}")
        
        if 'auc' in metrics and metrics['auc'] is not None:
            print(f"  AUC:       {metrics['auc']:.4f}")
        
        print("\nPer-Class Metrics:")
        print(classification_report(self.y_true, self.y_pred,
                                   target_names=self.class_names,
                                   zero_division=0))
        
        print("="*60)
    
    def plot_confusion_matrix(self, normalize: bool = False, 
                             figsize: tuple = (8, 6),
                             save_path: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Parameters
        ----------
        normalize : bool
            Whether to normalize confusion matrix
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        cm = self.conf_matrix()
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                  y_proba: Optional[np.ndarray] = None,
                  class_names: Optional[list] = None,
                  print_report: bool = True,
                  plot_cm: bool = False) -> Dict[str, Any]:
    """
    Convenience function to evaluate model performance.
    
    Parameters
    ----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    y_proba : ndarray, optional
        Predicted probabilities
    class_names : list, optional
        Class names
    print_report : bool
        Whether to print report
    plot_cm : bool
        Whether to plot confusion matrix
        
    Returns
    -------
    metrics : dict
        Dictionary of metrics
    """
    evaluator = ClassificationMetrics(y_true, y_pred, y_proba, class_names)
    
    metrics = evaluator.compute_all()
    
    if print_report:
        evaluator.print_report()
    
    if plot_cm:
        evaluator.plot_confusion_matrix()
    
    return metrics


def compute_per_class_accuracy(y_true: np.ndarray, 
                               y_pred: np.ndarray) -> Dict[int, float]:
    """
    Compute accuracy for each class separately.
    
    Parameters
    ----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
        
    Returns
    -------
    per_class_acc : dict
        Dictionary mapping class label to accuracy
    """
    per_class_acc = {}
    
    for class_label in np.unique(y_true):
        mask = y_true == class_label
        if np.sum(mask) > 0:
            acc = np.mean(y_pred[mask] == y_true[mask])
            per_class_acc[class_label] = acc
    
    return per_class_acc


if __name__ == "__main__":
    # Example usage
    from src.data.loader import load_eeg_data
    from src.data.parser import EventParser
    from src.preprocessing.filters import filter_eeg_data
    from src.preprocessing.segmentation import extract_epochs
    from src.features.csp import CSPExtractor
    from src.models.traditional import LDAClassifier
    from sklearn.model_selection import train_test_split
    
    print("Loading and preprocessing data...")
    raw = load_eeg_data("S001", "R04", data_dir="data/raw")
    raw_filtered = filter_eeg_data(raw)
    
    parser = EventParser("data/raw")
    events, event_id = parser.parse_events_from_raw(raw_filtered)
    _, class_labels = parser.map_events_to_classes(events, event_id, "R04")
    
    epochs = extract_epochs(raw_filtered, events, class_labels)
    X = epochs.get_data()
    y = epochs.events[:, 2]
    
    # Extract features
    csp = CSPExtractor(n_components=6)
    X_csp = csp.fit_transform(X, y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_csp, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    model = LDAClassifier()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Evaluate
    class_names = ['Left Hand', 'Right Hand']
    metrics = evaluate_model(y_test, y_pred, y_proba, 
                            class_names=class_names,
                            print_report=True,
                            plot_cm=True)
