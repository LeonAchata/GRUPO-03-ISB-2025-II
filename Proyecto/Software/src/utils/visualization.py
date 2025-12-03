"""
Main utility module for visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
import mne


def plot_spatial_patterns(patterns: np.ndarray, info: mne.Info,
                          components: Optional[List[int]] = None,
                          figsize: tuple = (12, 8),
                          save_path: Optional[str] = None):
    """
    Plot CSP spatial patterns as topographic maps.
    
    Parameters
    ----------
    patterns : ndarray, shape (n_channels, n_components)
        Spatial patterns
    info : mne.Info
        MNE info object with channel locations
    components : list of int, optional
        Which components to plot (default: first 6)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    if components is None:
        components = list(range(min(6, patterns.shape[1])))
    
    n_components = len(components)
    n_cols = 3
    n_rows = int(np.ceil(n_components / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_components > 1 else [axes]
    
    for idx, comp_idx in enumerate(components):
        mne.viz.plot_topomap(patterns[:, comp_idx], info, 
                            axes=axes[idx], show=False)
        axes[idx].set_title(f'Component {comp_idx + 1}')
    
    # Hide unused subplots
    for idx in range(n_components, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('CSP Spatial Patterns')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_feature_distributions(X: np.ndarray, y: np.ndarray,
                               feature_names: Optional[List[str]] = None,
                               n_features: int = 6,
                               figsize: tuple = (14, 8),
                               save_path: Optional[str] = None):
    """
    Plot feature distributions for each class.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Features
    y : ndarray
        Labels
    feature_names : list of str, optional
        Feature names
    n_features : int
        Number of features to plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    n_features = min(n_features, X.shape[1])
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    unique_classes = np.unique(y)
    
    for idx in range(n_features):
        for class_label in unique_classes:
            mask = y == class_label
            axes[idx].hist(X[mask, idx], alpha=0.6, bins=20, 
                          label=f'Class {class_label}')
        
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(feature_names[idx])
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Feature Distributions by Class')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    print("Visualization utilities loaded.")
