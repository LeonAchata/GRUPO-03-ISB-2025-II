"""
Common Spatial Patterns (CSP) Module for EEG-BCI Project.

CSP is a widely used algorithm for extracting spatial features from multi-channel
EEG data for motor imagery classification.
"""

import numpy as np
import mne
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Tuple
from mne.decoding import CSP as MNE_CSP


class CSPExtractor(BaseEstimator, TransformerMixin):
    """
    Common Spatial Patterns feature extractor for EEG signals.
    
    CSP finds spatial filters that maximize the variance of one class
    while minimizing the variance of another class.
    """
    
    def __init__(self, n_components: int = 6, reg: Optional[str] = None,
                 log: bool = True, cov_est: str = 'concat'):
        """
        Initialize CSP extractor.
        
        Parameters
        ----------
        n_components : int
            Number of CSP components to use (pairs from each end)
        reg : str or None
            Regularization method ('oas', 'ledoit_wolf', 'empirical', None)
        log : bool
            Whether to apply log transformation to features
        cov_est : str
            Covariance estimation method ('concat', 'epoch')
        """
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.cov_est = cov_est
        self.csp = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CSPExtractor':
        """
        Fit CSP spatial filters.
        
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            Training data
        y : ndarray, shape (n_epochs,)
            Class labels
            
        Returns
        -------
        self : CSPExtractor
            Fitted extractor
        """
        # Initialize MNE CSP
        self.csp = MNE_CSP(n_components=self.n_components,
                          reg=self.reg,
                          log=self.log,
                          cov_est=self.cov_est,
                          norm_trace=False)
        
        # Fit CSP
        self.csp.fit(X, y)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted CSP filters.
        
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            Data to transform
            
        Returns
        -------
        X_csp : ndarray, shape (n_epochs, n_components)
            CSP features
        """
        if self.csp is None:
            raise ValueError("CSP not fitted. Call fit() first.")
        
        return self.csp.transform(X)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit CSP and transform data.
        
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            Training data
        y : ndarray, shape (n_epochs,)
            Class labels
            
        Returns
        -------
        X_csp : ndarray, shape (n_epochs, n_components)
            CSP features
        """
        self.fit(X, y)
        return self.transform(X)
    
    def get_spatial_patterns(self) -> np.ndarray:
        """
        Get CSP spatial patterns (for visualization).
        
        Returns
        -------
        patterns : ndarray, shape (n_channels, n_components)
            Spatial patterns
        """
        if self.csp is None:
            raise ValueError("CSP not fitted. Call fit() first.")
        
        return self.csp.patterns_
    
    def get_spatial_filters(self) -> np.ndarray:
        """
        Get CSP spatial filters.
        
        Returns
        -------
        filters : ndarray, shape (n_channels, n_components)
            Spatial filters
        """
        if self.csp is None:
            raise ValueError("CSP not fitted. Call fit() first.")
        
        return self.csp.filters_


class MultiClassCSP:
    """
    Multi-class CSP using One-vs-Rest strategy.
    """
    
    def __init__(self, n_components: int = 6, **csp_kwargs):
        """
        Initialize multi-class CSP.
        
        Parameters
        ----------
        n_components : int
            Number of components per binary CSP
        **csp_kwargs : dict
            Additional arguments for CSPExtractor
        """
        self.n_components = n_components
        self.csp_kwargs = csp_kwargs
        self.csps = {}
        self.classes_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiClassCSP':
        """
        Fit one CSP per class (one-vs-rest).
        
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            Training data
        y : ndarray, shape (n_epochs,)
            Class labels
            
        Returns
        -------
        self : MultiClassCSP
            Fitted extractor
        """
        self.classes_ = np.unique(y)
        
        for class_label in self.classes_:
            # Create binary labels (one-vs-rest)
            y_binary = (y == class_label).astype(int)
            
            # Skip if only one class
            if len(np.unique(y_binary)) < 2:
                continue
            
            # Fit CSP for this class
            csp = CSPExtractor(n_components=self.n_components, **self.csp_kwargs)
            csp.fit(X, y_binary)
            self.csps[class_label] = csp
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using all fitted CSPs.
        
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            Data to transform
            
        Returns
        -------
        X_features : ndarray, shape (n_epochs, n_classes * n_components)
            Concatenated CSP features
        """
        features = []
        
        for class_label in sorted(self.csps.keys()):
            csp = self.csps[class_label]
            features.append(csp.transform(X))
        
        return np.hstack(features)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit and transform data.
        
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            Training data
        y : ndarray, shape (n_epochs,)
            Class labels
            
        Returns
        -------
        X_features : ndarray, shape (n_epochs, n_classes * n_components)
            CSP features
        """
        self.fit(X, y)
        return self.transform(X)


def extract_csp_features(X: np.ndarray, y: np.ndarray, 
                        n_components: int = 6) -> np.ndarray:
    """
    Convenience function to extract CSP features.
    
    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        EEG data
    y : ndarray, shape (n_epochs,)
        Class labels
    n_components : int
        Number of CSP components
        
    Returns
    -------
    X_csp : ndarray, shape (n_epochs, n_components)
        CSP features
    """
    csp = CSPExtractor(n_components=n_components)
    return csp.fit_transform(X, y)


if __name__ == "__main__":
    # Example usage
    from src.data.loader import load_eeg_data
    from src.data.parser import EventParser
    from src.preprocessing.filters import filter_eeg_data
    from src.preprocessing.segmentation import extract_epochs
    
    print("Loading and preprocessing data...")
    raw = load_eeg_data("S001", "R04", data_dir="data/raw")
    raw_filtered = filter_eeg_data(raw)
    
    parser = EventParser("data/raw")
    events, event_id = parser.parse_events_from_raw(raw_filtered)
    _, class_labels = parser.map_events_to_classes(events, event_id, "R04")
    
    epochs = extract_epochs(raw_filtered, events, class_labels)
    X = epochs.get_data()
    y = epochs.events[:, 2]
    
    print(f"\nOriginal data shape: {X.shape}")
    print(f"Labels: {np.unique(y)}")
    
    # Extract CSP features
    print("\nExtracting CSP features...")
    csp = CSPExtractor(n_components=6)
    X_csp = csp.fit_transform(X, y)
    
    print(f"CSP features shape: {X_csp.shape}")
    print(f"Feature statistics:")
    print(f"  Mean: {np.mean(X_csp, axis=0)}")
    print(f"  Std: {np.std(X_csp, axis=0)}")
