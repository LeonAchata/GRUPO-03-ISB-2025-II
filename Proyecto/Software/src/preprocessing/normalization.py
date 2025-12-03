"""
Normalization Module for EEG-BCI Project.

Implements various normalization techniques for EEG data.
"""

import numpy as np
import mne
from typing import Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler


class DataNormalizer:
    """Class for normalizing EEG data."""
    
    def __init__(self, method: str = 'zscore'):
        """
        Initialize data normalizer.
        
        Parameters
        ----------
        method : str
            Normalization method: 'zscore', 'robust', 'minmax', 'baseline'
        """
        self.method = method
        self.scaler = None
        
    def normalize_epochs(self, epochs: mne.Epochs, 
                        method: Optional[str] = None) -> mne.Epochs:
        """
        Normalize epoched data.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Epoched EEG data
        method : str, optional
            Normalization method (uses default if None)
            
        Returns
        -------
        epochs_norm : mne.Epochs
            Normalized epochs
        """
        method = method or self.method
        epochs_norm = epochs.copy()
        
        if method == 'zscore':
            epochs_norm = self._zscore_normalize(epochs_norm)
        elif method == 'robust':
            epochs_norm = self._robust_normalize(epochs_norm)
        elif method == 'minmax':
            epochs_norm = self._minmax_normalize(epochs_norm)
        elif method == 'baseline':
            # Baseline already applied during epoching, just ensure it's there
            if epochs_norm.baseline is None:
                print("Warning: No baseline set. Applying baseline correction.")
                epochs_norm.apply_baseline(baseline=(-0.5, 0))
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return epochs_norm
    
    def _zscore_normalize(self, epochs: mne.Epochs) -> mne.Epochs:
        """
        Apply z-score normalization (standardization).
        
        Normalize each channel to have mean=0 and std=1.
        """
        data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
        
        # Normalize per channel across all epochs and time points
        mean = np.mean(data, axis=(0, 2), keepdims=True)
        std = np.std(data, axis=(0, 2), keepdims=True)
        
        data_norm = (data - mean) / (std + 1e-10)
        
        # Update epochs data
        epochs._data = data_norm
        
        print(f"Applied z-score normalization: mean={np.mean(data_norm):.3f}, "
              f"std={np.std(data_norm):.3f}")
        
        return epochs
    
    def _robust_normalize(self, epochs: mne.Epochs) -> mne.Epochs:
        """
        Apply robust normalization using median and IQR.
        
        More robust to outliers than z-score.
        """
        data = epochs.get_data()
        
        # Use median and IQR instead of mean and std
        median = np.median(data, axis=(0, 2), keepdims=True)
        q75 = np.percentile(data, 75, axis=(0, 2), keepdims=True)
        q25 = np.percentile(data, 25, axis=(0, 2), keepdims=True)
        iqr = q75 - q25
        
        data_norm = (data - median) / (iqr + 1e-10)
        
        epochs._data = data_norm
        
        print(f"Applied robust normalization: median={np.median(data_norm):.3f}")
        
        return epochs
    
    def _minmax_normalize(self, epochs: mne.Epochs) -> mne.Epochs:
        """
        Apply min-max normalization to [0, 1] range.
        """
        data = epochs.get_data()
        
        # Normalize per channel
        data_min = np.min(data, axis=(0, 2), keepdims=True)
        data_max = np.max(data, axis=(0, 2), keepdims=True)
        
        data_norm = (data - data_min) / (data_max - data_min + 1e-10)
        
        epochs._data = data_norm
        
        print(f"Applied min-max normalization: min={np.min(data_norm):.3f}, "
              f"max={np.max(data_norm):.3f}")
        
        return epochs
    
    def normalize_array(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                       fit: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Normalize numpy array data.
        
        Parameters
        ----------
        X : ndarray
            Data array to normalize
        y : ndarray, optional
            Labels (returned unchanged if provided)
        fit : bool
            Whether to fit the scaler (True for training, False for test)
            
        Returns
        -------
        X_norm : ndarray
            Normalized data
        y : ndarray (if provided)
            Original labels
        """
        original_shape = X.shape
        
        # Reshape to 2D for sklearn scalers
        if X.ndim == 3:
            # (n_epochs, n_channels, n_times) -> (n_epochs, n_channels * n_times)
            X_2d = X.reshape(X.shape[0], -1)
        else:
            X_2d = X
        
        if self.method in ['zscore', 'standard']:
            if fit or self.scaler is None:
                self.scaler = StandardScaler()
                X_norm = self.scaler.fit_transform(X_2d)
            else:
                X_norm = self.scaler.transform(X_2d)
                
        elif self.method == 'robust':
            if fit or self.scaler is None:
                self.scaler = RobustScaler()
                X_norm = self.scaler.fit_transform(X_2d)
            else:
                X_norm = self.scaler.transform(X_2d)
        else:
            # Simple per-sample normalization
            mean = np.mean(X_2d, axis=1, keepdims=True)
            std = np.std(X_2d, axis=1, keepdims=True)
            X_norm = (X_2d - mean) / (std + 1e-10)
        
        # Reshape back to original
        if X.ndim == 3:
            X_norm = X_norm.reshape(original_shape)
        
        if y is not None:
            return X_norm, y
        return X_norm
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit normalizer and transform data.
        
        Parameters
        ----------
        X : ndarray
            Training data
            
        Returns
        -------
        X_norm : ndarray
            Normalized data
        """
        return self.normalize_array(X, fit=True)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted normalizer.
        
        Parameters
        ----------
        X : ndarray
            Data to transform
            
        Returns
        -------
        X_norm : ndarray
            Normalized data
        """
        return self.normalize_array(X, fit=False)


def normalize_epochs(epochs: mne.Epochs, method: str = 'zscore') -> mne.Epochs:
    """
    Convenience function to normalize epochs.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data
    method : str
        Normalization method
        
    Returns
    -------
    epochs_norm : mne.Epochs
        Normalized epochs
    """
    normalizer = DataNormalizer(method=method)
    return normalizer.normalize_epochs(epochs)


def normalize_data(X: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Convenience function to normalize numpy array.
    
    Parameters
    ----------
    X : ndarray
        Data to normalize
    method : str
        Normalization method
        
    Returns
    -------
    X_norm : ndarray
        Normalized data
    """
    normalizer = DataNormalizer(method=method)
    return normalizer.normalize_array(X)


if __name__ == "__main__":
    # Example usage
    from src.data.loader import load_eeg_data
    from src.data.parser import EventParser
    from src.preprocessing.filters import filter_eeg_data
    from src.preprocessing.segmentation import extract_epochs
    
    # Load and preprocess data
    print("Loading and preprocessing EEG data...")
    raw = load_eeg_data("S001", "R04", data_dir="data/raw")
    raw_filtered = filter_eeg_data(raw)
    
    # Parse events
    parser = EventParser("data/raw")
    events, event_id = parser.parse_events_from_raw(raw_filtered)
    _, class_labels = parser.map_events_to_classes(events, event_id, "R04")
    
    # Extract epochs
    epochs = extract_epochs(raw_filtered, events, class_labels)
    
    # Normalize
    print("\nNormalizing epochs...")
    normalizer = DataNormalizer(method='zscore')
    epochs_norm = normalizer.normalize_epochs(epochs)
    
    # Get data
    X_norm, y = epochs_norm.get_data(), epochs_norm.events[:, 2]
    
    print(f"\nNormalized data shape: {X_norm.shape}")
    print(f"Mean: {np.mean(X_norm):.6f}")
    print(f"Std: {np.std(X_norm):.6f}")
    print(f"Min: {np.min(X_norm):.6f}")
    print(f"Max: {np.max(X_norm):.6f}")
