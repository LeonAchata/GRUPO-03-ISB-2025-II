"""
Spectral Features Module for EEG-BCI Project.

Implements Power Spectral Density (PSD) and other frequency-domain features.
"""

import numpy as np
import mne
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Tuple, List


class PSDExtractor(BaseEstimator, TransformerMixin):
    """
    Power Spectral Density (PSD) feature extractor.
    """
    
    def __init__(self, sfreq: float = 160.0, fmin: float = 8.0, 
                 fmax: float = 30.0, method: str = 'welch',
                 n_fft: int = 256, n_per_seg: Optional[int] = None):
        """
        Initialize PSD extractor.
        
        Parameters
        ----------
        sfreq : float
            Sampling frequency (Hz)
        fmin : float
            Minimum frequency (Hz)
        fmax : float
            Maximum frequency (Hz)
        method : str
            Method for PSD computation ('welch', 'multitaper', 'fft')
        n_fft : int
            FFT length
        n_per_seg : int, optional
            Length of each segment for Welch method
        """
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax
        self.method = method
        self.n_fft = n_fft
        self.n_per_seg = n_per_seg or n_fft
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'PSDExtractor':
        """
        Fit (dummy method for sklearn compatibility).
        
        Parameters
        ----------
        X : ndarray
            Training data (not used)
        y : ndarray, optional
            Labels (not used)
            
        Returns
        -------
        self : PSDExtractor
        """
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extract PSD features from EEG data.
        
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            EEG data
            
        Returns
        -------
        X_psd : ndarray, shape (n_epochs, n_channels * n_freqs)
            PSD features (flattened)
        """
        n_epochs, n_channels, n_times = X.shape
        
        if self.method == 'welch':
            psd_list = []
            for epoch in X:
                psd_epoch = []
                for ch_data in epoch:
                    freqs, psd = signal.welch(ch_data, 
                                             fs=self.sfreq,
                                             nperseg=self.n_per_seg,
                                             nfft=self.n_fft)
                    # Select frequency range
                    freq_mask = (freqs >= self.fmin) & (freqs <= self.fmax)
                    psd_epoch.append(psd[freq_mask])
                psd_list.append(np.hstack(psd_epoch))
            
            X_psd = np.array(psd_list)
            
        elif self.method == 'fft':
            # Simple FFT-based PSD
            X_psd = self._compute_fft_psd(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return X_psd
    
    def _compute_fft_psd(self, X: np.ndarray) -> np.ndarray:
        """
        Compute PSD using FFT.
        
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            EEG data
            
        Returns
        -------
        psd : ndarray
            PSD features
        """
        n_epochs, n_channels, n_times = X.shape
        
        # Compute FFT
        fft_data = np.fft.rfft(X, n=self.n_fft, axis=2)
        psd = np.abs(fft_data) ** 2 / n_times
        
        # Frequency bins
        freqs = np.fft.rfftfreq(self.n_fft, 1/self.sfreq)
        freq_mask = (freqs >= self.fmin) & (freqs <= self.fmax)
        
        # Select frequency range and flatten
        psd_selected = psd[:, :, freq_mask]
        psd_flat = psd_selected.reshape(n_epochs, -1)
        
        return psd_flat
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit and transform data.
        
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            EEG data
        y : ndarray, optional
            Labels (not used)
            
        Returns
        -------
        X_psd : ndarray
            PSD features
        """
        self.fit(X, y)
        return self.transform(X)


class BandPowerExtractor(BaseEstimator, TransformerMixin):
    """
    Extract band power features from specific frequency bands.
    """
    
    # Standard EEG frequency bands
    BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'mu': (8, 13),       # Same as alpha, over motor cortex
        'beta': (13, 30),
        'gamma': (30, 100)
    }
    
    def __init__(self, sfreq: float = 160.0, 
                 bands: Optional[dict] = None,
                 relative: bool = True):
        """
        Initialize band power extractor.
        
        Parameters
        ----------
        sfreq : float
            Sampling frequency (Hz)
        bands : dict, optional
            Custom frequency bands {name: (fmin, fmax)}
            If None, uses mu and beta bands
        relative : bool
            Whether to compute relative power (normalized by total power)
        """
        self.sfreq = sfreq
        self.bands = bands or {'mu': (8, 13), 'beta': (13, 30)}
        self.relative = relative
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BandPowerExtractor':
        """Fit (dummy method)."""
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extract band power features.
        
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            EEG data
            
        Returns
        -------
        X_bp : ndarray, shape (n_epochs, n_channels * n_bands)
            Band power features
        """
        n_epochs, n_channels, n_times = X.shape
        n_bands = len(self.bands)
        
        # Initialize output
        band_powers = np.zeros((n_epochs, n_channels, n_bands))
        
        for epoch_idx in range(n_epochs):
            for ch_idx in range(n_channels):
                # Compute PSD using Welch
                freqs, psd = signal.welch(X[epoch_idx, ch_idx, :],
                                         fs=self.sfreq,
                                         nperseg=min(256, n_times))
                
                # Compute total power (for relative power)
                if self.relative:
                    total_power = np.sum(psd)
                else:
                    total_power = 1.0
                
                # Extract power in each band
                for band_idx, (band_name, (fmin, fmax)) in enumerate(self.bands.items()):
                    freq_mask = (freqs >= fmin) & (freqs <= fmax)
                    band_power = np.sum(psd[freq_mask])
                    band_powers[epoch_idx, ch_idx, band_idx] = band_power / total_power
        
        # Flatten: (n_epochs, n_channels * n_bands)
        X_bp = band_powers.reshape(n_epochs, -1)
        
        return X_bp
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)


def extract_psd_features(X: np.ndarray, sfreq: float = 160.0,
                        fmin: float = 8.0, fmax: float = 30.0) -> np.ndarray:
    """
    Convenience function to extract PSD features.
    
    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        EEG data
    sfreq : float
        Sampling frequency
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
        
    Returns
    -------
    X_psd : ndarray
        PSD features
    """
    extractor = PSDExtractor(sfreq=sfreq, fmin=fmin, fmax=fmax)
    return extractor.fit_transform(X)


def extract_band_power(X: np.ndarray, sfreq: float = 160.0) -> np.ndarray:
    """
    Convenience function to extract band power features.
    
    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        EEG data
    sfreq : float
        Sampling frequency
        
    Returns
    -------
    X_bp : ndarray
        Band power features
    """
    extractor = BandPowerExtractor(sfreq=sfreq)
    return extractor.fit_transform(X)


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
    
    # Extract PSD features
    print("\nExtracting PSD features...")
    psd_extractor = PSDExtractor(sfreq=160.0, fmin=8.0, fmax=30.0)
    X_psd = psd_extractor.fit_transform(X)
    print(f"PSD features shape: {X_psd.shape}")
    
    # Extract band power features
    print("\nExtracting band power features...")
    bp_extractor = BandPowerExtractor(sfreq=160.0)
    X_bp = bp_extractor.fit_transform(X)
    print(f"Band power features shape: {X_bp.shape}")
