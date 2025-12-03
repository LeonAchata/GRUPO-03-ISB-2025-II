"""
Artifact Removal Module for EEG-BCI Project.

Implements methods for detecting and removing artifacts from EEG signals,
including EOG (eye movements), EMG (muscle activity), and extreme values.
"""

import numpy as np
import mne
from typing import Optional, List, Tuple, Union
from mne.preprocessing import ICA


class ArtifactRemover:
    """Class for detecting and removing artifacts from EEG data."""
    
    def __init__(self, threshold: float = 100.0):
        """
        Initialize artifact remover.
        
        Parameters
        ----------
        threshold : float
            Amplitude threshold in microvolts for rejection
        """
        self.threshold = threshold
        
    def detect_bad_channels(self, raw: mne.io.Raw, 
                           method: str = 'correlation') -> List[str]:
        """
        Detect bad channels in EEG data.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data
        method : str
            Method for detection ('correlation' or 'variance')
            
        Returns
        -------
        bad_channels : list of str
            List of bad channel names
        """
        # Use MNE's automated bad channel detection
        raw_copy = raw.copy()
        
        # Detect flat channels
        raw_copy.info['bads'] = []
        
        # Simple variance-based detection
        data = raw_copy.get_data()
        channel_std = np.std(data, axis=1)
        
        # Channels with very low variance are bad
        threshold_low = np.percentile(channel_std, 5)
        threshold_high = np.percentile(channel_std, 95) * 3
        
        bad_channels = []
        for i, ch_name in enumerate(raw_copy.ch_names):
            if channel_std[i] < threshold_low or channel_std[i] > threshold_high:
                bad_channels.append(ch_name)
        
        return bad_channels
    
    def interpolate_bad_channels(self, raw: mne.io.Raw, 
                                bad_channels: Optional[List[str]] = None) -> mne.io.Raw:
        """
        Interpolate bad channels.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data
        bad_channels : list of str, optional
            List of bad channels (auto-detect if None)
            
        Returns
        -------
        raw_interp : mne.io.Raw
            Data with interpolated channels
        """
        raw_interp = raw.copy()
        
        if bad_channels is None:
            bad_channels = self.detect_bad_channels(raw_interp)
        
        if bad_channels:
            raw_interp.info['bads'] = bad_channels
            # Check if digitization info exists for interpolation
            if raw_interp.info.get('dig') is not None:
                raw_interp.interpolate_bads(reset_bads=True, verbose=False)
                print(f"Interpolated {len(bad_channels)} bad channels: {bad_channels}")
            else:
                # For EEG without digitization (e.g., PhysioNet), drop bad channels instead
                print(f"No digitization info found. Dropping {len(bad_channels)} bad channels: {bad_channels}")
                raw_interp.drop_channels(bad_channels)
        
        return raw_interp
    
    def apply_ica(self, raw: mne.io.Raw, n_components: int = 20,
                  random_state: int = 42) -> Tuple[mne.io.Raw, ICA]:
        """
        Apply Independent Component Analysis for artifact removal.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data (should be filtered first)
        n_components : int
            Number of ICA components
        random_state : int
            Random seed for reproducibility
            
        Returns
        -------
        raw_clean : mne.io.Raw
            Cleaned EEG data
        ica : ICA
            Fitted ICA object
        """
        # Create ICA object
        ica = ICA(n_components=n_components, random_state=random_state, 
                 max_iter='auto', verbose=False)
        
        # Fit ICA
        print("Fitting ICA...")
        ica.fit(raw, verbose=False)
        
        # Auto-detect EOG artifacts
        # Find EOG-related components
        eog_indices = []
        try:
            # Try to find EOG channels
            eog_indices, eog_scores = ica.find_bads_eog(raw, verbose=False)
            if eog_indices:
                print(f"Detected EOG components: {eog_indices}")
        except:
            print("No EOG channels found, skipping EOG detection")
        
        # Exclude detected components
        ica.exclude = eog_indices
        
        # Apply ICA to remove artifacts
        raw_clean = raw.copy()
        ica.apply(raw_clean, verbose=False)
        
        return raw_clean, ica
    
    def reject_by_amplitude(self, epochs: mne.Epochs, 
                           threshold: Optional[float] = None) -> mne.Epochs:
        """
        Reject epochs with amplitude exceeding threshold.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Epoched EEG data
        threshold : float, optional
            Rejection threshold in microvolts (uses default if None)
            
        Returns
        -------
        epochs_clean : mne.Epochs
            Epochs after rejection
        """
        threshold = threshold or self.threshold
        
        # Create rejection criteria (in volts for MNE)
        reject_criteria = dict(eeg=threshold * 1e-6)  # Convert μV to V
        
        # Apply rejection
        epochs_clean = epochs.copy()
        epochs_clean.drop_bad(reject=reject_criteria, verbose=False)
        
        n_rejected = len(epochs) - len(epochs_clean)
        print(f"Rejected {n_rejected}/{len(epochs)} epochs (threshold: {threshold} μV)")
        
        return epochs_clean
    
    def detect_muscle_artifacts(self, raw: mne.io.Raw, 
                               threshold: float = 3.0) -> np.ndarray:
        """
        Detect muscle artifacts (EMG) using high-frequency power.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data
        threshold : float
            Z-score threshold for detection
            
        Returns
        -------
        artifact_mask : ndarray
            Boolean mask indicating artifact presence
        """
        # Calculate power in high frequency band (40-100 Hz)
        raw_high = raw.copy().filter(l_freq=40, h_freq=100, verbose=False)
        data = raw_high.get_data()
        
        # Calculate envelope (RMS in sliding window)
        window_size = int(raw.info['sfreq'] * 0.5)  # 0.5 second windows
        envelope = np.array([
            np.sqrt(np.mean(data[:, i:i+window_size]**2, axis=1))
            for i in range(0, data.shape[1] - window_size, window_size)
        ]).T
        
        # Detect outliers
        mean_env = np.mean(envelope, axis=1, keepdims=True)
        std_env = np.std(envelope, axis=1, keepdims=True)
        z_scores = (envelope - mean_env) / (std_env + 1e-10)
        
        # Create mask
        artifact_mask = np.any(np.abs(z_scores) > threshold, axis=0)
        
        return artifact_mask
    
    def clean_data(self, raw: mne.io.Raw, apply_ica: bool = True,
                  interpolate_bads: bool = True) -> mne.io.Raw:
        """
        Apply complete artifact removal pipeline.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data (should be filtered)
        apply_ica : bool
            Whether to apply ICA
        interpolate_bads : bool
            Whether to interpolate bad channels
            
        Returns
        -------
        raw_clean : mne.io.Raw
            Cleaned EEG data
        """
        raw_clean = raw.copy()
        
        # Step 1: Interpolate bad channels
        if interpolate_bads:
            print("Step 1: Detecting and interpolating bad channels...")
            raw_clean = self.interpolate_bad_channels(raw_clean)
        
        # Step 2: Apply ICA
        if apply_ica:
            print("\nStep 2: Applying ICA for artifact removal...")
            raw_clean, _ = self.apply_ica(raw_clean)
        
        print("\nArtifact removal complete!")
        return raw_clean


def remove_artifacts(raw: mne.io.Raw, threshold: float = 100.0,
                    apply_ica: bool = True) -> mne.io.Raw:
    """
    Convenience function to remove artifacts.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    threshold : float
        Amplitude threshold for rejection (μV)
    apply_ica : bool
        Whether to apply ICA
        
    Returns
    -------
    raw_clean : mne.io.Raw
        Cleaned data
    """
    remover = ArtifactRemover(threshold=threshold)
    return remover.clean_data(raw, apply_ica=apply_ica)


if __name__ == "__main__":
    # Example usage
    from src.data.loader import load_eeg_data
    from src.preprocessing.filters import filter_eeg_data
    
    # Load and filter data first
    print("Loading sample EEG data...")
    raw = load_eeg_data("S001", "R04", data_dir="data/raw")
    
    print("\nApplying bandpass filter...")
    raw_filtered = filter_eeg_data(raw, l_freq=8.0, h_freq=30.0)
    
    # Remove artifacts
    print("\nRemoving artifacts...")
    remover = ArtifactRemover(threshold=100.0)
    raw_clean = remover.clean_data(raw_filtered, apply_ica=True)
    
    print(f"\nOriginal: {raw}")
    print(f"Cleaned: {raw_clean}")
