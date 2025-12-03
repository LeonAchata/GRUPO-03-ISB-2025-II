"""
Signal Filtering Module for EEG-BCI Project.

Implements bandpass filtering (mu and beta bands) and notch filtering
for powerline noise removal.
"""

import numpy as np
import mne
from typing import Optional, Tuple, Union


class SignalFilter:
    """Class for filtering EEG signals."""
    
    def __init__(self, l_freq: float = 8.0, h_freq: float = 30.0, 
                 notch_freq: Optional[float] = 60.0):
        """
        Initialize signal filter.
        
        Parameters
        ----------
        l_freq : float
            Low cutoff frequency for bandpass filter (Hz)
        h_freq : float
            High cutoff frequency for bandpass filter (Hz)
        notch_freq : float or None
            Notch filter frequency for powerline noise (Hz)
            Use 50.0 for Europe, 60.0 for Americas, None to skip
        """
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        
    def apply_bandpass(self, raw: mne.io.Raw, l_freq: Optional[float] = None,
                       h_freq: Optional[float] = None, 
                       copy: bool = True) -> mne.io.Raw:
        """
        Apply bandpass filter to EEG data.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data
        l_freq : float, optional
            Low cutoff frequency (uses default if None)
        h_freq : float, optional
            High cutoff frequency (uses default if None)
        copy : bool
            Whether to copy data before filtering
            
        Returns
        -------
        raw_filtered : mne.io.Raw
            Filtered EEG data
        """
        l_freq = l_freq or self.l_freq
        h_freq = h_freq or self.h_freq
        
        if copy:
            raw_filtered = raw.copy()
        else:
            raw_filtered = raw
            
        # Apply bandpass filter (FIR filter)
        raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, 
                           fir_design='firwin', verbose=False)
        
        return raw_filtered
    
    def apply_notch(self, raw: mne.io.Raw, notch_freq: Optional[float] = None,
                    copy: bool = True) -> mne.io.Raw:
        """
        Apply notch filter to remove powerline noise.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data
        notch_freq : float, optional
            Notch frequency (uses default if None)
        copy : bool
            Whether to copy data before filtering
            
        Returns
        -------
        raw_filtered : mne.io.Raw
            Filtered EEG data
        """
        notch_freq = notch_freq or self.notch_freq
        
        if notch_freq is None:
            return raw
        
        if copy:
            raw_filtered = raw.copy()
        else:
            raw_filtered = raw
            
        # Apply notch filter
        raw_filtered.notch_filter(freqs=notch_freq, verbose=False)
        
        return raw_filtered
    
    def filter_data(self, raw: mne.io.Raw, apply_notch: bool = True,
                   copy: bool = True) -> mne.io.Raw:
        """
        Apply complete filtering pipeline (bandpass + notch).
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data
        apply_notch : bool
            Whether to apply notch filter
        copy : bool
            Whether to copy data before filtering
            
        Returns
        -------
        raw_filtered : mne.io.Raw
            Filtered EEG data
        """
        if copy:
            raw_filtered = raw.copy()
        else:
            raw_filtered = raw
        
        # Apply notch filter first (if requested)
        if apply_notch and self.notch_freq is not None:
            raw_filtered = self.apply_notch(raw_filtered, copy=False)
        
        # Apply bandpass filter
        raw_filtered = self.apply_bandpass(raw_filtered, copy=False)
        
        return raw_filtered
    
    def get_filter_response(self, sfreq: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get frequency response of the bandpass filter.
        
        Parameters
        ----------
        sfreq : float
            Sampling frequency (Hz)
            
        Returns
        -------
        freqs : ndarray
            Frequency values
        response : ndarray
            Magnitude response
        """
        from scipy import signal
        
        # Design FIR filter
        nyq = sfreq / 2
        numtaps = min(int(sfreq * 0.5), 1001)  # Filter length
        taps = signal.firwin(numtaps, [self.l_freq, self.h_freq], 
                           pass_zero=False, fs=sfreq)
        
        # Get frequency response
        freqs, response = signal.freqz(taps, worN=8000, fs=sfreq)
        
        return freqs, np.abs(response)


def apply_bandpass_filter(raw: mne.io.Raw, l_freq: float = 8.0, 
                          h_freq: float = 30.0) -> mne.io.Raw:
    """
    Convenience function to apply bandpass filter.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    l_freq : float
        Low cutoff frequency (Hz)
    h_freq : float
        High cutoff frequency (Hz)
        
    Returns
    -------
    raw_filtered : mne.io.Raw
        Filtered data
    """
    signal_filter = SignalFilter(l_freq=l_freq, h_freq=h_freq)
    return signal_filter.apply_bandpass(raw)


def apply_notch_filter(raw: mne.io.Raw, notch_freq: float = 60.0) -> mne.io.Raw:
    """
    Convenience function to apply notch filter.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    notch_freq : float
        Notch frequency (Hz)
        
    Returns
    -------
    raw_filtered : mne.io.Raw
        Filtered data
    """
    signal_filter = SignalFilter(notch_freq=notch_freq)
    return signal_filter.apply_notch(raw)


def filter_eeg_data(raw: mne.io.Raw, l_freq: float = 8.0, h_freq: float = 30.0,
                   notch_freq: Optional[float] = 60.0) -> mne.io.Raw:
    """
    Apply complete filtering pipeline.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    l_freq : float
        Low cutoff frequency (Hz)
    h_freq : float
        High cutoff frequency (Hz)
    notch_freq : float or None
        Notch frequency (Hz), None to skip
        
    Returns
    -------
    raw_filtered : mne.io.Raw
        Filtered data
    """
    signal_filter = SignalFilter(l_freq=l_freq, h_freq=h_freq, 
                                 notch_freq=notch_freq)
    return signal_filter.filter_data(raw)


if __name__ == "__main__":
    # Example usage
    from src.data.loader import load_eeg_data
    
    # Load sample data
    print("Loading sample EEG data...")
    raw = load_eeg_data("S001", "R04", data_dir="data/raw")
    print(f"Original data: {raw}")
    
    # Create filter
    signal_filter = SignalFilter(l_freq=8.0, h_freq=30.0, notch_freq=60.0)
    
    # Apply filtering
    print("\nApplying filters...")
    raw_filtered = signal_filter.filter_data(raw)
    print(f"Filtered data: {raw_filtered}")
    
    # Compare PSDs
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    raw.plot_psd(fmin=1, fmax=50, ax=axes[0], show=False, average=True)
    axes[0].set_title('Original Signal PSD')
    
    raw_filtered.plot_psd(fmin=1, fmax=50, ax=axes[1], show=False, average=True)
    axes[1].set_title('Filtered Signal PSD (8-30 Hz)')
    
    plt.tight_layout()
    plt.savefig('reports/figures/filtering_comparison.png', dpi=150)
    print("\nPSD comparison saved to reports/figures/filtering_comparison.png")
