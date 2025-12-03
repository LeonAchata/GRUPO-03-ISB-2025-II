"""
EDF File Loader for EEG-BCI Motor Imagery Project.

This module provides functions to load and read EDF files containing EEG data
from the PhysioNet Motor Movement/Imagery Dataset.
"""

import os
import numpy as np
import mne
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import warnings


class EEGLoader:
    """Class for loading EEG data from EDF files."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the EEG loader.
        
        Parameters
        ----------
        data_dir : str
            Path to the directory containing subject folders
        """
        self.data_dir = Path(data_dir)
        
    def load_raw(self, subject: str, run: str, preload: bool = True) -> mne.io.Raw:
        """
        Load raw EEG data from an EDF file.
        
        Parameters
        ----------
        subject : str
            Subject ID (e.g., 'S001')
        run : str
            Run ID (e.g., 'R01' or 'R04')
        preload : bool
            Whether to preload data into memory
            
        Returns
        -------
        raw : mne.io.Raw
            Raw EEG data object
        """
        # Construct file path
        filename = f"{subject}{run}.edf"
        filepath = self.data_dir / subject / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"EDF file not found: {filepath}")
        
        # Load the EDF file
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            raw = mne.io.read_raw_edf(str(filepath), preload=preload, verbose=False)
        
        return raw
    
    def load_multiple_runs(self, subject: str, runs: List[str]) -> List[mne.io.Raw]:
        """
        Load multiple runs for a subject.
        
        Parameters
        ----------
        subject : str
            Subject ID (e.g., 'S001')
        runs : list of str
            List of run IDs (e.g., ['R04', 'R08', 'R12'])
            
        Returns
        -------
        raw_list : list of mne.io.Raw
            List of raw EEG data objects
        """
        raw_list = []
        for run in runs:
            try:
                raw = self.load_raw(subject, run)
                raw_list.append(raw)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
        
        return raw_list
    
    def get_info(self, subject: str, run: str) -> Dict:
        """
        Get information about an EEG recording.
        
        Parameters
        ----------
        subject : str
            Subject ID
        run : str
            Run ID
            
        Returns
        -------
        info_dict : dict
            Dictionary containing recording information
        """
        raw = self.load_raw(subject, run, preload=False)
        
        info_dict = {
            'subject': subject,
            'run': run,
            'n_channels': len(raw.ch_names),
            'channel_names': raw.ch_names,
            'sampling_rate': raw.info['sfreq'],
            'duration': raw.times[-1],
            'n_samples': len(raw.times)
        }
        
        return info_dict
    
    def list_available_subjects(self) -> List[str]:
        """
        List all available subjects in the data directory.
        
        Returns
        -------
        subjects : list of str
            List of subject IDs
        """
        subjects = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and item.name.startswith('S'):
                subjects.append(item.name)
        return sorted(subjects)
    
    def list_available_runs(self, subject: str) -> List[str]:
        """
        List all available runs for a subject.
        
        Parameters
        ----------
        subject : str
            Subject ID
            
        Returns
        -------
        runs : list of str
            List of run IDs
        """
        subject_dir = self.data_dir / subject
        if not subject_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
        
        runs = []
        for file in subject_dir.glob("*.edf"):
            # Extract run ID from filename (e.g., S001R04.edf -> R04)
            run_id = file.stem.replace(subject, '')
            if run_id and not file.name.endswith('.event'):
                runs.append(run_id)
        
        return sorted(runs)


def load_eeg_data(subject: str, run: str, data_dir: str = "data") -> mne.io.Raw:
    """
    Convenience function to load EEG data.
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'S001')
    run : str
        Run ID (e.g., 'R04')
    data_dir : str
        Path to data directory
        
    Returns
    -------
    raw : mne.io.Raw
        Raw EEG data
    """
    loader = EEGLoader(data_dir)
    return loader.load_raw(subject, run)


if __name__ == "__main__":
    # Example usage
    loader = EEGLoader("data")
    
    # List available subjects
    print("Available subjects:")
    subjects = loader.list_available_subjects()
    print(subjects[:5])  # Show first 5
    
    # List runs for first subject
    if subjects:
        subject = subjects[0]
        print(f"\nAvailable runs for {subject}:")
        runs = loader.list_available_runs(subject)
        print(runs)
        
        # Load a sample run
        if runs:
            print(f"\nLoading {subject}{runs[0]}...")
            info = loader.get_info(subject, runs[0])
            print(f"Channels: {info['n_channels']}")
            print(f"Sampling rate: {info['sampling_rate']} Hz")
            print(f"Duration: {info['duration']:.2f} seconds")
