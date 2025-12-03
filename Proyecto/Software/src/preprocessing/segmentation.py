"""
Segmentation Module for EEG-BCI Project.

Implements epoching (segmentation) of continuous EEG data based on events.
"""

import numpy as np
import mne
from typing import Optional, List, Dict, Tuple


class EpochExtractor:
    """Class for extracting epochs from continuous EEG data."""
    
    def __init__(self, tmin: float = -0.5, tmax: float = 4.0,
                 baseline: Optional[Tuple[float, float]] = (-0.5, 0.0)):
        """
        Initialize epoch extractor.
        
        Parameters
        ----------
        tmin : float
            Start time before event (seconds)
        tmax : float
            End time after event (seconds)
        baseline : tuple or None
            Baseline correction window (start, end) in seconds
        """
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        
    def create_epochs(self, raw: mne.io.Raw, events: np.ndarray,
                     event_id: Optional[Dict] = None,
                     picks: Optional[List[str]] = None,
                     preload: bool = True) -> mne.Epochs:
        """
        Create epochs from raw data and events.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Continuous EEG data
        events : ndarray
            Events array (n_events, 3)
        event_id : dict, optional
            Mapping of event names to IDs
        picks : list of str, optional
            Channels to include
        preload : bool
            Whether to preload data into memory
            
        Returns
        -------
        epochs : mne.Epochs
            Epoched data
        """
        # Create epochs
        epochs = mne.Epochs(raw, events, event_id=event_id,
                           tmin=self.tmin, tmax=self.tmax,
                           baseline=self.baseline,
                           picks=picks, preload=preload,
                           verbose=False)
        
        return epochs
    
    def create_epochs_from_classes(self, raw: mne.io.Raw, 
                                   events: np.ndarray,
                                   class_labels: List[str],
                                   exclude_rest: bool = True) -> mne.Epochs:
        """
        Create epochs with motor imagery class labels.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Continuous EEG data
        events : ndarray
            Events array
        class_labels : list of str
            Class labels for each event
        exclude_rest : bool
            Whether to exclude rest events
            
        Returns
        -------
        epochs : mne.Epochs
            Epoched data with class labels
        """
        # Create label mapping
        label_map = {
            'left_hand': 1,
            'right_hand': 2,
            'both_hands': 3,
            'both_feet': 4,
            'rest': 0
        }
        
        # Filter events based on classes
        if exclude_rest:
            mask = np.array([label not in ['rest', 'unknown'] 
                           for label in class_labels])
            filtered_events = events[mask].copy()
            filtered_labels = [label for label in class_labels if label not in ['rest', 'unknown']]
        else:
            filtered_events = events.copy()
            filtered_labels = class_labels
        
        # Update event IDs in events array
        for i, label in enumerate(filtered_labels):
            filtered_events[i, 2] = label_map.get(label, 0)
        
        # Create event_id dict
        unique_labels = set(filtered_labels)
        event_id = {label: label_map[label] for label in unique_labels 
                   if label in label_map}
        
        # Create epochs
        epochs = self.create_epochs(raw, filtered_events, event_id=event_id)
        
        return epochs
    
    def extract_motor_imagery_epochs(self, raw: mne.io.Raw,
                                    events: np.ndarray,
                                    class_labels: List[str],
                                    motor_channels: Optional[List[str]] = None) -> mne.Epochs:
        """
        Extract epochs specifically for motor imagery tasks.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Continuous EEG data
        events : ndarray
            Events array
        class_labels : list of str
            Class labels
        motor_channels : list of str, optional
            Motor cortex channel names
            
        Returns
        -------
        epochs : mne.Epochs
            Motor imagery epochs
        """
        # Select motor channels if specified
        picks = None
        if motor_channels is not None:
            available_channels = [ch for ch in motor_channels if ch in raw.ch_names]
            if available_channels:
                picks = available_channels
                print(f"Using {len(picks)} motor channels: {picks}")
        
        # Create epochs
        epochs = self.create_epochs_from_classes(raw, events, class_labels,
                                                exclude_rest=True)
        
        return epochs
    
    def balance_classes(self, epochs: mne.Epochs) -> mne.Epochs:
        """
        Balance classes by undersampling majority class.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Epoched data
            
        Returns
        -------
        epochs_balanced : mne.Epochs
            Balanced epochs
        """
        # Get class counts
        class_ids = epochs.events[:, 2]
        unique_ids, counts = np.unique(class_ids, return_counts=True)
        
        print(f"\nOriginal class distribution:")
        for id_val, count in zip(unique_ids, counts):
            print(f"  Class {id_val}: {count} epochs")
        
        # Find minimum count
        min_count = np.min(counts)
        
        # Sample epochs for each class
        balanced_indices = []
        for id_val in unique_ids:
            class_indices = np.where(class_ids == id_val)[0]
            sampled_indices = np.random.choice(class_indices, size=min_count, 
                                             replace=False)
            balanced_indices.extend(sampled_indices)
        
        # Sort indices to maintain temporal order
        balanced_indices = sorted(balanced_indices)
        
        # Create balanced epochs
        epochs_balanced = epochs[balanced_indices]
        
        print(f"\nBalanced class distribution:")
        class_ids_balanced = epochs_balanced.events[:, 2]
        unique_ids_balanced, counts_balanced = np.unique(class_ids_balanced, 
                                                        return_counts=True)
        for id_val, count in zip(unique_ids_balanced, counts_balanced):
            print(f"  Class {id_val}: {count} epochs")
        
        return epochs_balanced
    
    def get_epoch_data(self, epochs: mne.Epochs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get epoch data and labels as numpy arrays.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Epoched data
            
        Returns
        -------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            Epoch data
        y : ndarray, shape (n_epochs,)
            Class labels
        """
        X = epochs.get_data()
        y = epochs.events[:, 2]
        
        return X, y


def extract_epochs(raw: mne.io.Raw, events: np.ndarray,
                  class_labels: List[str],
                  tmin: float = -0.5, tmax: float = 4.0,
                  baseline: Optional[Tuple[float, float]] = (-0.5, 0.0)) -> mne.Epochs:
    """
    Convenience function to extract epochs.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Continuous EEG data
    events : ndarray
        Events array
    class_labels : list of str
        Class labels for each event
    tmin : float
        Start time before event
    tmax : float
        End time after event
    baseline : tuple or None
        Baseline correction window
        
    Returns
    -------
    epochs : mne.Epochs
        Epoched data
    """
    extractor = EpochExtractor(tmin=tmin, tmax=tmax, baseline=baseline)
    return extractor.create_epochs_from_classes(raw, events, class_labels)


if __name__ == "__main__":
    # Example usage
    from src.data.loader import load_eeg_data
    from src.data.parser import EventParser
    from src.preprocessing.filters import filter_eeg_data
    
    # Load data
    print("Loading EEG data...")
    raw = load_eeg_data("S001", "R04", data_dir="data/raw")
    
    # Filter
    print("Filtering...")
    raw_filtered = filter_eeg_data(raw)
    
    # Parse events
    print("Parsing events...")
    parser = EventParser("data/raw")
    events, event_id = parser.parse_events_from_raw(raw_filtered)
    _, class_labels = parser.map_events_to_classes(events, event_id, "R04")
    
    # Extract epochs
    print("\nExtracting epochs...")
    extractor = EpochExtractor(tmin=-0.5, tmax=4.0)
    epochs = extractor.create_epochs_from_classes(raw_filtered, events, class_labels)
    
    print(f"\nExtracted {len(epochs)} epochs")
    print(epochs)
    
    # Get data
    X, y = extractor.get_epoch_data(epochs)
    print(f"\nData shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Unique labels: {np.unique(y)}")
