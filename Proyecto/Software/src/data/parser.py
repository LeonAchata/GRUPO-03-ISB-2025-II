"""
Event Parser for EEG-BCI Motor Imagery Project.

This module provides functions to parse event annotations from EDF files
and .event files, mapping them to motor imagery classes.
"""

import numpy as np
import mne
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings


class EventParser:
    """Class for parsing and handling EEG event annotations."""
    
    # Event code mapping from dataset
    EVENT_CODES = {
        'T0': 0,  # Rest
        'T1': 1,  # Left fist OR Both fists (depending on run)
        'T2': 2   # Right fist OR Both feet (depending on run)
    }
    
    # Run type classification
    HAND_RUNS = ['R03', 'R04', 'R07', 'R08', 'R11', 'R12']  # Left/Right hand
    FIST_FEET_RUNS = ['R05', 'R06', 'R09', 'R10', 'R13', 'R14']  # Both hands/feet
    
    # Motor imagery runs (imagined movements)
    IMAGERY_RUNS = ['R04', 'R06', 'R08', 'R10', 'R12', 'R14']
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the event parser.
        
        Parameters
        ----------
        data_dir : str
            Path to the directory containing subject folders
        """
        self.data_dir = Path(data_dir)
    
    def get_run_type(self, run: str) -> str:
        """
        Determine the type of run (hand or fist_feet).
        
        Parameters
        ----------
        run : str
            Run ID (e.g., 'R04')
            
        Returns
        -------
        run_type : str
            Either 'hand' or 'fist_feet'
        """
        if run.upper() in self.HAND_RUNS:
            return 'hand'
        elif run.upper() in self.FIST_FEET_RUNS:
            return 'fist_feet'
        else:
            return 'unknown'
    
    def parse_events_from_raw(self, raw: mne.io.Raw) -> Tuple[np.ndarray, Dict]:
        """
        Extract events from MNE Raw object annotations.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data with annotations
            
        Returns
        -------
        events : ndarray, shape (n_events, 3)
            MNE events array [sample, 0, event_id]
        event_id : dict
            Mapping of event descriptions to IDs
        """
        # Get events from annotations
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        return events, event_id
    
    def map_events_to_classes(self, events: np.ndarray, event_id: Dict, 
                              run: str) -> Tuple[np.ndarray, List[str]]:
        """
        Map raw events to motor imagery class labels.
        
        Parameters
        ----------
        events : ndarray
            MNE events array
        event_id : dict
            Event ID mapping
        run : str
            Run ID to determine class mapping
            
        Returns
        -------
        events_mapped : ndarray
            Events with updated class IDs
        class_labels : list of str
            Corresponding class labels for each event
        """
        run_type = self.get_run_type(run)
        
        # Create inverse mapping (ID -> description)
        id_to_desc = {v: k for k, v in event_id.items()}
        
        class_labels = []
        events_mapped = events.copy()
        
        for i, event in enumerate(events):
            event_desc = id_to_desc.get(event[2], 'unknown')
            
            # Map T0 (rest) - skip for classification
            if 'T0' in event_desc:
                class_labels.append('rest')
                continue
            
            # Map T1 and T2 based on run type
            if 'T1' in event_desc:
                if run_type == 'hand':
                    class_labels.append('left_hand')
                elif run_type == 'fist_feet':
                    class_labels.append('both_hands')
                else:
                    class_labels.append('unknown')
                    
            elif 'T2' in event_desc:
                if run_type == 'hand':
                    class_labels.append('right_hand')
                elif run_type == 'fist_feet':
                    class_labels.append('both_feet')
                else:
                    class_labels.append('unknown')
            else:
                class_labels.append('unknown')
        
        return events_mapped, class_labels
    
    def get_class_events(self, events: np.ndarray, class_labels: List[str],
                         exclude_rest: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter events to get only motor imagery events (exclude rest).
        
        Parameters
        ----------
        events : ndarray
            MNE events array
        class_labels : list of str
            Class labels for each event
        exclude_rest : bool
            Whether to exclude rest (T0) events
            
        Returns
        -------
        filtered_events : ndarray
            Filtered events array
        filtered_labels : ndarray
            Corresponding labels as integers
        """
        # Create label encoding
        label_map = {
            'left_hand': 0,
            'right_hand': 1,
            'both_hands': 2,
            'both_feet': 3,
            'rest': 4
        }
        
        # Filter events
        mask = np.array([label != 'rest' and label != 'unknown' 
                        for label in class_labels])
        
        if not exclude_rest:
            mask = np.array([label != 'unknown' for label in class_labels])
        
        filtered_events = events[mask]
        filtered_labels = np.array([label_map[label] for label in class_labels 
                                   if mask[class_labels.index(label)]])
        
        return filtered_events, filtered_labels
    
    def read_event_file(self, subject: str, run: str) -> List[Tuple[float, str]]:
        """
        Read events from .event file (alternative to annotations in EDF).
        
        Parameters
        ----------
        subject : str
            Subject ID
        run : str
            Run ID
            
        Returns
        -------
        events : list of tuples
            List of (timestamp, event_code) tuples
        """
        filename = f"{subject}{run}.edf.event"
        filepath = self.data_dir / subject / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Event file not found: {filepath}")
        
        events = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    timestamp = float(parts[0])
                    event_code = parts[2]
                    events.append((timestamp, event_code))
        
        return events
    
    def get_event_summary(self, events: np.ndarray, class_labels: List[str]) -> Dict:
        """
        Get summary statistics of events.
        
        Parameters
        ----------
        events : ndarray
            Events array
        class_labels : list of str
            Class labels
            
        Returns
        -------
        summary : dict
            Dictionary with event counts and statistics
        """
        from collections import Counter
        
        label_counts = Counter(class_labels)
        
        summary = {
            'total_events': len(events),
            'class_counts': dict(label_counts),
            'unique_classes': list(label_counts.keys()),
            'n_classes': len([k for k in label_counts.keys() 
                            if k not in ['rest', 'unknown']])
        }
        
        return summary


def parse_events(raw: mne.io.Raw, run: str) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Convenience function to parse events from raw data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    run : str
        Run ID
        
    Returns
    -------
    events : ndarray
        Events array
    class_labels : list of str
        Class labels
    summary : dict
        Event summary
    """
    parser = EventParser()
    events, event_id = parser.parse_events_from_raw(raw)
    events_mapped, class_labels = parser.map_events_to_classes(events, event_id, run)
    summary = parser.get_event_summary(events_mapped, class_labels)
    
    return events_mapped, class_labels, summary


if __name__ == "__main__":
    # Example usage
    from src.data.loader import EEGLoader
    
    loader = EEGLoader("data")
    parser = EventParser("data")
    
    # Load a sample run
    subject = "S001"
    run = "R04"  # Motor imagery run (left/right hand)
    
    print(f"Loading {subject}{run}...")
    raw = loader.load_raw(subject, run)
    
    # Parse events
    events, event_id = parser.parse_events_from_raw(raw)
    print(f"\nFound {len(events)} events")
    print(f"Event IDs: {event_id}")
    
    # Map to classes
    events_mapped, class_labels = parser.map_events_to_classes(events, event_id, run)
    print(f"\nClass labels (first 10): {class_labels[:10]}")
    
    # Get summary
    summary = parser.get_event_summary(events_mapped, class_labels)
    print(f"\nEvent summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
