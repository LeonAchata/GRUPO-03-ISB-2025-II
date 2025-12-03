"""
Cross-Validation Module for EEG-BCI Project.

Implements various cross-validation strategies:
- Leave-One-Run-Out (LORO) for intra-subject validation
- K-Fold for standard validation
- Leave-One-Subject-Out (LOSO) for inter-subject validation
"""

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import BaseEstimator, clone
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict


class LeaveOneRunOut:
    """
    Leave-One-Run-Out (LORO) cross-validation.
    
    Each iteration uses one run as test set and remaining runs as training set.
    """
    
    def __init__(self, n_splits: Optional[int] = None):
        """
        Initialize LORO cross-validator.
        
        Parameters
        ----------
        n_splits : int, optional
            Number of splits (auto-detected if None)
        """
        self.n_splits = n_splits
        
    def split(self, X: np.ndarray, y: np.ndarray, 
             run_labels: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits.
        
        Parameters
        ----------
        X : ndarray
            Features (not used, for API consistency)
        y : ndarray
            Labels (not used, for API consistency)
        run_labels : ndarray
            Run identifier for each sample
            
        Returns
        -------
        splits : list of tuples
            List of (train_indices, test_indices) tuples
        """
        unique_runs = np.unique(run_labels)
        
        if self.n_splits is None:
            self.n_splits = len(unique_runs)
        
        splits = []
        for run in unique_runs:
            test_idx = np.where(run_labels == run)[0]
            train_idx = np.where(run_labels != run)[0]
            splits.append((train_idx, test_idx))
        
        return splits
    
    def get_n_splits(self, X: Optional[np.ndarray] = None, 
                    y: Optional[np.ndarray] = None,
                    run_labels: Optional[np.ndarray] = None) -> int:
        """Get number of splits."""
        if self.n_splits is not None:
            return self.n_splits
        elif run_labels is not None:
            return len(np.unique(run_labels))
        else:
            raise ValueError("Cannot determine n_splits")


class LeaveOneSubjectOut:
    """
    Leave-One-Subject-Out (LOSO) cross-validation.
    
    Each iteration uses one subject as test set and remaining subjects as training set.
    """
    
    def __init__(self):
        """Initialize LOSO cross-validator."""
        pass
        
    def split(self, X: np.ndarray, y: np.ndarray,
             subject_labels: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits.
        
        Parameters
        ----------
        X : ndarray
            Features (not used, for API consistency)
        y : ndarray
            Labels (not used, for API consistency)
        subject_labels : ndarray
            Subject identifier for each sample
            
        Returns
        -------
        splits : list of tuples
            List of (train_indices, test_indices) tuples
        """
        unique_subjects = np.unique(subject_labels)
        
        splits = []
        for subject in unique_subjects:
            test_idx = np.where(subject_labels == subject)[0]
            train_idx = np.where(subject_labels != subject)[0]
            splits.append((train_idx, test_idx))
        
        return splits
    
    def get_n_splits(self, X: Optional[np.ndarray] = None,
                    y: Optional[np.ndarray] = None,
                    subject_labels: Optional[np.ndarray] = None) -> int:
        """Get number of splits."""
        if subject_labels is not None:
            return len(np.unique(subject_labels))
        else:
            raise ValueError("Cannot determine n_splits without subject_labels")


def cross_validate_loro(model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                       run_labels: np.ndarray,
                       return_predictions: bool = False) -> Dict[str, Any]:
    """
    Perform Leave-One-Run-Out cross-validation.
    
    Parameters
    ----------
    model : BaseEstimator
        Classifier to evaluate
    X : ndarray, shape (n_samples, n_features)
        Features
    y : ndarray, shape (n_samples,)
        Labels
    run_labels : ndarray, shape (n_samples,)
        Run identifier for each sample
    return_predictions : bool
        Whether to return predictions for each fold
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'scores': accuracy for each fold
        - 'mean_score': mean accuracy
        - 'std_score': standard deviation of accuracy
        - 'predictions': predictions for each fold (if return_predictions=True)
        - 'true_labels': true labels for each fold (if return_predictions=True)
    """
    loro = LeaveOneRunOut()
    splits = loro.split(X, y, run_labels)
    
    scores = []
    all_predictions = []
    all_true_labels = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Clone and fit model
        model_fold = clone(model)
        model_fold.fit(X_train, y_train)
        
        # Predict and score
        y_pred = model_fold.predict(X_test)
        score = np.mean(y_pred == y_test)
        scores.append(score)
        
        if return_predictions:
            all_predictions.append(y_pred)
            all_true_labels.append(y_test)
        
        print(f"Fold {fold_idx + 1}/{len(splits)}: Accuracy = {score:.4f}")
    
    results = {
        'scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores)
    }
    
    if return_predictions:
        results['predictions'] = all_predictions
        results['true_labels'] = all_true_labels
    
    return results


def cross_validate_kfold(model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                        n_splits: int = 5, stratified: bool = True,
                        random_state: int = 42) -> Dict[str, Any]:
    """
    Perform K-Fold cross-validation.
    
    Parameters
    ----------
    model : BaseEstimator
        Classifier to evaluate
    X : ndarray, shape (n_samples, n_features)
        Features
    y : ndarray, shape (n_samples,)
        Labels
    n_splits : int
        Number of folds
    stratified : bool
        Whether to use stratified K-Fold
    random_state : int
        Random seed
        
    Returns
    -------
    results : dict
        Dictionary containing scores and statistics
    """
    if stratified:
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                               random_state=random_state)
    else:
        kfold = KFold(n_splits=n_splits, shuffle=True, 
                     random_state=random_state)
    
    scores = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model_fold = clone(model)
        model_fold.fit(X_train, y_train)
        
        score = model_fold.score(X_test, y_test)
        scores.append(score)
        
        print(f"Fold {fold_idx + 1}/{n_splits}: Accuracy = {score:.4f}")
    
    results = {
        'scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores)
    }
    
    return results


def cross_validate_loso(model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                       subject_labels: np.ndarray) -> Dict[str, Any]:
    """
    Perform Leave-One-Subject-Out cross-validation.
    
    Parameters
    ----------
    model : BaseEstimator
        Classifier to evaluate
    X : ndarray, shape (n_samples, n_features)
        Features
    y : ndarray, shape (n_samples,)
        Labels
    subject_labels : ndarray, shape (n_samples,)
        Subject identifier for each sample
        
    Returns
    -------
    results : dict
        Dictionary containing scores and statistics
    """
    loso = LeaveOneSubjectOut()
    splits = loso.split(X, y, subject_labels)
    
    scores = []
    subject_scores = {}
    
    unique_subjects = np.unique(subject_labels)
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        subject = unique_subjects[fold_idx]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model_fold = clone(model)
        model_fold.fit(X_train, y_train)
        
        score = model_fold.score(X_test, y_test)
        scores.append(score)
        subject_scores[subject] = score
        
        print(f"Subject {subject}: Accuracy = {score:.4f}")
    
    results = {
        'scores': scores,
        'subject_scores': subject_scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores)
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    from src.data.loader import load_eeg_data
    from src.data.parser import EventParser
    from src.preprocessing.filters import filter_eeg_data
    from src.preprocessing.segmentation import extract_epochs
    from src.features.csp import CSPExtractor
    from src.models.traditional import LDAClassifier
    
    print("Loading data from multiple runs...")
    
    # Load data from multiple runs
    runs = ["R04", "R08", "R12"]
    all_X = []
    all_y = []
    run_ids = []
    
    for run_idx, run in enumerate(runs):
        raw = load_eeg_data("S001", run, data_dir="data/raw")
        raw_filtered = filter_eeg_data(raw)
        
        parser = EventParser("data/raw")
        events, event_id = parser.parse_events_from_raw(raw_filtered)
        _, class_labels = parser.map_events_to_classes(events, event_id, run)
        
        epochs = extract_epochs(raw_filtered, events, class_labels)
        X = epochs.get_data()
        y = epochs.events[:, 2]
        
        all_X.append(X)
        all_y.append(y)
        run_ids.extend([run_idx] * len(y))
    
    # Concatenate all data
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    run_labels = np.array(run_ids)
    
    print(f"\nTotal data shape: {X_all.shape}")
    print(f"Runs: {np.unique(run_labels)}")
    
    # Extract CSP features
    print("\nExtracting CSP features...")
    csp = CSPExtractor(n_components=6)
    X_csp = csp.fit_transform(X_all, y_all)
    
    # Test LORO cross-validation
    print("\n" + "="*60)
    print("Leave-One-Run-Out Cross-Validation")
    print("="*60)
    
    model = LDAClassifier()
    results = cross_validate_loro(model, X_csp, y_all, run_labels)
    
    print(f"\nMean Accuracy: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
