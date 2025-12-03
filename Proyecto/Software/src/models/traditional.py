"""
Traditional Machine Learning Models for EEG-BCI Classification.

Implements standard classifiers: LDA, SVM, Random Forest.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional, Dict, Any


class LDAClassifier(BaseEstimator, ClassifierMixin):
    """
    Linear Discriminant Analysis classifier.
    
    LDA is a common baseline for BCI applications due to its simplicity
    and effectiveness with CSP features.
    """
    
    def __init__(self, solver: str = 'svd', shrinkage: Optional[str] = None):
        """
        Initialize LDA classifier.
        
        Parameters
        ----------
        solver : str
            Solver to use ('svd', 'lsqr', 'eigen')
        shrinkage : str or float, optional
            Shrinkage parameter for regularization
        """
        self.solver = solver
        self.shrinkage = shrinkage
        self.model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LDAClassifier':
        """
        Fit LDA model.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training features
        y : ndarray, shape (n_samples,)
            Training labels
            
        Returns
        -------
        self : LDAClassifier
        """
        self.model = LinearDiscriminantAnalysis(
            solver=self.solver,
            shrinkage=self.shrinkage
        )
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Test features
            
        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : ndarray
            Test features
            
        Returns
        -------
        proba : ndarray, shape (n_samples, n_classes)
            Class probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.
        
        Parameters
        ----------
        X : ndarray
            Test features
        y : ndarray
            True labels
            
        Returns
        -------
        score : float
            Accuracy score
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.score(X, y)


class SVMClassifier(BaseEstimator, ClassifierMixin):
    """
    Support Vector Machine classifier.
    """
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, 
                 gamma: str = 'scale', probability: bool = True):
        """
        Initialize SVM classifier.
        
        Parameters
        ----------
        kernel : str
            Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
        C : float
            Regularization parameter
        gamma : str or float
            Kernel coefficient
        probability : bool
            Whether to enable probability estimates
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability
        self.model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifier':
        """Fit SVM model."""
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=self.probability
        )
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if not self.probability:
            raise ValueError("Probability not enabled. Set probability=True.")
        return self.model.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.score(X, y)


class RFClassifier(BaseEstimator, ClassifierMixin):
    """
    Random Forest classifier.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 10,
                 random_state: int = 42):
        """
        Initialize Random Forest classifier.
        
        Parameters
        ----------
        n_estimators : int
            Number of trees
        max_depth : int, optional
            Maximum tree depth
        random_state : int
            Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RFClassifier':
        """Fit Random Forest model."""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.score(X, y)


def create_classifier(model_type: str, **kwargs) -> BaseEstimator:
    """
    Factory function to create classifiers.
    
    Parameters
    ----------
    model_type : str
        Type of classifier ('lda', 'svm', 'rf')
    **kwargs : dict
        Model parameters
        
    Returns
    -------
    classifier : BaseEstimator
        Classifier instance
    """
    if model_type.lower() == 'lda':
        return LDAClassifier(**kwargs)
    elif model_type.lower() == 'svm':
        return SVMClassifier(**kwargs)
    elif model_type.lower() in ['rf', 'random_forest']:
        return RFClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Example usage
    from src.data.loader import load_eeg_data
    from src.data.parser import EventParser
    from src.preprocessing.filters import filter_eeg_data
    from src.preprocessing.segmentation import extract_epochs
    from src.features.csp import CSPExtractor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    print("Loading and preprocessing data...")
    raw = load_eeg_data("S001", "R04", data_dir="data/raw")
    raw_filtered = filter_eeg_data(raw)
    
    parser = EventParser("data/raw")
    events, event_id = parser.parse_events_from_raw(raw_filtered)
    _, class_labels = parser.map_events_to_classes(events, event_id, "R04")
    
    epochs = extract_epochs(raw_filtered, events, class_labels)
    X = epochs.get_data()
    y = epochs.events[:, 2]
    
    print(f"Data shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    
    # Extract CSP features
    print("\nExtracting CSP features...")
    csp = CSPExtractor(n_components=6)
    X_csp = csp.fit_transform(X, y)
    print(f"CSP features shape: {X_csp.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_csp, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Test classifiers
    models = {
        'LDA': LDAClassifier(),
        'SVM (RBF)': SVMClassifier(kernel='rbf', C=1.0),
        'SVM (Linear)': SVMClassifier(kernel='linear', C=1.0),
        'Random Forest': RFClassifier(n_estimators=100, max_depth=10)
    }
    
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n{name}:")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Accuracy: {acc:.4f}")
