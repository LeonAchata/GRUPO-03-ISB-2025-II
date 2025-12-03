"""
Script de entrenamiento de modelos para clasificación de Motor Imagery
Entrena LDA, SVM y Random Forest con validación cruzada LORO
"""

import numpy as np
from pathlib import Path
import sys
import joblib
from collections import defaultdict

sys.path.append('.')

from src.features.csp import CSPExtractor, MultiClassCSP
from src.features.spectral import BandPowerExtractor
from src.models.traditional import LDAClassifier, SVMClassifier, RFClassifier
from src.validation.cross_validation import LeaveOneRunOut, cross_validate_loro
from src.validation.metrics import ClassificationMetrics
from src.utils.config import load_config

def load_preprocessed_data(processed_dir, task_type='hands'):
    """
    Carga todos los datos preprocesados para un tipo de tarea.
    
    Parameters
    ----------
    processed_dir : Path
        Directorio con datos procesados
    task_type : str
        'hands' o 'fists_feet'
    
    Returns
    -------
    X : ndarray (n_epochs, n_channels, n_times)
    y : ndarray (n_epochs,)
    run_labels : ndarray (n_epochs,)
    subject_labels : ndarray (n_epochs,)
    """
    print(f"\n{'='*70}")
    print(f"CARGANDO DATOS: {task_type}")
    print(f"{'='*70}")
    
    all_X = []
    all_y = []
    all_runs = []
    all_subjects = []
    channel_counts = []
    
    # Iterar sobre todos los sujetos
    subject_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
    
    for subject_dir in sorted(subject_dirs):
        task_dir = subject_dir / task_type
        
        if not task_dir.exists():
            continue
        
        subject = subject_dir.name
        
        # Cargar todos los archivos .npz de este sujeto
        npz_files = sorted(task_dir.glob('*.npz'))
        
        for npz_file in npz_files:
            data = np.load(npz_file)
            X = data['X']
            y = data['y']
            run = str(data['run'])
            
            all_X.append(X)
            all_y.append(y)
            all_runs.extend([run] * len(y))
            all_subjects.extend([subject] * len(y))
            channel_counts.append(X.shape[1])
            
            print(f"  ✓ {subject}/{run}: {len(y)} epochs, {X.shape[1]} channels")
    
    # Encontrar el número mínimo de canales y tiempo
    min_channels = min(channel_counts)
    time_lengths = [X.shape[2] for X in all_X]
    min_time = min(time_lengths)
    
    print(f"\n⚠️  Número de canales varía: {min(channel_counts)} - {max(channel_counts)}")
    print(f"⚠️  Duración temporal varía: {min(time_lengths)} - {max(time_lengths)} muestras")
    print(f"  → Recortando todos a {min_channels} canales y {min_time} muestras")
    
    # Recortar todos los datos al mismo número de canales y tiempo
    all_X_trimmed = [X[:, :min_channels, :min_time] for X in all_X]
    
    # Concatenar todo
    X = np.concatenate(all_X_trimmed, axis=0)
    y = np.concatenate(all_y, axis=0)
    run_labels = np.array(all_runs)
    subject_labels = np.array(all_subjects)
    
    print(f"\n📊 DATOS CARGADOS:")
    print(f"  Total epochs: {len(y)}")
    print(f"  Shape: {X.shape}")
    print(f"  Sujetos únicos: {len(np.unique(subject_labels))}")
    print(f"  Runs únicos: {np.unique(run_labels)}")
    print(f"  Clases: {np.unique(y)} (counts: {np.bincount(y)})")
    
    return X, y, run_labels, subject_labels


def extract_features(X, y, feature_type='csp', n_components=6):
    """
    Extrae características de los datos.
    
    Parameters
    ----------
    X : ndarray (n_epochs, n_channels, n_times)
    y : ndarray (n_epochs,)
    feature_type : str
        'csp', 'bandpower', o 'both'
    n_components : int
        Número de componentes CSP
    
    Returns
    -------
    X_features : ndarray (n_epochs, n_features)
    """
    print(f"\n{'='*70}")
    print(f"EXTRAYENDO CARACTERÍSTICAS: {feature_type}")
    print(f"{'='*70}")
    
    features_list = []
    
    if feature_type in ['csp', 'both']:
        print(f"  → CSP ({n_components} componentes)...")
        n_classes = len(np.unique(y))
        
        if n_classes == 2:
            csp = CSPExtractor(n_components=n_components)
            csp.fit(X, y)
            X_csp = csp.transform(X)
        else:
            csp = MultiClassCSP(n_components=n_components)
            csp.fit(X, y)
            X_csp = csp.transform(X)
        
        features_list.append(X_csp)
        print(f"    ✓ CSP features shape: {X_csp.shape}")
    
    if feature_type in ['bandpower', 'both']:
        print(f"  → Band Power (mu/beta)...")
        bp_extractor = BandPowerExtractor(
            sfreq=160, 
            bands={'mu': (8, 13), 'beta': (13, 30)},
            relative=True
        )
        X_bp = bp_extractor.transform(X)
        features_list.append(X_bp)
        print(f"    ✓ Band Power features shape: {X_bp.shape}")
    
    # Concatenar características
    X_features = np.hstack(features_list)
    print(f"\n  ✓ Features finales shape: {X_features.shape}")
    
    return X_features


def train_and_evaluate(X, y, run_labels, subject_labels, task_type, output_dir):
    """
    Entrena y evalúa múltiples modelos con validación LORO.
    """
    print(f"\n{'='*70}")
    print(f"ENTRENAMIENTO Y VALIDACIÓN - {task_type}")
    print(f"{'='*70}")
    
    # Modelos a entrenar
    models = {
        'LDA': LDAClassifier(solver='svd'),
        'SVM_RBF': SVMClassifier(kernel='rbf', C=1.0),
        'SVM_Linear': SVMClassifier(kernel='linear', C=1.0),
        'RandomForest': RFClassifier(n_estimators=100, max_depth=10)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'─'*70}")
        print(f"🤖 MODELO: {model_name}")
        print(f"{'─'*70}")
        
        # Validación cruzada LORO
        print(f"  Ejecutando Leave-One-Run-Out CV...")
        cv_results = cross_validate_loro(model, X, y, run_labels, return_predictions=True)
        
        # Concatenar predicciones de todos los folds
        y_true_all = np.concatenate(cv_results['true_labels'])
        y_pred_all = np.concatenate(cv_results['predictions'])
        
        # Calcular métricas
        metrics = ClassificationMetrics(y_true_all, y_pred_all)
        metrics_dict = metrics.compute_all()
        
        # Mostrar resultados
        print(f"\n  📊 RESULTADOS {model_name}:")
        print(f"    Accuracy:  {metrics_dict['accuracy']:.4f}")
        print(f"    Precision: {metrics_dict['precision']:.4f}")
        print(f"    Recall:    {metrics_dict['recall']:.4f}")
        print(f"    F1-Score:  {metrics_dict['f1_score']:.4f}")
        print(f"    Kappa:     {metrics_dict['kappa']:.4f}")
        
        # Guardar
        results[model_name] = {
            'metrics': metrics_dict,
            'fold_scores': cv_results['scores'],
            'confusion_matrix': metrics_dict['confusion_matrix']
        }
        
        # Entrenar modelo final con todos los datos
        print(f"  → Entrenando modelo final con todos los datos...")
        model.fit(X, y)
        
        # Guardar modelo
        model_file = output_dir / f"{task_type}_{model_name.lower()}.pkl"
        joblib.dump(model, model_file)
        print(f"  💾 Modelo guardado: {model_file}")
    
    return results


def main():
    """
    Pipeline completo de entrenamiento.
    """
    print("="*70)
    print("ENTRENAMIENTO DE MODELOS - EEG Motor Imagery")
    print("="*70)
    
    # Configuración
    config = load_config('config.yaml')
    processed_dir = Path(config['data']['processed_path'])
    output_dir = Path(config['output']['models'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSP parameters
    n_components = config['features']['csp']['n_components']
    
    # ====================================================================
    # TAREA 1: Clasificación de Manos (izquierda vs derecha)
    # ====================================================================
    print(f"\n{'#'*70}")
    print("# TAREA 1: CLASIFICACIÓN DE MANOS (Izquierda vs Derecha)")
    print(f"{'#'*70}")
    
    X_hands, y_hands, runs_hands, subjects_hands = load_preprocessed_data(
        processed_dir, task_type='hands'
    )
    
    # Extraer características
    X_hands_features = extract_features(
        X_hands, y_hands, 
        feature_type='both',  # CSP + Band Power
        n_components=n_components
    )
    
    # Entrenar y evaluar
    results_hands = train_and_evaluate(
        X_hands_features, y_hands, runs_hands, subjects_hands,
        task_type='hands',
        output_dir=output_dir
    )
    
    # ====================================================================
    # TAREA 2: Clasificación Puños/Pies (ambas manos vs ambos pies)
    # ====================================================================
    print(f"\n{'#'*70}")
    print("# TAREA 2: CLASIFICACIÓN PUÑOS/PIES (Ambas Manos vs Ambos Pies)")
    print(f"{'#'*70}")
    
    X_fists, y_fists, runs_fists, subjects_fists = load_preprocessed_data(
        processed_dir, task_type='fists_feet'
    )
    
    # Extraer características
    X_fists_features = extract_features(
        X_fists, y_fists,
        feature_type='both',
        n_components=n_components
    )
    
    # Entrenar y evaluar
    results_fists = train_and_evaluate(
        X_fists_features, y_fists, runs_fists, subjects_fists,
        task_type='fists_feet',
        output_dir=output_dir
    )
    
    # ====================================================================
    # RESUMEN FINAL
    # ====================================================================
    print(f"\n{'='*70}")
    print("📊 RESUMEN COMPARATIVO")
    print(f"{'='*70}")
    
    print(f"\n{'TAREA 1: MANOS (Izq vs Der)':^70}")
    print("─"*70)
    for model_name, result in results_hands.items():
        acc = result['metrics']['accuracy']
        f1 = result['metrics']['f1_score']
        print(f"  {model_name:15s} → Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    print(f"\n{'TAREA 2: PUÑOS/PIES (Ambas Manos vs Ambos Pies)':^70}")
    print("─"*70)
    for model_name, result in results_fists.items():
        acc = result['metrics']['accuracy']
        f1 = result['metrics']['f1_score']
        print(f"  {model_name:15s} → Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    # Guardar resumen
    summary_file = output_dir / 'training_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("TRAINING SUMMARY - EEG Motor Imagery Classification\n")
        f.write("="*70 + "\n\n")
        
        f.write("TAREA 1: MANOS (Izquierda vs Derecha)\n")
        f.write("-"*70 + "\n")
        for model_name, result in results_hands.items():
            f.write(f"{model_name}:\n")
            for metric, value in result['metrics'].items():
                if metric != 'confusion_matrix':
                    f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")
        
        f.write("\nTAREA 2: PUÑOS/PIES (Ambas Manos vs Ambos Pies)\n")
        f.write("-"*70 + "\n")
        for model_name, result in results_fists.items():
            f.write(f"{model_name}:\n")
            for metric, value in result['metrics'].items():
                if metric != 'confusion_matrix':
                    f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")
    
    print(f"\n💾 Resumen guardado: {summary_file}")
    print(f"\n✅ ENTRENAMIENTO COMPLETADO")
    print(f"📁 Modelos guardados en: {output_dir.absolute()}")
    print("="*70)


if __name__ == "__main__":
    main()
