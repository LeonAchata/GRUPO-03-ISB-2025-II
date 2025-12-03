"""
Script de entrenamiento para modelo de 4 clases
Clasifica: Mano Izquierda, Mano Derecha, Ambas Manos, Ambos Pies
"""

import numpy as np
from pathlib import Path
import sys
import joblib

sys.path.append('.')

from src.features.csp import MultiClassCSP
from src.features.spectral import BandPowerExtractor
from src.models.traditional import LDAClassifier, SVMClassifier, RFClassifier
from src.validation.cross_validation import cross_validate_loro
from src.validation.metrics import ClassificationMetrics
from src.utils.config import load_config

def load_all_data(processed_dir):
    """
    Carga TODOS los datos (hands + fists_feet) con etiquetas correctas.
    
    Returns
    -------
    X : ndarray (n_epochs, n_channels, n_times)
    y : ndarray (n_epochs,) - 0: left_hand, 1: right_hand, 2: both_hands, 3: both_feet
    run_labels : ndarray (n_epochs,)
    subject_labels : ndarray (n_epochs,)
    """
    print(f"\n{'='*70}")
    print("CARGANDO TODOS LOS DATOS (4 CLASES)")
    print(f"{'='*70}")
    
    all_X = []
    all_y = []
    all_runs = []
    all_subjects = []
    channel_counts = []
    
    subject_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
    
    for subject_dir in sorted(subject_dirs):
        subject = subject_dir.name
        
        # Cargar datos de manos (left_hand=0, right_hand=1)
        hands_dir = subject_dir / 'hands'
        if hands_dir.exists():
            for npz_file in sorted(hands_dir.glob('*.npz')):
                data = np.load(npz_file)
                X = data['X']
                y = data['y']  # Ya vienen como 0 (left) y 1 (right)
                run = str(data['run'])
                
                all_X.append(X)
                all_y.append(y)
                all_runs.extend([run] * len(y))
                all_subjects.extend([subject] * len(y))
                channel_counts.append(X.shape[1])
                
                print(f"  ✓ {subject}/{run} (hands): {len(y)} epochs, {X.shape[1]} channels")
        
        # Cargar datos de puños/pies (both_hands=2, both_feet=3)
        fists_dir = subject_dir / 'fists_feet'
        if fists_dir.exists():
            for npz_file in sorted(fists_dir.glob('*.npz')):
                data = np.load(npz_file)
                X = data['X']
                y = data['y'] + 2  # Shift: 0->2 (both_hands), 1->3 (both_feet)
                run = str(data['run'])
                
                all_X.append(X)
                all_y.append(y)
                all_runs.extend([run] * len(y))
                all_subjects.extend([subject] * len(y))
                channel_counts.append(X.shape[1])
                
                print(f"  ✓ {subject}/{run} (fists_feet): {len(y)} epochs, {X.shape[1]} channels")
    
    # Recortar al mínimo de canales y tiempo
    min_channels = min(channel_counts)
    time_lengths = [X.shape[2] for X in all_X]
    min_time = min(time_lengths)
    
    print(f"\n⚠️  Recortando todos a {min_channels} canales y {min_time} muestras de tiempo")
    all_X_trimmed = [X[:, :min_channels, :min_time] for X in all_X]
    
    # Concatenar
    X = np.concatenate(all_X_trimmed, axis=0)
    y = np.concatenate(all_y, axis=0)
    run_labels = np.array(all_runs)
    subject_labels = np.array(all_subjects)
    
    print(f"\n📊 DATOS CARGADOS:")
    print(f"  Total epochs: {len(y)}")
    print(f"  Shape: {X.shape}")
    print(f"  Sujetos únicos: {len(np.unique(subject_labels))}")
    print(f"  Runs únicos: {np.unique(run_labels)}")
    print(f"  Distribución de clases:")
    for class_id, class_name in enumerate(['Mano Izq', 'Mano Der', 'Ambas Manos', 'Ambos Pies']):
        count = np.sum(y == class_id)
        print(f"    {class_name}: {count} epochs")
    
    return X, y, run_labels, subject_labels


def extract_features(X, y, n_components=6):
    """Extrae características CSP + Band Power para 4 clases"""
    print(f"\n{'='*70}")
    print("EXTRAYENDO CARACTERÍSTICAS (4 CLASES)")
    print(f"{'='*70}")
    
    # CSP multi-clase
    print(f"  → CSP multi-clase ({n_components} componentes)...")
    csp = MultiClassCSP(n_components=n_components)
    csp.fit(X, y)
    X_csp = csp.transform(X)
    print(f"    ✓ CSP features: {X_csp.shape}")
    
    # Band Power
    print(f"  → Band Power (mu/beta)...")
    bp_extractor = BandPowerExtractor(
        sfreq=160,
        bands={'mu': (8, 13), 'beta': (13, 30)},
        relative=True
    )
    X_bp = bp_extractor.transform(X)
    print(f"    ✓ Band Power features: {X_bp.shape}")
    
    # Concatenar
    X_features = np.hstack([X_csp, X_bp])
    print(f"\n  ✓ Features finales: {X_features.shape}")
    
    return X_features, csp, bp_extractor


def train_and_evaluate(X, y, run_labels, output_dir):
    """Entrena y evalúa modelos de 4 clases"""
    print(f"\n{'='*70}")
    print("ENTRENAMIENTO Y VALIDACIÓN (4 CLASES)")
    print(f"{'='*70}")
    
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
        
        # Validación LORO
        print(f"  Ejecutando Leave-One-Run-Out CV...")
        cv_results = cross_validate_loro(model, X, y, run_labels, return_predictions=True)
        
        # Concatenar predicciones
        y_true_all = np.concatenate(cv_results['true_labels'])
        y_pred_all = np.concatenate(cv_results['predictions'])
        
        # Métricas
        metrics = ClassificationMetrics(y_true_all, y_pred_all)
        metrics_dict = metrics.compute_all()
        
        print(f"\n  📊 RESULTADOS {model_name}:")
        print(f"    Accuracy:  {metrics_dict['accuracy']:.4f}")
        print(f"    Precision: {metrics_dict['precision']:.4f}")
        print(f"    Recall:    {metrics_dict['recall']:.4f}")
        print(f"    F1-Score:  {metrics_dict['f1_score']:.4f}")
        print(f"    Kappa:     {metrics_dict['kappa']:.4f}")
        
        results[model_name] = {
            'metrics': metrics_dict,
            'fold_scores': cv_results['scores'],
            'y_true': y_true_all,
            'y_pred': y_pred_all
        }
        
        # Entrenar modelo final
        print(f"  → Entrenando modelo final...")
        model.fit(X, y)
        
        # Guardar
        model_file = output_dir / f"4class_{model_name.lower()}.pkl"
        joblib.dump(model, model_file)
        print(f"  💾 Guardado: {model_file}")
    
    return results


def main():
    """Pipeline completo"""
    print("="*70)
    print("ENTRENAMIENTO - MODELO DE 4 CLASES")
    print("="*70)
    
    # Configuración
    config = load_config('config.yaml')
    processed_dir = Path(config['data']['processed_path'])
    output_dir = Path(config['output']['models'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos
    X, y, run_labels, subject_labels = load_all_data(processed_dir)
    
    # Extraer características
    X_features, csp, bp_extractor = extract_features(X, y, n_components=6)
    
    # Guardar extractores de características
    joblib.dump(csp, output_dir / '4class_csp.pkl')
    joblib.dump(bp_extractor, output_dir / '4class_bandpower.pkl')
    print(f"\n💾 Extractores guardados")
    
    # Entrenar modelos
    results = train_and_evaluate(X_features, y, run_labels, output_dir)
    
    # Resumen
    print(f"\n{'='*70}")
    print("📊 RESUMEN FINAL - 4 CLASES")
    print(f"{'='*70}")
    print("\nClases: 0=Mano Izq, 1=Mano Der, 2=Ambas Manos, 3=Ambos Pies\n")
    
    for model_name, result in results.items():
        acc = result['metrics']['accuracy']
        f1 = result['metrics']['f1_score']
        print(f"  {model_name:15s} → Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    # Guardar resumen
    summary_file = output_dir / 'training_summary_4class.txt'
    with open(summary_file, 'w') as f:
        f.write("TRAINING SUMMARY - 4 Class Motor Imagery\n")
        f.write("="*70 + "\n\n")
        f.write("Classes:\n")
        f.write("  0: Mano Izquierda\n")
        f.write("  1: Mano Derecha\n")
        f.write("  2: Ambas Manos\n")
        f.write("  3: Ambos Pies\n\n")
        
        for model_name, result in results.items():
            f.write(f"\n{model_name}:\n")
            for metric, value in result['metrics'].items():
                if metric not in ['confusion_matrix', 'classification_report']:
                    f.write(f"  {metric}: {value:.4f}\n")
    
    print(f"\n💾 Resumen guardado: {summary_file}")
    print(f"\n✅ ENTRENAMIENTO COMPLETADO")
    print(f"📁 Modelos guardados en: {output_dir.absolute()}")
    print("="*70)


if __name__ == "__main__":
    main()
