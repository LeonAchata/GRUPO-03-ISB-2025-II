"""
Script de análisis y visualización para resultados del poster
Genera matrices de confusión, gráficos de rendimiento, y análisis estadísticos
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import sys
from scipy import stats

sys.path.append('.')

from src.features.csp import CSPExtractor, MultiClassCSP
from src.features.spectral import BandPowerExtractor
from src.validation.metrics import ClassificationMetrics
from src.validation.cross_validation import cross_validate_loro
from src.utils.config import load_config

# Configuración de estilo para figuras del poster
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def load_data_and_features(processed_dir, task_type):
    """Carga datos y extrae características"""
    print(f"\n📁 Cargando datos: {task_type}")
    
    all_X, all_y, all_runs, all_subjects = [], [], [], []
    channel_counts = []
    
    subject_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
    
    for subject_dir in sorted(subject_dirs):
        task_dir = subject_dir / task_type
        if not task_dir.exists():
            continue
        
        subject = subject_dir.name
        npz_files = sorted(task_dir.glob('*.npz'))
        
        for npz_file in npz_files:
            data = np.load(npz_file)
            X, y, run = data['X'], data['y'], str(data['run'])
            all_X.append(X)
            all_y.append(y)
            all_runs.extend([run] * len(y))
            all_subjects.extend([subject] * len(y))
            channel_counts.append(X.shape[1])
    
    # Recortar al mínimo de canales
    min_channels = min(channel_counts)
    all_X_trimmed = [X[:, :min_channels, :] for X in all_X]
    
    X = np.concatenate(all_X_trimmed, axis=0)
    y = np.concatenate(all_y, axis=0)
    run_labels = np.array(all_runs)
    subject_labels = np.array(all_subjects)
    
    # Extraer características
    print(f"  → Extrayendo CSP + Band Power...")
    n_classes = len(np.unique(y))
    
    if n_classes == 2:
        csp = CSPExtractor(n_components=6)
    else:
        csp = MultiClassCSP(n_components=6)
    
    csp.fit(X, y)
    X_csp = csp.transform(X)
    
    bp_extractor = BandPowerExtractor(sfreq=160, bands={'mu': (8, 13), 'beta': (13, 30)})
    X_bp = bp_extractor.transform(X)
    
    X_features = np.hstack([X_csp, X_bp])
    
    print(f"  ✓ {len(y)} epochs, {X_features.shape[1]} features")
    
    return X_features, y, run_labels, subject_labels


def plot_confusion_matrices(models_dir, output_dir, task_type, task_name, class_names):
    """Genera matrices de confusión para todos los modelos"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    model_names = ['LDA', 'SVM_RBF', 'SVM_Linear', 'RandomForest']
    
    for idx, model_name in enumerate(model_names):
        model_file = models_dir / f"{task_type}_{model_name.lower()}.pkl"
        
        if not model_file.exists():
            continue
        
        # Cargar modelo
        model = joblib.load(model_file)
        
        # Cargar datos
        config = load_config('config.yaml')
        processed_dir = Path(config['data']['processed_path'])
        X, y, run_labels, _ = load_data_and_features(processed_dir, task_type)
        
        # Validación cruzada
        cv_results = cross_validate_loro(model, X, y, run_labels, return_predictions=True)
        y_true = np.concatenate(cv_results['true_labels'])
        y_pred = np.concatenate(cv_results['predictions'])
        
        # Matriz de confusión
        metrics = ClassificationMetrics(y_true, y_pred)
        cm = metrics.conf_matrix()
        
        # Normalizar por fila (recall)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plotear
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[idx], cbar_kws={'label': 'Proporción'},
                   vmin=0, vmax=1)
        
        acc = metrics.accuracy()
        axes[idx].set_title(f'{model_name}\nAccuracy: {acc:.3f}', fontweight='bold')
        axes[idx].set_ylabel('Clase Real')
        axes[idx].set_xlabel('Clase Predicha')
    
    plt.suptitle(f'Matrices de Confusión - {task_name}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_dir / f'confusion_matrices_{task_type}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  💾 Guardado: {output_file}")
    plt.close()


def plot_model_comparison(models_dir, output_dir):
    """Compara el rendimiento de todos los modelos"""
    
    config = load_config('config.yaml')
    processed_dir = Path(config['data']['processed_path'])
    
    results = {'hands': {}, 'fists_feet': {}}
    model_names = ['LDA', 'SVM_RBF', 'SVM_Linear', 'RandomForest']
    
    for task_type in ['hands', 'fists_feet']:
        X, y, run_labels, _ = load_data_and_features(processed_dir, task_type)
        
        for model_name in model_names:
            model_file = models_dir / f"{task_type}_{model_name.lower()}.pkl"
            if not model_file.exists():
                continue
            
            model = joblib.load(model_file)
            cv_results = cross_validate_loro(model, X, y, run_labels, return_predictions=True)
            
            y_true = np.concatenate(cv_results['true_labels'])
            y_pred = np.concatenate(cv_results['predictions'])
            
            metrics = ClassificationMetrics(y_true, y_pred)
            metrics_dict = metrics.compute_all()
            
            results[task_type][model_name] = {
                'accuracy': metrics_dict['accuracy'],
                'precision': metrics_dict['precision'],
                'recall': metrics_dict['recall'],
                'f1_score': metrics_dict['f1_score'],
                'fold_scores': cv_results['scores']
            }
    
    # Gráfico de barras comparativo
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(model_names))
    width = 0.2
    
    for task_idx, (task_type, task_name) in enumerate([('hands', 'Manos (Izq/Der)'), 
                                                        ('fists_feet', 'Puños/Pies')]):
        ax = axes[task_idx]
        
        for metric_idx, metric in enumerate(metrics_to_plot):
            values = [results[task_type][m][metric] for m in model_names]
            offset = width * (metric_idx - 1.5)
            ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Modelo', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(f'{task_name}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.suptitle('Comparación de Rendimiento por Modelo', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / 'model_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  💾 Guardado: {output_file}")
    plt.close()
    
    return results


def plot_fold_variability(models_dir, output_dir):
    """Muestra variabilidad entre folds de validación cruzada"""
    
    config = load_config('config.yaml')
    processed_dir = Path(config['data']['processed_path'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    model_names = ['LDA', 'SVM_RBF', 'SVM_Linear', 'RandomForest']
    
    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        
        for task_type, color, label in [('hands', 'blue', 'Manos'), 
                                        ('fists_feet', 'green', 'Puños/Pies')]:
            model_file = models_dir / f"{task_type}_{model_name.lower()}.pkl"
            if not model_file.exists():
                continue
            
            X, y, run_labels, _ = load_data_and_features(processed_dir, task_type)
            model = joblib.load(model_file)
            cv_results = cross_validate_loro(model, X, y, run_labels, return_predictions=True)
            
            fold_scores = cv_results['scores']
            folds = range(1, len(fold_scores) + 1)
            
            ax.plot(folds, fold_scores, 'o-', color=color, label=label, linewidth=2, markersize=8)
            ax.axhline(np.mean(fold_scores), color=color, linestyle='--', alpha=0.5,
                      label=f'{label} (mean: {np.mean(fold_scores):.3f})')
        
        ax.set_xlabel('Fold', fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title(f'{model_name}', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.suptitle('Variabilidad de Accuracy por Fold (LORO CV)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / 'fold_variability.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  💾 Guardado: {output_file}")
    plt.close()


def generate_summary_table(results, output_dir):
    """Genera tabla resumen de resultados"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Preparar datos
    rows = []
    for task_type, task_name in [('hands', 'Manos (Izq/Der)'), ('fists_feet', 'Puños/Pies')]:
        rows.append([f'--- {task_name} ---', '', '', '', ''])
        
        for model_name in ['LDA', 'SVM_RBF', 'SVM_Linear', 'RandomForest']:
            if model_name not in results[task_type]:
                continue
            
            r = results[task_type][model_name]
            rows.append([
                model_name,
                f"{r['accuracy']:.4f}",
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1_score']:.4f}"
            ])
    
    headers = ['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center',
                    loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Estilo
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i, row in enumerate(rows):
        if '---' in row[0]:
            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor('#E0E0E0')
                table[(i+1, j)].set_text_props(weight='bold')
    
    plt.title('Resumen de Resultados - Clasificación Motor Imagery', 
             fontsize=14, fontweight='bold', pad=20)
    
    output_file = output_dir / 'results_table.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  💾 Guardado: {output_file}")
    plt.close()


def statistical_analysis(results, output_dir):
    """Análisis estadístico de significancia"""
    
    print("\n📊 ANÁLISIS ESTADÍSTICO")
    print("="*70)
    
    with open(output_dir / 'statistical_analysis.txt', 'w') as f:
        f.write("ANÁLISIS ESTADÍSTICO - Motor Imagery Classification\n")
        f.write("="*70 + "\n\n")
        
        for task_type, task_name in [('hands', 'Manos (Izq/Der)'), 
                                     ('fists_feet', 'Puños/Pies')]:
            f.write(f"\n{task_name}\n")
            f.write("-"*70 + "\n")
            
            # Comparar modelos con test de Friedman
            model_scores = []
            model_names = []
            
            for model_name in ['LDA', 'SVM_RBF', 'SVM_Linear', 'RandomForest']:
                if model_name in results[task_type]:
                    model_scores.append(results[task_type][model_name]['fold_scores'])
                    model_names.append(model_name)
            
            if len(model_scores) > 2:
                statistic, p_value = stats.friedmanchisquare(*model_scores)
                f.write(f"\nFriedman Test (diferencias entre modelos):\n")
                f.write(f"  Estadístico: {statistic:.4f}\n")
                f.write(f"  P-value: {p_value:.4f}\n")
                
                if p_value < 0.05:
                    f.write(f"  ✓ Hay diferencias significativas entre modelos (p < 0.05)\n")
                else:
                    f.write(f"  ✗ No hay diferencias significativas (p >= 0.05)\n")
            
            # Estadísticas por modelo
            f.write(f"\nEstadísticas por modelo:\n")
            for model_name in model_names:
                scores = results[task_type][model_name]['fold_scores']
                f.write(f"\n{model_name}:\n")
                f.write(f"  Media: {np.mean(scores):.4f}\n")
                f.write(f"  Desv. Std: {np.std(scores):.4f}\n")
                f.write(f"  Min: {np.min(scores):.4f}\n")
                f.write(f"  Max: {np.max(scores):.4f}\n")
                f.write(f"  IC 95%: [{np.percentile(scores, 2.5):.4f}, {np.percentile(scores, 97.5):.4f}]\n")
    
    print(f"  💾 Análisis guardado: {output_dir / 'statistical_analysis.txt'}")


def main():
    """Pipeline completo de visualización"""
    
    print("="*70)
    print("GENERACIÓN DE VISUALIZACIONES PARA POSTER")
    print("="*70)
    
    config = load_config('config.yaml')
    models_dir = Path(config['output']['models'])
    output_dir = Path(config['output']['figures'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Matrices de confusión
    print("\n📊 Generando matrices de confusión...")
    plot_confusion_matrices(models_dir, output_dir, 'hands', 
                           'Clasificación de Manos (Izquierda vs Derecha)',
                           ['Izquierda', 'Derecha'])
    
    plot_confusion_matrices(models_dir, output_dir, 'fists_feet',
                           'Clasificación Puños/Pies (Ambas Manos vs Ambos Pies)',
                           ['Ambas Manos', 'Ambos Pies'])
    
    # 2. Comparación de modelos
    print("\n📊 Comparando modelos...")
    results = plot_model_comparison(models_dir, output_dir)
    
    # 3. Variabilidad entre folds
    print("\n📊 Analizando variabilidad entre folds...")
    plot_fold_variability(models_dir, output_dir)
    
    # 4. Tabla resumen
    print("\n📊 Generando tabla resumen...")
    generate_summary_table(results, output_dir)
    
    # 5. Análisis estadístico
    print("\n📊 Realizando análisis estadístico...")
    statistical_analysis(results, output_dir)
    
    print("\n" + "="*70)
    print("✅ VISUALIZACIONES COMPLETADAS")
    print(f"📁 Figuras guardadas en: {output_dir.absolute()}")
    print("="*70)
    
    print("\n📋 ARCHIVOS GENERADOS:")
    print("  1. confusion_matrices_hands.png - Matrices de confusión manos")
    print("  2. confusion_matrices_fists_feet.png - Matrices de confusión puños/pies")
    print("  3. model_comparison.png - Comparación de rendimiento")
    print("  4. fold_variability.png - Variabilidad de validación cruzada")
    print("  5. results_table.png - Tabla resumen de métricas")
    print("  6. statistical_analysis.txt - Análisis estadístico completo")


if __name__ == "__main__":
    main()
