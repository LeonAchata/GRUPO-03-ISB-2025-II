"""
Script de preprocesamiento automático para todos los sujetos y runs
Procesa todos los datos disponibles en data/raw y los guarda listos para entrenar
"""

import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Agregar src al path
sys.path.append('.')

from src.data.loader import EEGLoader
from src.data.parser import EventParser
from src.preprocessing.filters import SignalFilter
from src.preprocessing.artifacts import ArtifactRemover
from src.preprocessing.segmentation import EpochExtractor
from src.preprocessing.normalization import DataNormalizer
from src.utils.config import load_config

def preprocess_subject_run(loader, parser, config, subject, run):
    """
    Procesa un sujeto y run específico.
    Retorna X, y, channels si exitoso, None si falla.
    """
    try:
        # Parámetros
        BANDPASS_LOW = config['preprocessing']['bandpass']['low']
        BANDPASS_HIGH = config['preprocessing']['bandpass']['high']
        NOTCH_FREQ = config['preprocessing']['notch']['freq']
        TMIN = config['preprocessing']['epoch']['tmin']
        TMAX = config['preprocessing']['epoch']['tmax']
        BASELINE = tuple(config['preprocessing']['epoch']['baseline'])
        
        print(f"  → Cargando datos...")
        raw = loader.load_raw(subject, run)
        
        print(f"  → Filtrando ({BANDPASS_LOW}-{BANDPASS_HIGH} Hz, notch {NOTCH_FREQ} Hz)...")
        signal_filter = SignalFilter(BANDPASS_LOW, BANDPASS_HIGH, NOTCH_FREQ)
        raw_filtered = signal_filter.filter_data(raw, apply_notch=True)
        
        print(f"  → Removiendo artefactos (ICA)...")
        artifact_remover = ArtifactRemover(threshold=999999)  # Sin rechazo por amplitud
        raw_clean = artifact_remover.clean_data(raw_filtered, apply_ica=True, interpolate_bads=True)
        
        print(f"  → Extrayendo eventos...")
        events, event_id = parser.parse_events_from_raw(raw_clean)
        events_mapped, class_labels = parser.map_events_to_classes(events, event_id, run)
        
        print(f"  → Creando epochs [{TMIN}, {TMAX}] s...")
        epoch_extractor = EpochExtractor(TMIN, TMAX, BASELINE)
        epochs = epoch_extractor.create_epochs_from_classes(
            raw_clean, events_mapped, class_labels, exclude_rest=True
        )
        
        if len(epochs) == 0:
            print(f"  ✗ No hay epochs válidos")
            return None, None, None
        
        print(f"  → Normalizando (z-score)...")
        normalizer = DataNormalizer('zscore')
        epochs_norm = normalizer.normalize_epochs(epochs)
        
        print(f"  → Extrayendo datos...")
        X, y = epoch_extractor.get_epoch_data(epochs_norm)
        
        print(f"  ✓ {X.shape[0]} epochs procesados")
        return X, y, epochs_norm.ch_names
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None, None, None


def main():
    """
    Procesa todos los sujetos y runs disponibles en data/raw
    """
    print("="*70)
    print("PREPROCESAMIENTO AUTOMÁTICO - EEG Motor Imagery")
    print("="*70)
    
    # Cargar configuración
    config = load_config('config.yaml')
    DATA_DIR = config['data']['raw_path']
    OUTPUT_DIR = Path(config['data']['processed_path'])
    IMAGERY_RUNS = config['subjects']['runs_of_interest']
    
    # Inicializar componentes
    loader = EEGLoader(DATA_DIR)
    parser = EventParser(DATA_DIR)
    
    # Descubrir sujetos disponibles
    print(f"\n📁 Escaneando directorio: {DATA_DIR}")
    available_subjects = loader.list_available_subjects()
    print(f"✓ Encontrados {len(available_subjects)} sujetos: {available_subjects[:10]}...")
    
    # Estadísticas
    total_processed = 0
    total_failed = 0
    results = {}
    
    # Procesar cada sujeto
    for subject in available_subjects:
        print(f"\n{'='*70}")
        print(f"📊 PROCESANDO: {subject}")
        print(f"{'='*70}")
        
        subject_results = {}
        
        # Descubrir runs disponibles para este sujeto
        available_runs = loader.list_available_runs(subject)
        # Filtrar solo runs de imaginación motora
        motor_imagery_runs = [r for r in available_runs if r in IMAGERY_RUNS]
        
        if not motor_imagery_runs:
            print(f"⊘ No hay runs de imaginación motora disponibles")
            total_failed += 1
            continue
        
        print(f"Runs disponibles: {motor_imagery_runs}")
        
        # Procesar cada run
        for run in motor_imagery_runs:
            print(f"\n  📝 {subject} - {run}")
            
            X, y, channels = preprocess_subject_run(loader, parser, config, subject, run)
            
            if X is not None:
                # Determinar tipo de tarea
                if run in ['R04', 'R08', 'R12']:
                    task_type = 'hands'  # left_hand vs right_hand
                else:  # R06, R10, R14
                    task_type = 'fists_feet'  # both_hands vs both_feet
                
                # Organizar en carpetas por tipo de tarea
                output_path = OUTPUT_DIR / subject / task_type
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Guardar
                output_file = output_path / f"{subject}_{run}.npz"
                np.savez_compressed(
                    output_file,
                    X=X,
                    y=y,
                    channels=channels,
                    subject=subject,
                    run=run,
                    task_type=task_type
                )
                
                print(f"  💾 Guardado: {output_file}")
                
                subject_results[run] = {
                    'n_epochs': X.shape[0],
                    'n_channels': X.shape[1],
                    'n_times': X.shape[2],
                    'task_type': task_type,
                    'file': str(output_file)
                }
                total_processed += 1
            else:
                total_failed += 1
        
        if subject_results:
            results[subject] = subject_results
    
    # Resumen final
    print(f"\n{'='*70}")
    print("📈 RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"✓ Sujetos procesados: {len(results)}/{len(available_subjects)}")
    print(f"✓ Runs exitosos: {total_processed}")
    print(f"✗ Runs fallidos: {total_failed}")
    
    # Crear archivo de índice
    index_file = OUTPUT_DIR / 'dataset_index.txt'
    with open(index_file, 'w') as f:
        f.write("DATASET INDEX - Preprocessed EEG Motor Imagery\n")
        f.write("="*70 + "\n\n")
        
        for subject, runs in results.items():
            f.write(f"{subject}:\n")
            for run, info in runs.items():
                f.write(f"  {run}: {info['n_epochs']} epochs, ")
                f.write(f"{info['n_channels']} channels, ")
                f.write(f"task={info['task_type']}\n")
            f.write("\n")
    
    print(f"\n💾 Índice guardado: {index_file}")
    
    # Estadísticas por tipo de tarea
    hands_total = 0
    fists_feet_total = 0
    
    for subject_data in results.values():
        for run_data in subject_data.values():
            if run_data['task_type'] == 'hands':
                hands_total += run_data['n_epochs']
            else:
                fists_feet_total += run_data['n_epochs']
    
    print(f"\n📊 EPOCHS POR TIPO DE TAREA:")
    print(f"  Manos (izq/der): {hands_total} epochs")
    print(f"  Puños/Pies: {fists_feet_total} epochs")
    print(f"  Total: {hands_total + fists_feet_total} epochs")
    
    print(f"\n✅ PREPROCESAMIENTO COMPLETADO")
    print(f"📁 Datos listos en: {OUTPUT_DIR.absolute()}")
    print("="*70)


if __name__ == "__main__":
    main()
