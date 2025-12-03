"""
Descargador de datos PhysioNet EEG Motor Movement/Imagery
Descarga automáticamente las runs R04, R06, R08, R10, R12, R14 para sujetos especificados
"""

import mne
from pathlib import Path

# Configuración
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Runs de imaginación motora
RUNS = [4, 6, 8, 10, 12, 14]

# Sujetos a descargar (modifica esta lista según necesites)
SUBJECTS = list(range(50,110))  # Puedes agregar más: [1, 2, 3, 4, 5]

print("="*60)
print("DESCARGA DE DATOS - PhysioNet Motor Imagery")
print("="*60)
print(f"Directorio destino: {DATA_DIR.absolute()}")
print(f"Sujetos: {SUBJECTS}")
print(f"Runs: {RUNS}")
print("="*60)

for subject in SUBJECTS:
    print(f"\n▶ Descargando sujeto S{subject:03d}...")
    
    subject_dir = DATA_DIR / f"S{subject:03d}"
    subject_dir.mkdir(exist_ok=True)
    
    for run in RUNS:
        try:
            # Verificar si ya existe
            edf_file = subject_dir / f"S{subject:03d}R{run:02d}.edf"
            if edf_file.exists():
                print(f"  ⊙ S{subject:03d}R{run:02d} - Ya existe, omitiendo")
                continue
            
            # Descargar usando MNE
            print(f"  ↓ Descargando S{subject:03d}R{run:02d}...", end=" ", flush=True)
            
            raw = mne.io.read_raw_edf(
                mne.datasets.eegbci.load_data(subject, run, path=str(DATA_DIR.parent))[0],
                preload=False,
                verbose=False
            )
            
            print("✓")
            
        except Exception as e:
            print(f"✗ Error: {e}")

print("\n" + "="*60)
print("✅ DESCARGA COMPLETA")
print(f"Archivos guardados en: {DATA_DIR.absolute()}")
print("="*60)