"""
Script para limpiar y renombrar archivos del dataset con caracteres especiales
"""

import os
from pathlib import Path
import shutil
import unicodedata

def clean_filename(filename):
    """
    Limpia un nombre de archivo removiendo caracteres especiales
    """
    # Normalizar caracteres Unicode (é -> e, ñ -> n, etc.)
    normalized = unicodedata.normalize('NFKD', filename)
    ascii_string = normalized.encode('ASCII', 'ignore').decode('ASCII')
    
    # Reemplazar espacios por guiones bajos
    ascii_string = ascii_string.replace(' ', '_')
    
    # Remover caracteres no alfanuméricos excepto . _ -
    clean = ''.join(c for c in ascii_string if c.isalnum() or c in '._-')
    
    return clean

def clean_dataset():
    """
    Limpia todos los archivos del dataset
    """
    dataset_paths = [
        Path("data/dataset/DATASET_IA"),
        Path("data/dataset/datasetCompleto"),
        Path("data/dataset/DatasetCompleto2")
    ]
    
    print("\n" + "="*60)
    print("🧹 LIMPIANDO DATASET")
    print("="*60)
    
    total_renamed = 0
    total_errors = 0
    total_files = 0
    
    for dataset_path in dataset_paths:
        if not dataset_path.exists():
            print(f"\n⚠️  No existe: {dataset_path}")
            continue
        
        print(f"\n📁 Procesando: {dataset_path.name}")
        
        # Buscar todos los archivos
        files_in_folder = 0
        for file_path in dataset_path.rglob('*'):
            if not file_path.is_file():
                continue
            
            files_in_folder += 1
            total_files += 1
            
            original_name = file_path.name
            
            # Verificar si tiene caracteres problemáticos
            needs_cleaning = False
            try:
                original_name.encode('ascii')
                # Si funciona, no hay problema
            except UnicodeEncodeError:
                # Tiene caracteres especiales
                needs_cleaning = True
            
            if not needs_cleaning:
                continue
            
            # Limpiar nombre
            clean_name = clean_filename(original_name)
            
            if clean_name != original_name:
                new_path = file_path.parent / clean_name
                
                # Verificar si ya existe
                if new_path.exists():
                    # Agregar número al final
                    stem = new_path.stem
                    suffix = new_path.suffix
                    counter = 1
                    while new_path.exists():
                        new_path = file_path.parent / f"{stem}_{counter}{suffix}"
                        counter += 1
                    clean_name = new_path.name
                
                try:
                    # Renombrar
                    file_path.rename(new_path)
                    total_renamed += 1
                    
                    if total_renamed <= 10:  # Mostrar primeros 10
                        print(f"   ✅ Renombrado:")
                        print(f"      {original_name}")
                        print(f"      → {clean_name}")
                    elif total_renamed == 11:
                        print(f"   ... (mostrando solo primeros 10)")
                
                except Exception as e:
                    total_errors += 1
                    if total_errors <= 5:
                        print(f"   ❌ Error: {original_name}")
                        print(f"      {str(e)}")
        
        print(f"   Total archivos en carpeta: {files_in_folder}")
    
    print(f"\n{'='*60}")
    print("📊 RESUMEN")
    print(f"{'='*60}")
    print(f"📁 Total de archivos escaneados: {total_files}")
    print(f"✅ Archivos renombrados: {total_renamed}")
    print(f"❌ Errores: {total_errors}")
    print(f"{'='*60}\n")
    
    if total_renamed > 0:
        print("✅ Limpieza completada exitosamente")
        print("\n🎯 Próximos pasos:")
        print("   1. Verifica las estadísticas: python main.py stats")
        print("   2. Entrena el modelo: python main.py train")
    elif total_files == 0:
        print("⚠️  No se encontraron archivos en los datasets")
        print("   Verifica que las carpetas existan:")
        print("   - data/dataset/DATASET_IA/")
        print("   - data/dataset/datasetCompleto/")
        print("   - data/dataset/DatasetCompleto2/")
    else:
        print("✅ No se encontraron archivos con caracteres especiales")
        print("   Puedes proceder directamente al entrenamiento:")
        print("   python main.py train")

if __name__ == "__main__":
    clean_dataset()